"""Synthetic correctness and resource-budget regressions for the memory hot path."""

import json
import os
import threading
import unittest
from unittest.mock import patch

import test_provider as providers
import test_store as stores

plugin = providers.plugin
UniichatStore = stores.UniichatStore


class OptimizedStoreTests(unittest.TestCase):
    setUp = stores.StoreTests.setUp

    def test_fallback_summary_can_be_upgraded(self):
        calls = []
        store = UniichatStore(self.root, summarizer=lambda text: None)
        store.append("note", "a" * 512)
        store.append("note", "b" * 512)
        store.build_tree()
        store.summarizer = lambda text: calls.append(text) or "successful upgrade"
        store.build_tree(force=True)
        self.assertTrue(calls)
        self.assertEqual(store._load_tree()[(1, 0)]["text"], "successful upgrade")

    def test_menu_budget_covers_all_records(self):
        self.store.append_many([("note", "receipt " * 100)] * 257)
        self.store.build_tree()
        blocks = self.store.menu_blocks(max_bytes=4096)
        menu = self.store.index_menu(max_bytes=4096)
        self.assertLessEqual(len(menu.encode()), 4096)
        self.assertTrue(all(any(start <= i < start + span for start, span, _ in blocks)
                            for i in range(257)))

    def test_log_stats_follow_replacement_sparse_ids_and_partial_tail(self):
        self.store.append("note", "first")
        self.assertEqual(self.store.message_count(), 1)
        path = next((self.root / "main").glob("*.jsonl"))
        replacement = path.with_suffix(".new")
        replacement.write_text(json.dumps({"i": 40, "text": "sparse", "kind": "note"}) + "\n")
        os.replace(replacement, path)
        with path.open("a") as handle:
            handle.write('{"i":41,"text":"unfinished')
        self.assertEqual(self.store.append("note", "after interrupted writer"), 41)
        self.assertEqual([message["i"] for message in self.store.messages()], [40, 41])
        self.assertEqual(self.store.message_count(), 2)

    def test_append_does_not_materialize_history(self):
        self.store.append("note", "first")
        with patch.object(self.store, "messages", side_effect=AssertionError("full-history allocation")):
            self.assertEqual(self.store.append("note", "second"), 1)
            self.assertEqual(self.store.message_count(), 2)

    def test_cheap_tree_is_stale_for_requested_summarization(self):
        self.store.append_many([("note", "a" * 512), ("note", "b" * 512)])
        self.store.build_tree()
        upgraded = UniichatStore(self.root, summarizer=lambda text: "now summarized")
        self.assertFalse(upgraded.tree_valid())
        upgraded.build_tree()
        self.assertEqual(upgraded._load_tree()[(1, 0)]["text"], "now summarized")

    def test_legacy_left_only_parent_is_not_promoted_to_a_successful_summary(self):
        self.store.append_many([("note", "a" * 512), ("note", "b" * 512)])
        path = self.root / "tree" / "all.jsonl"
        path.parent.mkdir()
        old = [{"l": 0, "i": 0, "text": "a" * 512},
               {"l": 0, "i": 1, "text": "b" * 512},
               {"l": 1, "i": 0, "text": "a" * 512}]
        path.write_text("".join(json.dumps(record) + "\n" for record in old))
        calls = []
        upgraded = UniichatStore(self.root, summarizer=lambda text: calls.append(text) or "both inputs")
        upgraded.build_tree()
        self.assertTrue(calls)
        self.assertIn("b" * 512, calls[0])
        self.assertEqual(upgraded._load_tree()[(1, 0)]["text"], "both inputs")

    def test_model_calls_have_a_per_build_budget_and_can_progress(self):
        calls = []
        store = UniichatStore(
            self.root, summarizer=lambda text: calls.append(text) or "summarized", max_summaries=1,
        )
        store.append_many([("note", "x" * 512)] * 16)
        store.build_tree()
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(record.get("summary", False) for record in store._load_tree().values()), 1)
        store.build_tree(force=True)
        self.assertEqual(len(calls), 2)

    def test_background_build_skips_another_process_lock(self):
        with self.store._file_lock(".tree.lock"):
            competitor = UniichatStore(self.root, summarize=False)
            self.assertEqual(competitor.build_tree(blocking=False), 0)
            self.assertFalse(competitor.tree_valid())
        competitor.build_tree(blocking=False)
        self.assertTrue(competitor.tree_valid())

    def test_external_append_truncation_and_same_size_edit_refresh_stats(self):
        self.store.append("note", "old")
        self.assertEqual(self.store.message_count(), 1)
        path = next((self.root / "main").glob("*.jsonl"))
        other = UniichatStore(self.root, summarize=False)
        self.assertEqual(other.append("note", "external"), 1)
        self.assertEqual(self.store.append("note", "observed external"), 2)
        path.write_text(json.dumps({"i": 90, "text": "edit"}) + "\n")
        self.assertEqual(self.store.message_count(), 1)
        path.write_text(json.dumps({"i": 91, "text": "edit"}) + "\n")
        self.assertEqual(self.store.append("note", "after edit"), 92)

    def test_complete_record_without_newline_is_not_lost(self):
        self.store.append("note", "first")
        path = next((self.root / "main").glob("*.jsonl"))
        path.write_text(json.dumps({"i": 30, "text": "complete", "kind": "note"}))
        self.assertEqual(self.store.append("note", "next"), 31)
        self.assertEqual([record["i"] for record in self.store.messages()], [30, 31])

    def test_completed_turn_pairs_do_not_interleave(self):
        stores = [UniichatStore(self.root, summarize=False) for _ in range(3)]
        def append(store, index):
            store.append_many([("user", str(index)), ("unii", str(index))], session_id=str(index))
        threads = [threading.Thread(target=append, args=(store, index)) for index, store in enumerate(stores)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        records = self.store.messages()
        self.assertEqual([record["i"] for record in records], list(range(6)))
        self.assertTrue(all(records[index]["session_id"] == records[index + 1]["session_id"]
                            for index in range(0, 6, 2)))

    def test_failed_atomic_publication_preserves_the_previous_tree(self):
        path = self.root / "tree" / "all.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("previous tree\n")
        with patch.object(stores.module.os, "replace", side_effect=OSError("publish failed")):
            with self.assertRaises(OSError):
                self.store._write_atomic(path, "new tree\n")
        self.assertEqual(path.read_text(), "previous tree\n")
        self.assertEqual(list(path.parent.iterdir()), [path])


class OptimizedProviderTests(unittest.TestCase):
    setUp = providers.ProviderTests.setUp
    def test_unicode_phrase_and_whole_token_excerpt_offsets(self):
        for prefix, query, evidence in [
            ("ß" * 35000, "needle", "needle proof"),
            ("deployment " + "x" * 40000, "deployment receipt", "deployment receipt"),
            ("concatenate " + "x" * 40000, "cat", "cat evidence"),
            ("ıssue " * 7000, "issue", "issue receipt"),
        ]:
            with self.subTest(query=query):
                text = prefix + " " + evidence
                index = self.provider.store.append("note", text)
                result = json.loads(self.provider.handle_tool_call("uniichat_recall", {"query": query}))
                record = next(record for record in result["messages"] if record["i"] == index)
                self.assertIn(evidence, record["text"])
                start = record["excerpt_start"]
                self.assertEqual(record["text"], text[start:start + len(record["text"])])

    def test_response_bytes_and_multiple_sources_are_bounded(self):
        for character in ["✓", '"', "\x00"]:
            with self.subTest(character=character):
                self.provider.store.append("note", "budgetneedle older answer")
                self.provider.store.append("note", "budgetneedle " + character * 40000)
                response = self.provider.handle_tool_call("uniichat_recall", {"query": "budgetneedle"})
                result = json.loads(response)
                self.assertLessEqual(len(response.encode()), plugin.TOOL_CLIP)
                self.assertGreaterEqual(len(result["messages"]), 2)
                self.assertTrue(all("budgetneedle" in record["text"] for record in result["messages"]))

    def test_partial_keyword_hit_does_not_suppress_semantic_recall(self):
        self.provider.store.append("note", "The launch was cancelled due to flooding.")
        self.provider.store.append("note", "Yesterday we chose blue for the button.")
        self.provider.store.build_tree()
        # Menu-routing fallback: with candidate seeding off, a partial keyword hit
        # must still reach semantic recall (one route call, one answer call).
        router = plugin.UniichatMemoryProvider({
            "root": self.provider.store.root, "summarize": False, "candidates_first": False,
        })
        with patch.object(plugin, "_chat", side_effect=["[0+2]", "Flooding cancelled the launch."]) as chat:
            result = json.loads(router.handle_tool_call("uniichat_recall", {
                "query": "Why did we postpone release yesterday?",
            }))
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["block"], "0+2")
        self.assertIn("Flooding", result["answer"])

    def test_candidate_window_answers_without_a_tree_or_route_call(self):
        self.provider.store.append("note", "The launch was cancelled due to flooding.")
        self.provider.store.append("note", "Yesterday we chose blue for the button.")
        with patch.object(plugin, "_chat", return_value="Flooding cancelled the launch.") as chat:
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "Why did we postpone release yesterday?",
            }))
        self.assertEqual(chat.call_count, 1)  # answer only: no menu hop
        self.assertEqual(result["mode"], "semantic")
        self.assertTrue(result["block"].startswith("candidates:"))
        self.assertIn("Flooding", result["answer"])
        self.assertIn(1, result["message_ids"])

    def test_unanswered_recall_is_recorded_as_a_miss(self):
        self.provider.store.append("note", "a known record about widgets")
        log = self.provider.store.root / "misses.jsonl"

        # No keyword candidate at all: the no-tree partial is still a miss.
        with patch.object(plugin, "_chat", return_value="NOT FOUND"):
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "zzq absent topic nothing",
            }))
        self.assertEqual(result["mode"], "partial")
        misses = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(misses), 1)
        self.assertEqual(misses[0]["query"], "zzq absent topic nothing")
        self.assertEqual(misses[0]["block"], "no-tree")
        self.assertTrue(misses[0]["ts"])

        # A candidate exists but the model finds nothing: a semantic miss with a
        # candidate block recorded, which is what makes the log actionable.
        with patch.object(plugin, "_chat", return_value="NOT FOUND"):
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "widgets inventory zzq",
            }))
        self.assertEqual(result["mode"], "semantic")
        misses = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(misses), 2)
        self.assertTrue(misses[1]["block"].startswith("candidates:"))

    def test_answered_recall_writes_no_miss(self):
        self.provider.store.append("note", "the launch was cancelled due to flooding")
        result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
            "query": "flooding launch cancelled",
        }))
        self.assertEqual(result["mode"], "raw")
        self.assertFalse((self.provider.store.root / "misses.jsonl").exists())

    def test_small_route_tag_and_invalid_model_tag(self):
        self.provider.store.append("note", "launch cancelled by flooding")
        self.provider.store.build_tree()
        with patch.object(plugin, "_chat", side_effect=["[0+1]", "flooding"]):
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {"query": "release delay"}))
        self.assertEqual(result["block"], "0+1")
        with patch.object(plugin, "_chat", return_value="[999+4]") as chat:
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {"query": "release delay"}))
        self.assertNotIn("block", result)
        self.assertEqual(chat.call_count, 1)

    def test_session_switch_updates_note_provenance(self):
        self.provider.on_session_switch("new-session")
        with patch.object(self.provider, "_ensure_tree_async"):
            self.provider.handle_tool_call("uniichat_note", {"text": "new note"})
        self.assertEqual(self.provider.store.messages()[-1]["session_id"], "new-session")

    def test_prefetch_batches_updates_and_accepts_empty_valid_tree(self):
        self.provider.store.build_tree()
        with patch.object(plugin, "spawn_context_thread") as spawn:
            self.provider.prefetch("empty")
        spawn.assert_not_called()
        self.provider.store.append("note", "old indexed fact")
        self.provider.store.build_tree()
        self.provider.store.append("note", "new unindexed fact")
        with patch.object(plugin, "spawn_context_thread") as spawn:
            self.provider.prefetch("new fact")
        spawn.assert_not_called()

    def test_only_one_builder_is_spawned_before_worker_starts(self):
        self.provider.store.append("note", "new fact")
        with patch.object(plugin, "spawn_context_thread") as spawn:
            self.provider._ensure_tree_async()
            self.provider._ensure_tree_async()
        self.assertEqual(spawn.call_count, 1)

    def test_semantic_material_contains_every_cited_excerpt(self):
        self.provider.store.append_many([
            ("note", "x" * 31000), ("note", "The address is Paris"),
        ])
        self.provider.store.build_tree()
        with patch.object(plugin, "_chat", side_effect=["[0+2]", "Paris"]) as chat:
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "where is the destination?",
            }))
        prompt = chat.call_args.args[0]
        self.assertIn("The address is Paris", prompt)
        self.assertEqual(result["message_ids"], [record["i"] for record in result["messages"]])
        self.assertLessEqual(len(prompt.encode()), 22000)

    def test_adaptive_routing_zooms_before_loading_a_large_block(self):
        self.provider.store.append_many([("note", "location " * 100)] * 256)
        self.provider.store.build_tree()
        tags = []
        def chat(prompt, **kwargs):
            if "Reply ONLY with an exact listed tag" in prompt:
                match = plugin.re.search(r"\[(\d+)\+(\d+)\]", prompt)
                tags.append(tuple(map(int, match.groups())))
                return match.group()
            return "generated"
        with patch.object(plugin, "_chat", side_effect=chat):
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {"query": "journey destination"}))
        self.assertGreater(len(tags), 1)
        self.assertTrue(all(tags[index + 1][1] < tags[index][1] for index in range(len(tags) - 1)))
        self.assertLessEqual(len(result["messages"]), 12)

    def test_generated_answer_cannot_break_the_json_byte_budget(self):
        self.provider.store.append("note", "location Paris")
        self.provider.store.build_tree()
        with patch.object(plugin, "_chat", side_effect=["[0+1]", "\x00" * 30000]):
            response = self.provider.handle_tool_call("uniichat_recall", {"query": "journey destination"})
        result = json.loads(response)
        self.assertLessEqual(len(response.encode()), plugin.TOOL_CLIP)
        self.assertTrue(result["answer_truncated"])

    def test_failed_thread_start_releases_builder_guard(self):
        self.provider.store.append("note", "fact")
        with patch.object(plugin, "spawn_context_thread") as spawn:
            spawn.return_value.start.side_effect = RuntimeError("thread start failed")
            with self.assertRaises(RuntimeError):
                self.provider._ensure_tree_async()
        self.assertTrue(self.provider._builder_lock.acquire(blocking=False))
        self.provider._builder_lock.release()

    def test_partial_evidence_survives_an_unavailable_semantic_route(self):
        self.provider.store.append("note", "certificate saved at the project root")
        self.provider.store.build_tree()
        with patch.object(plugin, "_chat", return_value=None):
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "where did we put certificate?",
            }))
        self.assertEqual(result["mode"], "partial")
        self.assertIn("certificate saved", result["messages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
