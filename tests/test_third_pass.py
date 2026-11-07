"""Cold-start, fairness, exact-source continuation, and semantic-budget regressions."""

import json
import threading
import unittest
from collections import OrderedDict
from unittest.mock import patch

import test_provider as providers
import test_store as stores

plugin = providers.plugin
UniichatStore = stores.UniichatStore


class ColdIndexTests(unittest.TestCase):
    setUp = stores.StoreTests.setUp

    def test_build_streams_raw_records_instead_of_materializing_turns(self):
        self.store.append_many([("note", "large turn " * 5000)] * 20)
        with patch.object(self.store, "messages", side_effect=AssertionError("whole-history allocation")):
            self.store.build_tree()
        self.assertTrue(self.store.tree_valid())

    def test_first_index_is_available_while_the_model_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        failures = []
        def summarize(text):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release model")
            return "generated summary"
        store = UniichatStore(self.root, summarizer=summarize)
        store.append_many([("note", "a" * 512), ("note", "b" * 512)])
        def build():
            try:
                store.build_tree()
            except Exception as error:
                failures.append(error)
        worker = threading.Thread(target=build)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            observer = UniichatStore(self.root, summarize=False)
            self.assertEqual(sum(level == 0 for level, _ in observer._load_tree()), 2)
            self.assertTrue(observer.tree_valid())
            self.assertFalse(store.tree_valid())
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures)
        self.assertTrue(store.tree_valid())

    def test_failed_node_cannot_starve_other_levels(self):
        calls = []
        store = UniichatStore(self.root, summarizer=lambda text: calls.append(text), max_summaries=1)
        store.append_many([("note", letter * 512) for letter in "abcd"])
        for _ in range(3):
            store.build_tree(force=True)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(set(calls)), 3)

    def test_malformed_derived_files_recover_from_the_raw_log(self):
        self.store.append("note", "authoritative raw receipt")
        tree = self.root / "tree"
        tree.mkdir()
        (tree / "all.jsonl").write_text('null\n[]\n{"l":"bad","i":0,"text":"bad"}\n')
        (tree / "state.json").write_text("null")
        self.assertFalse(self.store.tree_valid())
        self.store.build_tree()
        self.assertEqual(self.store._load_tree()[(0, 0)]["text"], "authoritative raw receipt")
        state = json.loads((tree / "state.json").read_text())
        state["retry_after"] = "not a timestamp"
        (tree / "state.json").write_text(json.dumps(state))
        self.assertFalse(self.store.tree_valid())

    def test_menu_cache_read_and_invalidation_do_not_race(self):
        ready, cleared = threading.Event(), threading.Event()
        self.store.append_many([("note", "alpha"), ("note", "beta")])
        self.store.build_tree()
        original = self.store.index_menu()
        errors, menus = [], []
        class YieldingCache(OrderedDict):
            def __contains__(self, key):
                found = super().__contains__(key)
                if found:
                    ready.set()
                    cleared.wait(2)
                return found
            def clear(self):
                super().clear()
                cleared.set()
        self.store._menus = YieldingCache(self.store._menus)
        def invalidate():
            if not ready.wait(5):
                errors.append("reader did not start")
                return
            peer = UniichatStore(self.root, summarize=False)
            peer.append("note", "gamma")
            peer.build_tree()
            self.store._load_tree()
        def read():
            try:
                menus.append(self.store.index_menu())
            except Exception as error:
                errors.append(error)
        writer, reader = threading.Thread(target=invalidate), threading.Thread(target=read)
        writer.start()
        reader.start()
        writer.join(5)
        reader.join(5)
        self.assertFalse(writer.is_alive() or reader.is_alive())
        self.assertFalse(errors)
        self.assertEqual(menus, [original])
        self.assertIn("gamma", self.store.index_menu())

    def test_narrow_zoom_and_leaf_count_use_existing_sorted_metadata(self):
        self.store.append_many([("note", "receipt")] * 256)
        self.store.build_tree()
        tree = self.store._load_tree()
        class NoTreeScan(dict):
            def __iter__(self):
                raise AssertionError("whole-tree scan")
        class NoLevelScan(list):
            def __iter__(self):
                raise AssertionError("whole-level scan for a narrow window")
        self.store._tree = NoTreeScan(tree)
        self.store._tree_levels[0] = NoLevelScan(self.store._tree_levels[0])
        self.assertEqual(self.store.indexed_count(), 256)
        self.assertEqual([index for index, _, _ in self.store.menu_blocks(level=0, lo=100, hi=104)],
                         [100, 101, 102, 103])


class ExactEvidenceTests(unittest.TestCase):
    setUp = providers.ProviderTests.setUp

    def test_cited_record_can_be_read_in_complete_bounded_chunks_without_a_model(self):
        original = "✓\x00\"" * 18000 + "the final receipt"
        index = self.provider.store.append("note", original, session_id="source-owner")
        chunks, offset = [], 0
        with patch.object(plugin, "_chat", side_effect=AssertionError("direct source needs no model")):
            for _ in range(30):
                text = self.provider.handle_tool_call("uniichat_recall", {
                    "query": "", "message_id": index, "excerpt_start": offset,
                })
                result = json.loads(text)
                self.assertLessEqual(len(text.encode()), plugin.TOOL_CLIP)
                record = result["messages"][0]
                self.assertEqual(record["excerpt_start"], offset)
                self.assertEqual(record["session_id"], "source-owner")
                chunks.append(record["text"])
                if not record["has_more"]:
                    break
                self.assertGreater(record["next_excerpt_start"], offset)
                offset = record["next_excerpt_start"]
            else:
                self.fail("continuation did not terminate")
        self.assertTrue("".join(chunks) == original, "source chunks lost or duplicated characters")

    def test_invalid_citation_and_offset_are_rejected(self):
        index = self.provider.store.append("note", "receipt")
        for args in [
            {"query": "", "message_id": -1},
            {"query": "", "message_id": True},
            {"query": "", "message_id": 999},
            {"query": "", "message_id": index, "excerpt_start": -1},
            {"query": "", "message_id": index, "excerpt_start": 100},
        ]:
            with self.subTest(args=args):
                result = json.loads(self.provider.handle_tool_call("uniichat_recall", args))
                self.assertIn("error", result)

    def test_exhausted_semantic_budget_stops_additional_model_calls(self):
        self.provider.store.append("note", "destination Paris")
        self.provider.store.build_tree()
        clock = [0.0]
        def route(prompt, **kwargs):
            clock[0] = 100.0
            return "[0+1]"
        with patch.object(plugin.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(plugin, "_chat", side_effect=route) as chat, \
             patch.object(plugin, "_agy_verify") as verify:
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "journey location",
            }))
        self.assertEqual(chat.call_count, 1)
        verify.assert_not_called()
        self.assertEqual(result["mode"], "partial")
        self.assertTrue(result["messages"])

    def test_answer_and_verification_share_the_remaining_budget(self):
        self.provider.store.append("note", "destination Paris")
        self.provider.store.build_tree()
        clock, timeouts = [0.0], []
        def chat(prompt, **kwargs):
            timeouts.append(kwargs["timeout"])
            if len(timeouts) == 1:
                clock[0] = 80.0
                return "[0+1]"
            clock[0] = 88.0
            return "NOT FOUND"
        with patch.object(plugin.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(plugin, "_chat", side_effect=chat), \
             patch.object(plugin, "_agy_verify", return_value="Paris") as verify:
            self.provider.handle_tool_call("uniichat_recall", {"query": "journey location"})
        self.assertEqual(timeouts, [90.0, 10.0])
        self.assertEqual(verify.call_args.kwargs["timeout"], 2.0)

    def test_unavailable_answer_is_partial_evidence_not_a_generated_answer(self):
        self.provider.store.append("note", "destination Paris")
        self.provider.store.build_tree()
        with patch.object(plugin, "_chat", side_effect=["[0+1]", None]), \
             patch.object(plugin, "_agy_verify") as verify:
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "journey location",
            }))
        self.assertEqual(result["mode"], "partial")
        self.assertTrue(result["messages"])
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
