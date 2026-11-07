"""Offline storage regressions, never touching a real memory log or model API."""

import importlib.util
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("uniichat_store", Path(__file__).parents[1] / "store.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
UniichatStore = module.UniichatStore


def _append_notes(root):
    store = UniichatStore(Path(root), summarize=False)
    for _ in range(10):
        store.append("note", "process receipt")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "memory"
        self.store = UniichatStore(self.root, summarize=False)

    def test_fresh_tree_append_and_cross_instance_invalidation(self):
        self.store.append("note", "first record")
        self.store.build_tree()
        observer = UniichatStore(self.root, summarize=False)
        self.assertTrue(observer.tree_valid())
        observer._load_tree()
        self.store.append("note", "new record")
        self.assertFalse(observer.tree_valid())
        self.store.build_tree()
        self.assertTrue(observer.tree_valid())
        self.assertEqual(len([key for key in observer._load_tree() if key[0] == 0]), 2)

    def test_odd_cached_node_includes_new_right_child(self):
        for text in ("first", "second", "third"):
            self.store.append("note", text)
        self.store.build_tree()
        self.store.append("note", "fourthneedle")
        self.store.build_tree()
        self.assertIn("fourthneedle", self.store._load_tree()[(1, 2)]["text"])

    def test_tree_build_does_not_block_append_or_claim_a_newer_snapshot(self):
        entered, release = threading.Event(), threading.Event()

        def summarize(text):
            entered.set()
            self.assertTrue(release.wait(5))
            return "summarized"

        store = UniichatStore(self.root, summarizer=summarize)
        store.append("note", "a" * 512)
        store.append("note", "b" * 512)
        builder = threading.Thread(target=store.build_tree)
        builder.start()
        try:
            self.assertTrue(entered.wait(5))
            store.append("note", "append during summary")
        finally:
            release.set()
            builder.join(5)
        self.assertFalse(builder.is_alive())
        self.assertFalse(store.tree_valid())

    def test_shared_store_appends_have_unique_ids_and_intact_utf8(self):
        stores = [UniichatStore(self.root, summarize=False) for _ in range(4)]
        threads = [threading.Thread(target=lambda store=store: [
            store.append("note", "receipt ✓", session_id="test-session") for _ in range(20)
        ]) for store in stores]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        records = self.store.messages()
        self.assertEqual(sorted(message["i"] for message in records), list(range(80)))
        self.assertTrue(all(message["text"] == "receipt ✓" for message in records))

    def test_raw_recall_finds_unindexed_facts_beyond_old_250_char_clip(self):
        self.store.append("note", "old indexed record")
        self.store.build_tree()
        self.store.append("note", "padding " * 100 + "uniquecompactionneedle proof", session_id="agent-two")
        hits = self.store.search("uniquecompactionneedle")
        self.assertEqual(len(hits), 1)
        self.assertIn("uniquecompactionneedle proof", hits[0]["text"])
        self.assertEqual(hits[0]["session_id"], "agent-two")

    def test_common_question_words_do_not_outrank_the_requested_fact(self):
        self.store.append("note", "What did we do about the next step?")
        self.store.append("note", "compactneedle recovered the archived evidence")
        hits = self.store.search("What did we do about compactneedle?")
        self.assertEqual([hit["i"] for hit in hits], [1])

    def test_independent_processes_share_the_append_lock(self):
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_append_notes, args=(str(self.root),)) for _ in range(3)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sorted(message["i"] for message in self.store.messages()), list(range(30)))

    def test_small_and_empty_indexes_and_utf8_budget(self):
        self.store.build_tree()
        self.assertTrue(self.store.tree_valid())
        self.assertEqual(self.store.index_menu(), "")
        self.store.append("note", "small log receipt")
        self.store.build_tree()
        self.assertIn("small log receipt", self.store.index_menu())
        clipped = module.clip("✓" * 200)
        self.assertLessEqual(len(clipped.encode("utf-8")), 512)
        self.assertNotIn("\ufffd", clipped)

    def test_unchanged_nodes_do_not_call_summarizer_again(self):
        calls = []
        store = UniichatStore(self.root, summarizer=lambda text: calls.append(text) or "summarized")
        for text in ("a" * 512, "b" * 512, "c" * 512, "d" * 512):
            store.append("note", text)
        store.build_tree()
        first = len(calls)
        store.build_tree(force=True)
        self.assertEqual(len(calls), first)
        self.assertTrue(store.tree_valid())


if __name__ == "__main__":
    unittest.main()
