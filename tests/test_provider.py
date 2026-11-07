"""Offline provider checks through its public lifecycle and tool methods."""

import contextvars
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "uniichat_test_plugin", root / "__init__.py", submodule_search_locations=[str(root)],
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.provider = plugin.UniichatMemoryProvider({
            "root": self.temporary.name, "summarize": False,
        })

    def test_unindexed_raw_recall_preserves_late_details_and_provenance_without_api(self):
        self.provider.store.append("note", "a known old record")
        self.provider.store.build_tree()
        self.provider.store.append(
            "note", "padding " * 100 + "exactreceiptneedle tests passed", session_id="agent-two",
        )
        with patch.object(plugin, "_chat", side_effect=AssertionError("no API needed")):
            result = json.loads(self.provider.handle_tool_call("uniichat_recall", {
                "query": "exactreceiptneedle",
            }))
        self.assertEqual(result["mode"], "raw")
        self.assertIn("exactreceiptneedle tests passed", result["messages"][0]["text"])
        self.assertEqual(result["messages"][0]["session_id"], "agent-two")

    def test_sync_stores_full_turn_not_a_silent_30000_character_prefix(self):
        user = "u" * 31_000 + "user end"
        assistant = "a" * 31_000 + "assistant end"
        with patch.object(self.provider, "_ensure_tree_async"):
            self.provider.sync_turn(user, assistant, session_id="turn-owner")
        records = self.provider.store.messages()
        self.assertEqual([record["text"] for record in records], [user, assistant])
        self.assertTrue(all(record["session_id"] == "turn-owner" for record in records))

    def test_raw_recall_response_is_bounded_and_marks_excerpt(self):
        self.provider.store.append("note", "p" * 40_000 + "tailneedle evidence")
        result = json.loads(self.provider.handle_tool_call("uniichat_recall", {"query": "tailneedle"}))
        record = result["messages"][0]
        self.assertLessEqual(len(record["text"]), plugin.TOOL_CLIP)
        self.assertTrue(record["truncated"])
        self.assertIn("tailneedle evidence", record["text"])
        self.assertGreater(record["excerpt_start"], 0)

    def test_background_builder_carries_the_callers_context(self):
        scope = contextvars.ContextVar("test_profile_scope", default="unbound")
        scope.set("agent-two")
        done = threading.Event()
        observed = []

        def build(**kwargs):
            observed.append(scope.get())
            done.set()
            return 0

        self.provider.store.append("note", "build this")
        with patch.object(self.provider.store, "build_tree", side_effect=build):
            self.provider._ensure_tree_async()
            self.assertTrue(done.wait(5))
        self.assertEqual(observed, ["agent-two"])

    def test_credentials_use_only_the_served_profiles_secret_scope(self):
        with patch.object(plugin, "get_secret", return_value=None) as secret:
            self.assertIsNone(plugin._summarizer_key())
        secret.assert_called_once_with("HERMES_CUSTOM_API_SURPLUSINTELLIGENCE_AI_API_KEY")


if __name__ == "__main__":
    unittest.main()
