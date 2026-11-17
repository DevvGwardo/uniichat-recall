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

    def test_tls_context_trusts_anchors_only_and_is_cached(self):
        context = plugin._tls_context()
        self.assertEqual(context.verify_mode, plugin.ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertIs(plugin._tls_context(), context)  # built once, reused


class QueryExpansionTests(unittest.TestCase):
    """The paraphrase tier: a query whose words never reach its gold record."""

    GOLD = ("The Contabo migration is parked at the payment gate: the card "
            "bounced with insufficient_funds and instance 203758793 sits at "
            "pending_payment.")

    def _provider(self, **overrides):
        config = {"root": self.temporary.name, "summarize": False,
                  "log_misses": False, **overrides}
        provider = plugin.UniichatMemoryProvider(config)
        provider.store.append("unii", self.GOLD, session_id="fleet-audit")
        return provider

    def _chat(self, variants, answer):
        def fake(prompt, *args, **kwargs):
            if "alternative SEARCH queries" in prompt:
                return variants
            if "Raw messages from the log" in prompt:
                return answer
            raise AssertionError(f"unexpected model call: {prompt[:60]!r}")
        return fake

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def test_paraphrased_query_misses_without_expansion(self):
        provider = self._provider(query_expansion=False)
        with patch.object(plugin, "_chat", self._chat(
                "contabo migration payment gate", "irrelevant — never called")):
            result = json.loads(provider.handle_tool_call("uniichat_recall", {
                "query": "why did the move to the new host stall?"}))
        self.assertNotIn(0, result.get("message_ids") or [])
        self.assertNotIn("payment gate", json.dumps(result))
        self.assertNotEqual(result.get("mode"), "raw")

    def test_query_expansion_rescues_a_paraphrase_miss(self):
        provider = self._provider()  # query_expansion defaults on
        with patch.object(plugin, "_chat", self._chat(
                "contabo migration payment gate\ninsufficient funds card declined",
                "The Contabo migration stalled at the payment gate: the card "
                "bounced with insufficient_funds.")):
            result = json.loads(provider.handle_tool_call("uniichat_recall", {
                "query": "why did the move to the new host stall?"}))
        self.assertEqual(result["mode"], "semantic")
        self.assertIn(0, result["message_ids"])
        self.assertIn("payment gate", result["answer"])
        self.assertIn("candidates", str(result.get("block", "")))

    def test_variant_hits_never_promote_to_raw_mode(self):
        # The strong gate demands the ORIGINAL query's own terms; variants only
        # widen the semantic tier, whose answer stage still abstains.
        provider = self._provider()
        with patch.object(plugin, "_chat", self._chat(
                "contabo", "NOT FOUND")):
            result = json.loads(provider.handle_tool_call("uniichat_recall", {
                "query": "why did the move to the new host stall?"}))
        self.assertNotEqual(result["mode"], "raw")
        self.assertIn("NOT FOUND", result["answer"])

    def test_expansion_failure_falls_back_to_unchanged_behavior(self):
        provider = self._provider()
        with patch.object(plugin, "_chat", return_value=None):
            result = json.loads(provider.handle_tool_call("uniichat_recall", {
                "query": "why did the move to the new host stall?"}))
        self.assertIn(result.get("mode"), ("partial", "semantic"))
        self.assertNotIn(0, result.get("message_ids") or [])
        self.assertIn("not a verified answer", result.get("answer", ""))


class OutboundRedactionTests(unittest.TestCase):
    """Credential-shaped tokens never leave for the model endpoint."""

    def test_redact_secrets_masks_key_shapes_and_keeps_plain_text(self):
        token = "apikey_" + "a1b2c3d4e5f6a7b8c9d0e1f2"
        text = f"pasted {token} and sk-ant-1234567890abcdef but kept ordinary words"
        out = plugin._redact_secrets(text)
        self.assertNotIn(token, out)
        self.assertNotIn("sk-ant-1234567890abcdef", out)
        self.assertIn("[redacted]", out)
        self.assertIn("kept ordinary words", out)

    def test_answer_stage_material_is_redacted_before_the_api_call(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = plugin.UniichatMemoryProvider({
            "root": temporary.name, "summarize": False, "log_misses": False})
        token = "apikey_" + "a1b2c3d4e5f6a7b8c9d0e1f2"
        provider.store.append("note", f"the key {token} was pasted", session_id="s")
        captured = {}

        def fake(prompt, *args, **kwargs):
            if "alternative SEARCH queries" in prompt:
                return None
            captured["prompt"] = prompt
            return "NOT FOUND"

        with patch.object(plugin, "_chat", fake):
            provider.handle_tool_call("uniichat_recall", {"query": "key pasted there"})
        self.assertIn("prompt", captured)  # a semantic answer call happened
        self.assertNotIn(token, captured["prompt"])
        self.assertIn("[redacted]", captured["prompt"])

    def test_raw_local_recall_is_never_redacted(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = plugin.UniichatMemoryProvider({
            "root": temporary.name, "summarize": False, "log_misses": False})
        token = "apikey_" + "a1b2c3d4e5f6a7b8c9d0e1f2"
        provider.store.append("note", f"the key {token} was pasted", session_id="s")
        with patch.object(plugin, "_chat", side_effect=AssertionError("raw needs no model")):
            result = json.loads(provider.handle_tool_call("uniichat_recall", {"query": "pasted"}))
        self.assertEqual(result["mode"], "raw")
        self.assertIn(token, result["messages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
