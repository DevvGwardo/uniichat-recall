"""Uniichat memory provider for Hermes — the chat log IS the memory.

Implements Taelin's UniiChat design (gist 91837951a5ce5b38f341ec1ba1df6449)
as a Hermes MemoryProvider:

- append-only jsonl log of every turn (never edited, never deleted)
- binary summary tree over the log (512-byte nodes, built once, cached)
- per-turn view: L2 index menu (wide lines) covering the whole history
- retrieval tool: route via index -> zoom to raw leaves -> answer

Storage: $HERMES_HOME/uniichat/ (main/*.jsonl, tree/all.jsonl).
Config: plugins.uniichat.summarize (bool, default true) — when false the
tree is built with cheap clipped nodes (no LLM calls).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

MAX_NODE = 512
MENU_LINE = 450          # chars per index line — keep key facts visible
MENU_LEVEL = 2           # L2 = 4-msg blocks
LEAF_WINDOW = 10         # raw messages shown around the routed block
TOOL_CLIP = 30_000

_TOK = re.compile(r"[a-zA-Z0-9_-]{4,}")


# ── surplus summarizer ──────────────────────────────────────────────────────

def _summarizer_key() -> Optional[str]:
    env = "HERMES_CUSTOM_API_SURPLUSINTELLIGENCE_AI_API_KEY"
    k = os.environ.get(env)
    if k:
        return k
    for env_file in (Path.home() / ".hermes" / ".env",
                     Path.home() / ".hermes" / "profiles" / "nub" / ".env"):
        try:
            for line in env_file.read_text().splitlines():
                if line.startswith(env + "="):
                    return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            continue
    return None


def _llm_summarize(text: str, budget: int = MAX_NODE) -> Optional[str]:
    """Summarize via surplus glm. Returns None on any failure."""
    key = _summarizer_key()
    if not key:
        return None
    prompt = (f"Compress the following chat transcript excerpt into at most {budget} bytes "
              f"of plain text. Keep concrete facts: names, paths, decisions, numbers, outcomes. "
              f"No preamble, output only the compressed text.\n\n---\n{text[:12000]}\n---")
    body = json.dumps({"model": "glm-5.3-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 2000, "temperature": 0.2}).encode()
    req = urllib.request.Request(
        "https://api.surplusintelligence.ai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read())
        out = (d["choices"][0]["message"].get("content") or "").strip()
        return out[:budget * 2] if out else None
    except Exception as exc:
        logger.debug("uniichat summarize failed: %s", exc)
        return None


# ── storage helpers ─────────────────────────────────────────────────────────

def _clip(s: str, limit: int = MAX_NODE) -> str:
    b = s.encode()
    return s if len(b) <= limit else b[:limit].decode(errors="replace")


class UniichatStore:
    """Append-only log + lazily-built summary tree + view/index."""

    def __init__(self, root: Path, summarize: bool = True):
        self.root = root
        self.summarize = summarize
        self._tree: Optional[Dict] = None
        self._tree_stamp: float = 0.0
        self._lock = threading.Lock()

    # -- log --

    def append(self, kind: str, text: str) -> int:
        main = self.root / "main"
        main.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        with self._lock:
            i = 0
            for f in sorted(main.glob("*.jsonl")):
                with open(f, "rb") as fh:
                    i += sum(1 for _ in fh)
            rec = {"i": i, "kind": kind, "text": text,
                   "size": len(text.encode()), "date": day}
            with open(main / f"{day}.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._tree_stamp = 0.0  # invalidate
        return i

    def messages(self) -> List[Dict]:
        out = []
        main = self.root / "main"
        if not main.exists():
            return out
        for f in sorted(main.glob("*.jsonl")):
            for line in open(f):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    # -- tree --

    def _tree_path(self) -> Path:
        return self.root / "tree" / "all.jsonl"

    def tree_valid(self) -> bool:
        tp = self._tree_path()
        return tp.exists() and tp.stat().st_mtime >= self._tree_stamp - 1

    def _l1_cache_path(self) -> Path:
        return self.root / "tree" / "l1cache.jsonl"

    def _load_l1_cache(self) -> Dict[int, str]:
        cache = {}
        try:
            for line in open(self._l1_cache_path()):
                d = json.loads(line)
                cache[d["i"]] = d["text"]
        except (OSError, json.JSONDecodeError):
            pass
        return cache

    def build_tree(self, force: bool = False) -> int:
        """Rebuild the summary tree. Returns node count. Skips if fresh.

        Incremental: level-1 summaries of already-seen messages are reused
        from a sidecar cache, so a rebuild only pays LLM calls for NEW
        messages (and higher levels, which are few)."""
        tp = self._tree_path()
        if not force and tp.exists() and self.tree_valid():
            return len(self._load_tree())
        msgs = self.messages()
        if not msgs:
            return 0
        l1_cache = self._load_l1_cache()
        levels = [{m["i"]: _clip(m["text"]) for m in msgs}]
        L = 0
        with self._lock:
            while len(levels[L]) > 1:
                cur = levels[L]
                keys = sorted(cur)
                nxt: Dict[int, str] = {}
                jobs = []
                for a, b in zip(keys[0::2], keys[1::2]):
                    jobs.append((a, cur[a] + "\n" + cur[b]))
                if len(keys) % 2:
                    jobs.append((keys[-1], cur[keys[-1]]))
                for a, text in jobs:
                    if len(text.encode()) > MAX_NODE and self.summarize:
                        if L == 0 and a in l1_cache:
                            nxt[a] = l1_cache[a]  # reuse — no LLM call
                            continue
                        out = _llm_summarize(text)
                        if out:
                            text = out
                    nxt[a] = _clip(text)
                levels.append(nxt)
                L += 1
            # refresh L1 cache
            if levels:
                with open(self._l1_cache_path(), "w") as f:
                    for i, text in levels[1].items() if len(levels) > 1 else []:
                        f.write(json.dumps({"i": i, "text": text},
                                           ensure_ascii=False) + "\n")
            tp.parent.mkdir(parents=True, exist_ok=True)
            with open(tp, "w") as f:
                for l, level in enumerate(levels):
                    for i, text in level.items():
                        f.write(json.dumps(
                            {"l": l, "i": i, "text": text,
                             "size": len(text.encode())}, ensure_ascii=False) + "\n")
        self._tree = None
        self._tree_stamp = time.time()
        return len(self._load_tree())

    def _load_tree(self) -> Dict:
        if self._tree is not None:
            return self._tree
        tree: Dict = {}
        tp = self._tree_path()
        if tp.exists():
            for line in open(tp):
                try:
                    d = json.loads(line)
                    tree[(d["l"], d["i"])] = d
                except json.JSONDecodeError:
                    continue
        self._tree = tree
        return tree

    # -- view --

    def index_menu(self, level: int = MENU_LEVEL, line_chars: int = MENU_LINE) -> str:
        tree = self._load_tree()
        span = 2 ** level
        nodes = sorted(k for k in tree if k[0] == level)
        return "\n".join(
            f"[{k[1]}+{span}] {tree[k]['text'][:line_chars]}" for k in nodes)

    def leaf_window(self, start: int, span: int) -> List[Dict]:
        msgs = self.messages()
        by_i = {m["i"]: m for m in msgs}
        out = []
        for i in range(max(0, start - 2), min(max(by_i) + 1 if by_i else 0, start + span + 2)):
            if i in by_i:
                out.append(by_i[i])
        return out

    def leaves(self, lo: int, hi: int) -> List[Dict]:
        return [m for m in self.messages() if lo <= m["i"] < hi]


# ── tool schemas ────────────────────────────────────────────────────────────

UNIICHAT_RECALL_SCHEMA = {
    "name": "uniichat_recall",
    "description": ("Search the persistent uniichat session log (every past turn, "
                    "summarized in a tree). Use for anything about past sessions, "
                    "decisions, or work history. Returns an answer with message ids."),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
        },
        "required": ["query"],
    },
}

UNIICHAT_APPEND_SCHEMA = {
    "name": "uniichat_note",
    "description": ("Append a note directly to the permanent uniichat log (a decision, "
                    "fact, or context worth remembering verbatim). Log lives forever."),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The note text."},
        },
        "required": ["text"],
    },
}


class UniichatMemoryProvider(MemoryProvider):
    """Memory provider: one endless chat whose log is the memory."""

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._session_id = str(uuid.uuid4())
        self._agent_context = "primary"
        self._turn = 0
        self._last_recall_count = 0
        root = Path(self._config.get("root") or
                    str(Path.home() / ".hermes" / "uniichat"))
        self.store = UniichatStore(
            root, summarize=bool(self._config.get("summarize", True)))
        self._pending: List = []
        self._builder_lock = threading.Lock()

    # -- lifecycle --

    @property
    def name(self) -> str:
        return "uniichat"

    def is_available(self) -> bool:
        try:
            self.store.root.mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else f"cannot write {self.store.root}"

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")
        if self._agent_context != "primary":
            return
        self.store.append("note", f"session start {session_id}")

    def shutdown(self) -> None:
        self._flush_pending()

    # -- background tree maintenance --

    def _ensure_tree_async(self) -> None:
        if self.store.tree_valid() and self._tree_path_has_msgs():
            return

        def _build():
            if not self._builder_lock.acquire(blocking=False):
                return
            try:
                n = self.store.build_tree()
                logger.info("uniichat tree rebuilt: %d nodes", n)
            except Exception as exc:
                logger.debug("uniichat build failed: %s", exc)
            finally:
                self._builder_lock.release()

        threading.Thread(target=_build, daemon=True, name="uniichat-tree").start()

    def _tree_path_has_msgs(self) -> bool:
        return len(self.store.messages()) > 0

    def _flush_pending(self):
        if self._pending:
            try:
                self.store.append("unii", self._pending.pop())
            except Exception:
                pass

    # -- per-turn hooks --

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._agent_context != "primary":
            return ""
        self._ensure_tree_async()
        tree = self.store._load_tree()
        if not tree:
            return ""
        menu = self.store.index_menu()
        return ("## Uniichat — persistent memory index\n"
                "Each line summarizes a few past messages. Use uniichat_recall "
                "to zoom into any block for the full story.\n\n" + menu)

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "") -> None:
        if self._agent_context != "primary":
            return
        self._turn += 1
        try:
            if user_content:
                self.store.append("user", str(user_content)[:TOOL_CLIP])
            if assistant_content:
                self.store.append("unii", str(assistant_content)[:TOOL_CLIP])
        except Exception as exc:
            logger.debug("uniichat sync failed: %s", exc)
        # rebuild when the log grew since the last tree build (stale-count
        # trigger, not per-session turn count — sessions are short)
        try:
            stale = len(self.store.messages()) - len(self.store._load_tree() and
                     [k for k in self.store._load_tree() if k[0] == 0])
            if stale >= 10:
                self._ensure_tree_async()
        except Exception:
            pass

    def recall_status(self) -> Optional[RecallStatus]:
        tree = self.store._load_tree()
        return RecallStatus(provider_label="uniichat", count=len(tree)) if tree else None

    # -- tools --

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [UNIICHAT_RECALL_SCHEMA, UNIICHAT_APPEND_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "uniichat_note":
            text = str(args.get("text", "")).strip()
            if not text:
                return json.dumps({"error": "text required"})
            i = self.store.append("note", text)
            self._ensure_tree_async()
            return json.dumps({"ok": True, "message_id": i})
        if tool_name != "uniichat_recall":
            return json.dumps({"error": f"unknown tool {tool_name}"})
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "query required"})
        t0 = time.time()
        tree = self.store._load_tree()
        if not tree:
            return json.dumps({"error": "log empty"})
        menu = self.store.index_menu()
        route_prompt = (f"Memory index (each line summarizes 4 messages of past sessions):\n\n"
                        f"{menu}\n\nQuestion: {query}\n"
                        f"Which index line covers it? Reply ONLY with its tag like [12+4]. "
                        f"If none relate, reply NONE.")
        route = _chat(route_prompt)
        m = re.search(r"(\d+)\s*\+\s*\d+", route or "")
        if not m:
            # glm can be flaky on the big menu — one retry
            time.sleep(2)
            route = _chat(route_prompt)
            m = re.search(r"(\d+)\s*\+\s*\d+", route or "")
        if not m or "NONE" in (route or "").upper()[:20]:
            return json.dumps({"answer": "no relevant block found in the index",
                               "latency_s": round(time.time() - t0, 1)})
        start = int(m.group(1))
        leaves = self.store.leaves(max(0, start - 2), start + 4 + 2)
        material = "\n".join(f"[{mm['i']}] {mm['text'][:250]}" for mm in leaves)
        answer = _chat(f"Raw messages from the log:\n\n{material}\n\n"
                       f"Question: {query}\nAnswer from the material only. "
                       f"If absent, say NOT FOUND.") or ""
        # 2nd opinion: gemini (agy) verifies NOT FOUND verdicts — catches
        # routing/paraphrase misses
        if "NOT FOUND" in answer.upper():
            second = _agy_verify(query, material, answer)
            if second:
                answer = f"{answer}\n\n[2nd opinion — gemini]: {second}"
        self._last_recall_count = 1
        return json.dumps({
            "answer": answer,
            "block": f"{start}+4",
            "message_ids": [mm["i"] for mm in leaves],
            "latency_s": round(time.time() - t0, 1),
        })

    # -- config --

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "summarize", "type": "bool", "default": True,
             "description": "Use surplus glm to summarize tree nodes (false = clipped verbatim)"},
            {"key": "root", "type": "string", "default": "~/.hermes/uniichat",
             "description": "Storage directory for the log + tree"},
        ]


def _agy_verify(query: str, material: str, first_answer: str) -> Optional[str]:
    """Second-opinion check via antigravity gemini (agy CLI). Returns None on failure."""
    import subprocess
    prompt = (f"Raw messages from a coding session log:\n\n{material[:12000]}\n\n"
              f"Question: {query}\n\nA first reviewer answered:\n{first_answer[:800]}\n\n"
              f"You are the second reviewer. Read the material yourself. If the first "
              f"reviewer missed the answer, give the correct one concisely. If they were "
              f"right that it's absent, say NOT FOUND.")
    try:
        r = subprocess.run(["agy", "-p=" + prompt], capture_output=True,
                           text=True, timeout=240)
        out = (r.stdout or "").strip()
        return out if out and "NOT FOUND" not in out.upper() else None
    except Exception as exc:
        logger.debug("uniichat agy verify failed: %s", exc)
        return None


def _chat(prompt: str, max_tokens: int = 1500) -> Optional[str]:
    key = _summarizer_key()
    if not key:
        return None
    body = json.dumps({"model": "glm-5.3-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.2}).encode()
    req = urllib.request.Request(
        "https://api.surplusintelligence.ai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read())
        return (d["choices"][0]["message"].get("content") or "").strip() or None
    except Exception as exc:
        logger.debug("uniichat chat failed: %s", exc)
        return None


def register(ctx) -> None:
    """Plugin entry point."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        plug_cfg = cfg_get(cfg, "plugins", "uniichat") or {}
    except Exception:
        plug_cfg = {}
    if not isinstance(plug_cfg, dict):
        plug_cfg = {}
    ctx.register_memory_provider(UniichatMemoryProvider(config=plug_cfg))
