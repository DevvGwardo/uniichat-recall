"""Append-only Uniichat storage and derived, replaceable search indexes."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

MAX_NODE = 512
MENU_BYTES = 8192
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with",
    "we", "our", "you", "i", "it", "this", "that", "is", "are", "was", "were",
    "do", "did", "what", "how", "when", "why", "about", "please",
})


def query_terms(query: str) -> set[str]:
    return set(re.findall(r"\w+", query.casefold())) - _QUERY_STOPWORDS


def clip(text: str, limit: int = MAX_NODE) -> str:
    """Never split a UTF-8 code point or exceed the byte budget."""
    return text[:limit].encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def match_info(text: str, query: str, terms: set[str]) -> tuple[int, int, int]:
    """Phrase priority and whole-token matches, anchored in ORIGINAL string coordinates."""
    matched = set()
    anchor = None
    folded = text.casefold()
    literal_position = folded.find(query.casefold())
    possible = {term for term in terms if term in folded}
    if literal_position < 0 and not possible:
        return 0, 0, 0
    folded_match = (re.search(r"(?<!\w)" + re.escape(query.casefold()) + r"(?!\w)", folded)
                    if literal_position >= 0 else None)
    if not folded_match:
        for token in re.finditer(r"\w+", text):
            normalized = token.group().casefold()
            if normalized in terms:
                matched.add(normalized)
                if anchor is None:
                    anchor = token.start()
                if len(matched) == len(possible):
                    break
    folded_position = folded_match.start() if folded_match else -1
    if not folded_match and not matched:
        # Preserve distinctive substring lookup inside identifiers, but never
        # let an incidental substring outrank an available whole-token match.
        folded_position = literal_position
    if folded_position >= 0:
        original = re.search(r"(?<!\w)" + re.escape(query) + r"(?!\w)", text, re.IGNORECASE)
        if original is None and not matched:
            original = re.search(re.escape(query), text, re.IGNORECASE)
        if original is not None and original.group().casefold() != query.casefold():
            original = None  # re.IGNORECASE also accepts variants that casefold keeps distinct.
        if original is not None:
            anchor = original.start()
        else:
            # Casefold can expand a code point (ß -> ss); its offsets are not
            # offsets into the original string. No unbounded mapping cache.
            folded_offset = 0
            for index, character in enumerate(text):
                if folded_offset >= folded_position:
                    anchor = index
                    break
                folded_offset += len(character.casefold())
    return int(folded_position >= 0), len(terms) if folded_match else len(matched), anchor or 0


def _decode_record(line: bytes) -> dict | None:
    try:
        record = json.loads(line)
        if (isinstance(record, dict) and type(record.get("i")) is int
                and record["i"] >= 0 and isinstance(record.get("text"), str)):
            return record
    except (ValueError, UnicodeDecodeError):
        pass
    return None


class UniichatStore:
    def __init__(self, root: Path, summarize: bool = True, summarizer: Callable | None = None,
                 *, max_summaries: int = 8):
        self.root = Path(root).expanduser()
        self.summarize = summarize
        self.summarizer = summarizer
        self.max_summaries = max(1, int(max_summaries))
        self._lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._tree = None
        self._tree_stamp = None
        self._stats_lock = threading.Lock()
        self._stats = {}
        self._tree_levels = {}
        self._menus = OrderedDict()
        self._cache_lock = threading.RLock()

    @contextlib.contextmanager
    def _file_lock(self, name: str, *, blocking: bool = True):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / name).open("a", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _log_signature(self) -> list[list]:
        return [[path.name, *self._version(path)]
                for path in sorted((self.root / "main").glob("*.jsonl"))]

    @staticmethod
    def _version(path: Path) -> list[int]:
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns]

    def _log_stats(self) -> tuple[int, int]:
        """Cache only complete-line offsets/counts/max IDs, never historical bodies.

        Cold starts reconstruct from raw logs. Appends consume each file's suffix;
        replacement, truncation, or same-size edits invalidate that file's cache.
        ID allocation calls this INSIDE the cross-process append lock.
        """
        with self._stats_lock:
            current = {}
            total, maximum = 0, -1
            for path in sorted((self.root / "main").glob("*.jsonl")):
                with path.open("rb") as handle:
                    stat = os.fstat(handle.fileno())
                    version = [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns]
                    old = self._stats.get(path.name)
                    if old and (old["version"] == version or
                                (old["version"][2] == stat.st_ino and old["version"][0] < stat.st_size)):
                        offset, count, highest = old["offset"], old["count"], old["max"]
                    else:
                        offset, count, highest = 0, 0, -1
                    handle.seek(offset)
                    tail = None
                    while line := handle.readline():
                        record = _decode_record(line)
                        if not line.endswith(b"\n"):
                            tail = record
                            break
                        offset = handle.tell()
                        if record is not None:
                            count += 1
                            highest = max(highest, record["i"])
                    current[path.name] = {"version": version, "offset": offset, "count": count, "max": highest}
                    total += count + int(tail is not None)
                    maximum = max(maximum, highest, tail["i"] if tail is not None else -1)
            self._stats = current
            return total, maximum

    def message_count(self) -> int:
        return self._log_stats()[0]

    def append(self, kind: str, text: str, *, session_id: str = "") -> int:
        return self.append_many([(kind, text)], session_id=session_id)[0]

    def append_many(self, entries: list[tuple[str, str]], *, session_id: str = "") -> list[int]:
        """Write a completed turn under one lock/fsync; JSONL is not a crash-atomic transaction."""
        if not entries:
            return []
        # Different agents have different store objects, so a thread lock alone
        # cannot protect the shared log's IDs or a multi-byte JSONL append.
        with self._lock, self._file_lock(".append.lock"):
            main = self.root / "main"
            main.mkdir(parents=True, exist_ok=True)
            index = self._log_stats()[1] + 1
            day = time.strftime("%Y-%m-%d")
            records = []
            for number, (kind, text) in enumerate(entries, index):
                record = {"i": number, "kind": kind, "text": text,
                          "size": len(text.encode("utf-8")), "date": day}
                if session_id:
                    record["session_id"] = session_id
                records.append(record)
            with (main / f"{day}.jsonl").open("a+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell():
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        # Keep an interrupted writer's bytes, but never concatenate
                        # the new record onto its incomplete JSON.
                        handle.write(b"\n")
                for record in records:
                    handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            return [record["i"] for record in records]

    def messages(self) -> list[dict]:
        return list(self.iter_messages())

    def iter_messages(self):
        for path in sorted((self.root / "main").glob("*.jsonl")):
            with path.open("rb") as handle:
                for line in handle:
                    record = _decode_record(line)
                    if record is not None:
                        yield record

    def _tree_path(self) -> Path:
        return self.root / "tree" / "all.jsonl"

    def tree_valid(self) -> bool:
        try:
            manifest = json.loads((self.root / "tree" / "state.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return False
            retry = manifest.get("retry_after")
            if retry is not None and (type(retry) not in (int, float) or not math.isfinite(retry)):
                return False
            return (manifest["log"] == self._log_signature()
                    and manifest["tree"] == self._tree_version()
                    and manifest["summarize"] is self.summarize
                    and (retry is None or time.time() < retry))
        except (OSError, ValueError, KeyError):
            return False

    def _tree_version(self) -> list[int]:
        return self._version(self._tree_path())

    @staticmethod
    def _write_atomic(path: Path, text: str | Iterable[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                if isinstance(text, str):
                    handle.write(text)
                else:
                    handle.writelines(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            # Strict same-directory rename is intentional: the host's generic
            # publisher can fall back to an in-place copy. Derived indexes must
            # remain atomic, and this store also supports standalone maintenance.
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def build_tree(self, force: bool = False, *, blocking: bool = True) -> int:
        # Tree building never holds the append lock across slow model calls.
        with self._build_lock, self._file_lock(".tree.lock", blocking=blocking) as acquired:
            if not acquired:
                return 0
            if not force and self.tree_valid():
                return len(self._load_tree())
            with self._file_lock(".append.lock"):
                signature = self._log_signature()
                leaves = {message["i"]: clip(message["text"]) for message in self.iter_messages()}
            previous = self._load_tree()
            if self.summarize and leaves and not previous:
                # Publish a complete raw index BEFORE any slow model call.
                # Its mode honestly records that quality enrichment is pending.
                previous, _ = self._build_records(leaves, {}, summarize=False)
                self._publish_tree(previous, signature, summarize=False, retry_needed=False)
            records, retry_needed = self._build_records(leaves, previous, summarize=self.summarize)
            self._publish_tree(records, signature, summarize=self.summarize, retry_needed=retry_needed)
            return len(records)

    def _build_records(self, current: dict, previous: dict, *, summarize: bool) -> tuple[dict, bool]:
        previous_keys = {}
        for old_level, old_index in sorted(previous):
            previous_keys.setdefault(old_level, []).append(old_index)
        pending_attempts = []
        for (height, index), node in (previous.items() if summarize else ()):
            if height == 0 or node.get("summary") is True or node.get("pending") is False:
                continue
            old_keys = previous_keys.get(height - 1, [])
            position = bisect_left(old_keys, index)
            children = old_keys[position:position + 2]
            inputs = "\n".join(previous[(height - 1, child)]["text"] for child in children)
            if len(inputs.encode()) > MAX_NODE:
                pending_attempts.append(node.get("attempts", 0))
        # Failed early nodes cannot keep spending every build's entire budget.
        # Inline nodes aren't candidates, even though they aren't model summaries.
        least_attempted = min(pending_attempts, default=0)
        records = {(0, index): {"l": 0, "i": index, "text": text,
                               "size": len(text.encode()), "span": 1}
                   for index, text in current.items()}
        retry_needed, summary_calls, level = False, 0, 0
        while len(current) > 1:
            keys, following = sorted(current), {}
            for position in range(0, len(keys), 2):
                children = keys[position:position + 2]
                index = children[0]
                text = "\n".join(current[child] for child in children)
                digest = hashlib.sha256(json.dumps(
                    [(child, current[child]) for child in children], ensure_ascii=False,
                ).encode()).hexdigest()
                cached = previous.get((level + 1, index))
                old_keys = previous_keys.get(level, [])
                old_position = bisect_left(old_keys, index)
                old_children = old_keys[old_position:old_position + 2]
                legacy_match = (cached is not None and not cached.get("inputs")
                                and old_children == children
                                and all(previous[(level, child)]["text"] == current[child]
                                        for child in children))
                unchanged = cached is not None and (cached.get("inputs") == digest or legacy_match)
                oversized, success = len(text.encode()) > MAX_NODE, False
                attempts = cached.get("attempts", 0) if unchanged else 0
                if unchanged and (not summarize or cached.get("summary") is True):
                    text, success = cached["text"], cached.get("summary", False)
                elif oversized:
                    summary = None
                    if (summarize and self.summarizer and summary_calls < self.max_summaries
                            and attempts <= least_attempted):
                        summary_calls += 1
                        attempts += 1
                        summary = self.summarizer(text)
                    text = summary or "\n".join(
                        clip(current[child], (MAX_NODE - len(children) + 1) // len(children))
                        for child in children)
                    success = bool(summary)
                pending = oversized and not success
                retry_needed |= summarize and pending
                following[index] = clip(text)
                records[(level + 1, index)] = {
                    "l": level + 1, "i": index, "text": following[index],
                    "size": len(following[index].encode()), "inputs": digest,
                    "summary": success, "pending": pending, "attempts": attempts,
                    "span": max(child + records[(level, child)]["span"] for child in children) - index,
                }
            current = following
            level += 1
        return records, retry_needed

    def _publish_tree(self, records: dict, signature: list, *, summarize: bool, retry_needed: bool) -> None:
        self._write_atomic(self._tree_path(), (
            json.dumps(record, ensure_ascii=False) + "\n" for record in records.values()))
        self._write_atomic(self.root / "tree" / "state.json", json.dumps(
            {"log": signature, "tree": self._tree_version(), "summarize": summarize,
             "retry_after": time.time() + 300 if retry_needed else None}))
        with self._cache_lock:
            self._tree = None

    def _load_tree(self) -> dict:
        with self._cache_lock:
            return self._read_tree()

    def indexed_count(self) -> int:
        with self._cache_lock:
            self._read_tree()
            return len(self._tree_levels.get(0, ()))

    def _read_tree(self) -> dict:
        try:
            version = self._tree_version()
        except OSError:
            self._tree = None
            self._tree_levels = {}
            self._menus.clear()
            return {}
        if self._tree is not None and self._tree_stamp == version:
            return self._tree
        tree = {}
        with self._tree_path().open("rb") as handle:
            for line in handle:
                record = _decode_record(line)
                if record is None or type(record.get("l")) is not int or record["l"] < 0:
                    continue
                if ("span" in record and (type(record["span"]) is not int or record["span"] <= 0)
                        or "attempts" in record and (type(record["attempts"]) is not int or record["attempts"] < 0)):
                    continue
                tree[(record["l"], record["i"])] = record
        self._tree, self._tree_stamp = tree, version
        self._tree_levels = {}
        for (level, index), record in sorted(tree.items()):
            self._tree_levels.setdefault(level, []).append((index, record))
        self._menus.clear()
        return tree

    def menu_blocks(self, level: int = 2, line_chars: int = 450, *, max_bytes: int = MENU_BYTES,
                    lo: int = 0, hi: int | None = None) -> list[tuple[int, int, str]]:
        with self._cache_lock:
            return self._menu_blocks(level, line_chars, max_bytes=max_bytes, lo=lo, hi=hi)

    def _menu_blocks(self, level: int, line_chars: int, *, max_bytes: int,
                     lo: int, hi: int | None) -> list[tuple[int, int, str]]:
        if max_bytes < 64:
            raise ValueError("menu budget must allow at least 64 bytes")
        tree = self._load_tree()
        if not tree:
            return []
        cache_key = (level, line_chars, max_bytes, lo, hi)
        if cache_key in self._menus:
            self._menus.move_to_end(cache_key)
            return list(self._menus[cache_key])
        maximum = max(self._tree_levels)
        level = min(level, maximum)
        while True:
            nodes = self._tree_levels.get(level, [])
            begin = bisect_left(nodes, lo, key=lambda node: node[0])
            end = len(nodes) if hi is None else bisect_left(nodes, hi, key=lambda node: node[0])
            blocks = [(index, record.get("span", 2 ** level), record["text"][:line_chars])
                      for index, record in nodes[begin:end]]
            size = sum(len(f"[{index}+{span}] {text}\n".encode()) for index, span, text in blocks)
            if size <= max_bytes or level == maximum:
                blocks = [(index, span, clip(text, max_bytes - len(f"[{index}+{span}] \n".encode())))
                          for index, span, text in blocks]
                self._menus[cache_key] = tuple(blocks)
                if len(self._menus) > 16:
                    self._menus.popitem(last=False)
                return blocks
            level += 1

    def index_menu(self, level: int = 2, line_chars: int = 450, *, max_bytes: int = MENU_BYTES) -> str:
        return "\n".join(f"[{index}+{span}] {text}"
                         for index, span, text in self.menu_blocks(level, line_chars, max_bytes=max_bytes))

    def leaves(self, lo: int, hi: int) -> list[dict]:
        return list(self.iter_leaves(lo, hi))

    def iter_leaves(self, lo: int, hi: int):
        return (message for message in self.iter_messages() if lo <= message["i"] < hi)

    def leaf_window(self, start: int, span: int) -> list[dict]:
        return self.leaves(max(0, start - 2), start + span + 2)

    def search(self, query: str, limit: int = 6) -> list[dict]:
        """Exact raw-log lookup, including records not yet in the summary tree.

        Ranking is phrase first, then a BM25-lite score: rare query terms weigh
        more, and a length normalisation (b=0.75) stops a long transcript dump
        from outranking a short record that covers the same query terms. Without
        the normalisation a 70 KB review dump wins every query it merely mentions.
        """
        terms = query_terms(query)
        query = query.casefold()
        if not terms:
            return []
        hits = []
        for message in self.iter_messages():
            phrase, matched, _ = match_info(message["text"], query, terms)
            if not (phrase or matched):
                continue
            folded = message["text"].casefold()
            # Only candidates can contain a query term, so this count is the
            # corpus document frequency without a second pass.
            present = {term for term in terms if term in folded}
            hits.append((phrase, matched, present, len(message["text"]), message["i"], message))
        if not hits:
            return []
        frequency = {term: sum(1 for hit in hits if term in hit[2]) for term in terms}
        corpus = len(hits)
        average = sum(hit[3] for hit in hits) / corpus or 1
        scored = []
        for phrase, matched, present, size, index, message in hits:
            # Coverage stays primary: a record matching one rare term must never
            # outrank one matching every term. The BM25 weight only separates
            # candidates with the same coverage, where it prefers the specific
            # record over a long transcript that merely mentions the terms.
            weight = sum(math.log(1 + corpus / (1 + frequency[term])) for term in present)
            specificity = weight / (0.25 + 0.75 * size / average)
            scored.append((phrase, matched, specificity, index, message))
        scored.sort(key=lambda item: item[:4], reverse=True)
        return [item[-1] for item in scored[:max(0, limit)]]
