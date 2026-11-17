#!/usr/bin/env python3
"""LLM-free query-expansion alternatives, measured — kept for the 20-day re-run.

Question (2026-11-17): does the paraphrase fix need an LLM, or can the corpus
bridge the vocabulary itself? Two stdlib-only expanders were built and driven
through the provider's real expansion hook (message_ids metric, identical to
the LLM A/B on the same 8-item paraphrase gold over the 3-day snapshot):

  baseline (no expansion)          gold-in-window 6/8
  PRF / Rocchio from weak hits     no gain (mined variants thematically
                                   adjacent but off-target: the top hits are
                                   the wrong sessions, so feedback follows
                                   the wrong theme)
  query-term co-occurrence mining  6/8, no gain — on a 120-record chatty log
                                   the strong co-occur neighbours are generic
                                   hub terms ("still", "localhost",
                                   "verified"); the precise breadcrumbs
                                   (port->netstat) co-occur once and cannot
                                   be separated from coincidence
  LLM variants (glm-5.3-flash)     gold-in-window 6/8, judged-correct 5/8 ->
                                   6/8 (its measured benefit is the answer
                                   stage, not retrieval, on this young log)

Why the offline tiers measured zero here: corpus-local expansion can only
follow statistical breadcrumbs, and a small dense project chat doesn't have
statistically visible ones yet. They may pay off on a large, old log — run
this file against the frozen 20-day snapshot (bench --snapshot --build 120
--build-paraphrase 6) before believing either way.

Usage (Hermes venv, plugin + agent on PYTHONPATH):
  PYTHONPATH=<hermes-agent>:<plugins-dir> python bench/offline_expanders.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
GOLD = BENCH / "gold.jsonl"
SNAPSHOT = BENCH / "snapshot"
WORD = re.compile(r"[a-z][a-z0-9_]{3,}")


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "uniichat", REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    plugin = importlib.util.module_from_spec(spec)
    sys.modules["uniichat"] = plugin
    spec.loader.exec_module(plugin)
    return plugin


class CorpusIndex:
    """Record-level postings for df and co-occurrence, built once."""

    def __init__(self, records):
        self.records = records
        self.n = len(records)
        self.postings: dict[str, set[int]] = {}
        self.df: Counter = Counter()
        for index, record in enumerate(records):
            for word in set(WORD.findall(record["text"].lower())):
                self.postings.setdefault(word, set()).add(index)
                self.df[word] += 1

    def rare(self, word) -> bool:
        count = self.df.get(word, 0)
        return 0 < count <= max(6, self.n * 0.25)


def prf_variants(store, query, index, top_k=5, per_variant=4, max_variants=2):
    """Rocchio: mine distinctive terms from the query's own weak hits."""
    from uniichat.store import query_terms
    hits = store.search(query)
    if not hits:
        return []
    query_set = {t.lower() for t in query_terms(query)}
    score: Counter = Counter()
    for hit in hits[:top_k]:
        for word in set(WORD.findall(hit["text"].lower())):
            if word in query_set or len(word) < 4 or not index.rare(word):
                continue
            score[word] += math.log(index.n / index.df[word])
    ranked = [w for w, _ in score.most_common(per_variant * max_variants)]
    return [" ".join(ranked[i:i + per_variant])
            for i in range(0, len(ranked), per_variant)][:max_variants]


def cooccurrence_variants(query, index, per_variant=4, max_variants=2, min_pairing=2):
    """Mine terms that repeatedly co-occur with the query's own terms."""
    from uniichat.store import query_terms
    terms = [t.lower() for t in query_terms(query)]
    score: Counter = Counter()
    for term in terms:
        base = index.postings.get(term)
        if not base:
            continue
        for neighbor, docs in index.postings.items():
            if neighbor in terms or len(neighbor) < 5 or not index.rare(neighbor):
                continue
            pairing = len(base & docs)
            if pairing < min_pairing:
                continue
            score[neighbor] += pairing * math.log(index.n / index.df[neighbor])
    ranked = [w for w, _ in score.most_common(per_variant * max_variants)]
    return [" ".join(ranked[i:i + per_variant])
            for i in range(0, len(ranked), per_variant)][:max_variants]


def run_arm(provider, gold, label, variants_for):
    wins = 0
    marks = []
    for item in gold:
        if variants_for is None:
            provider._config["query_expansion"] = False
        else:
            provider._expand_query = lambda query, deadline: variants_for(query)
        payload = json.loads(provider.handle_tool_call("uniichat_recall", {"query": item["query"]}))
        hit = item["gold_id"] in (payload.get("message_ids") or [])
        wins += hit
        marks.append(f"{item['id']}={'Y' if hit else 'n'}")
    provider._config["query_expansion"] = True
    print(f"{label:34s} gold-in-window {wins}/{len(gold)}  [{' '.join(marks)}]")


def main() -> int:
    plugin = load_plugin()
    provider = plugin.UniichatMemoryProvider(
        {"root": str(SNAPSHOT), "summarize": False, "log_misses": False})
    store = provider.store
    index = CorpusIndex(list(store.iter_messages()))
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines()
            if line.strip() and '"tier": "P"' in line]
    if not gold:
        raise SystemExit("no paraphrase gold: build it first (see module docstring)")
    original = plugin.UniichatMemoryProvider._expand_query

    run_arm(provider, gold, "baseline (no expansion)", None)
    run_arm(provider, gold, "PRF from weak hits",
            lambda query: prf_variants(store, query, index))
    run_arm(provider, gold, "query-term co-occurrence",
            lambda query: cooccurrence_variants(query, index))

    plugin.UniichatMemoryProvider._expand_query = original
    provider._expand_query = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
