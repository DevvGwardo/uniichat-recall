#!/usr/bin/env python3
"""Benchmark the Uniichat memory provider on real, frozen log data.

Tiers
  L  literal   query built from a record's own distinctive anchors -> gold id.
               Exercises the no-model keyword path (store.search + strong gate).
  N  negative  fabricated tokens that exist nowhere -> correct behaviour is a miss.
  S  semantic  the real provider end-to-end (route -> leaf window -> answer ->
               agy second opinion). Costs model calls; needs the Surplus key.

Usage (Hermes venv, plugin + agent on PYTHONPATH):
  PY=$HOME/.hermes/hermes-agent/.venv/bin/python
  PP=$HOME/.hermes/hermes-agent:$HOME/.hermes/plugins

  PYTHONPATH=$PP $PY bench/bench.py --snapshot              # freeze the log
  PYTHONPATH=$PP $PY bench/bench.py --build 120             # write bench/gold.jsonl
  PYTHONPATH=$PP $PY bench/bench.py                          # tiers L + N
  PYTHONPATH=$PP $PY bench/bench.py --semantic --sample 25   # + tier S

Never writes to the live log: every run reads a snapshot copy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LIVE_ROOT = Path.home() / ".hermes" / "uniichat"
SNAPSHOT = BENCH / "snapshot"
GOLD = BENCH / "gold.jsonl"
TOOL_CLIP = 30_000

SHA = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")
PATH = re.compile(r"\b[\w./-]{4,}\.(?:py|ts|tsx|js|mjs|cjs|md|json|ya?ml|sh|sql|toml)\b")
IDENT = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,3}\b")
WORD = re.compile(r"\b[a-z][a-z]{4,}\b")

STOP = {
    "which", "there", "these", "those", "would", "could", "should", "because", "while",
    "still", "after", "before", "about", "every", "other", "under", "where", "being",
    "their", "them", "then", "than", "into", "from", "with", "have", "that", "this",
    "what", "when", "were", "will", "your", "also", "only", "just", "like", "does",
}


def load_store(root: Path):
    from uniichat.store import UniichatStore  # noqa: PLC0415 - path setup by caller

    return UniichatStore(root, summarize=False)


def load_key() -> None:
    import os  # noqa: PLC0415

    env_file = Path.home() / ".hermes" / "profiles" / "agent-two" / ".env"
    for line in env_file.read_text().splitlines():
        if line.startswith("HERMES_CUSTOM_API_SURPLUSINTELLIGENCE_AI_API_KEY="):
            os.environ.setdefault(*line.strip().split("=", 1))


def patch_answer_model(model: str) -> None:
    """Swap the model behind `_chat` (routing menu + answer), keeping the call shape.

    The shipped model is a reasoning model: measured 2026-10-26 at ~86 s for a
    264-char answer at max_tokens=1500, and 0 chars / finish_reason=length at
    max_tokens=200. The provider's whole semantic budget is 90 s, so scoring
    routing with that model measures the model, not the retrieval.
    """
    import json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    import uniichat as plugin  # noqa: PLC0415

    load_key()

    def _chat(prompt: str, max_tokens: int = 1500, *, timeout: float = 90):
        import os  # noqa: PLC0415

        key = os.environ.get("HERMES_CUSTOM_API_SURPLUSINTELLIGENCE_AI_API_KEY")
        if not key:
            return None
        body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0.2,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            "https://api.surplusintelligence.ai/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as handle:
                data = json.loads(handle.read())
            return (data["choices"][0]["message"].get("content") or "").strip() or None
        except Exception:
            return None

    plugin._chat = _chat
    print(f"answer model forced to {model}")


def freeze(src: Path = LIVE_ROOT, dst: Path = SNAPSHOT) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for part in ("main", "tree"):
        if (src / part).is_dir():
            shutil.copytree(src / part, dst / part)
    return dst


def df_map(records: list[dict]) -> dict[str, int]:
    df: dict[str, int] = {}
    for record in records:
        for word in set(WORD.findall(record["text"].lower())):
            df[word] = df.get(word, 0) + 1
    return df


def anchors(text: str) -> list[str]:
    out = [m.group(0) for m in SHA.finditer(text)]
    out += [m.group(0) for m in PATH.finditer(text)]
    out += [m.group(0) for m in IDENT.finditer(text)]
    return out


def build_gold(count: int, seed: int = 7) -> list[dict]:
    store = load_store(SNAPSHOT)
    records = [r for r in store.iter_messages() if len(r["text"]) >= 240]
    rng = random.Random(seed)
    rng.shuffle(records)
    df = df_map(records)
    ceiling = max(8, int(len(records) * 0.15))  # drop near-ubiquitous words
    gold: list[dict] = []
    for record in records:
        if len(gold) >= count:
            break
        found = [a for a in anchors(record["text"]) if len(a) >= 5]
        if not found:
            continue
        words = [
            w for w in dict.fromkeys(WORD.findall(record["text"].lower()))
            if w not in STOP and df.get(w, 0) <= ceiling
        ]
        if len(words) < 3:
            continue
        pick = rng.sample(words, 3)
        anchor = max(found, key=len)
        gold.append({
            "id": f"L{len(gold) + 1:03d}", "tier": "L", "query": " ".join([anchor, *pick]),
            "gold_id": record["i"], "anchor": anchor, "words": pick,
            "date": record.get("date", ""), "chars": len(record["text"]),
        })
    vocab = [w for w in df if df[w] <= ceiling and w not in STOP]
    for n in range(1, 21):
        junk = rng.sample(vocab, 3)
        gold.append({
            "id": f"N{n:03d}", "tier": "N",
            "query": f"zq{n:02d}qlorpt " + " ".join(junk),
            "gold_id": None, "anchor": f"zq{n:02d}qlorpt", "words": junk,
            "date": "", "chars": 0,
        })
    return gold


def score_row(store, item: dict, k_hits: list[dict]) -> dict:
    from uniichat.store import match_info, query_terms  # noqa: PLC0415

    terms = query_terms(item["query"])
    gate_strong = False
    for hit in k_hits:
        phrase, matched, _ = match_info(hit["text"], item["query"], terms)
        if phrase or matched == len(terms):
            gate_strong = True

    gold_words: set[str] = set()
    if item["gold_id"] is not None:
        gold_text = next(
            (r["text"] for r in store.iter_leaves(item["gold_id"], item["gold_id"] + 1)), "")
        gold_words = set(WORD.findall(gold_text.lower()))

    strict = lenient = None
    for rank, hit in enumerate(k_hits, 1):
        overlap = len(gold_words & set(WORD.findall(hit["text"].lower()))) if gold_words else 0
        if hit["i"] == item["gold_id"]:
            strict = strict or rank
            lenient = lenient or rank
        elif overlap >= 2 and lenient is None:
            lenient = rank

    return {
        "id": item["id"], "tier": item["tier"], "query": item["query"],
        "gold_id": item["gold_id"], "n_hits": len(k_hits),
        "strict_rank": strict, "lenient_rank": lenient,
        "gate_strong": gate_strong,
        "top1": k_hits[0]["i"] if k_hits else None,
    }


def summarize(rows: list[dict], latency: list[float]) -> dict:
    lit = [r for r in rows if r["tier"] == "L"]
    neg = [r for r in rows if r["tier"] == "N"]
    para = [r for r in rows if r["tier"] == "P"]
    def rate(items, key, k):
        if not items:
            return None
        return round(sum(1 for r in items if r[key] and r[key] <= k) / len(items), 3)
    mrr = [
        1 / r["lenient_rank"] for r in lit if r["lenient_rank"]
    ]
    return {
        "tier_L_n": len(lit), "tier_N_n": len(neg),
        "strict_hit@1": rate(lit, "strict_rank", 1),
        "strict_hit@3": rate(lit, "strict_rank", 3),
        "strict_hit@6": rate(lit, "strict_rank", 6),
        "lenient_hit@1": rate(lit, "lenient_rank", 1),
        "lenient_hit@3": rate(lit, "lenient_rank", 3),
        "lenient_hit@6": rate(lit, "lenient_rank", 6),
        "mrr_lenient": round(statistics.mean(mrr), 3) if mrr else None,
        "strong_gate_rate": rate(lit, "gate_strong", 1),
        "neg_gate_false_positive": rate(neg, "gate_strong", 1),
        "neg_candidate_noise": (
            round(sum(1 for r in neg if r["n_hits"] > 0) / len(neg), 3) if neg else None
        ),
        "tier_P_n": len(para),
        "paraphrase_keyword_hit@6": rate(para, "strict_rank", 6),
        "latency_p50_ms": round(statistics.median(latency) * 1000, 1) if latency else None,
        "latency_p95_ms": (
            round(sorted(latency)[int(len(latency) * 0.95) - 1] * 1000, 1) if latency else None
        ),
    }


def run_keyword(gold: list[dict]) -> tuple[list[dict], list[float]]:
    store = load_store(SNAPSHOT)
    rows, latency = [], []
    for item in gold:
        t0 = time.monotonic()
        hits = store.search(item["query"], limit=6)
        latency.append(time.monotonic() - t0)
        rows.append(score_row(store, item, hits))
    return rows, latency


def run_semantic(gold: list[dict], sample: int) -> list[dict]:
    import uniichat as plugin  # noqa: PLC0415

    load_key()
    provider = plugin.UniichatMemoryProvider(config={"root": str(SNAPSHOT)})
    provider.initialize("bench")
    store = provider.store
    items = [g for g in gold if g["tier"] == "L"][:sample]
    items += [g for g in gold if g["tier"] == "P"][: max(2, sample // 2)]
    items += [g for g in gold if g["tier"] == "N"][: max(2, sample // 4)]
    rows = []
    for item in items:
        t0 = time.monotonic()
        payload = json.loads(provider.handle_tool_call("uniichat_recall", {"query": item["query"]}))
        elapsed = time.monotonic() - t0
        answer = str(payload.get("answer", ""))
        ids = payload.get("message_ids") or []
        probe = (item.get("anchor") or item.get("check") or "").lower()
        hit = (item["gold_id"] in ids) if item["gold_id"] is not None else None
        blob = json.dumps(payload, ensure_ascii=False).lower()
        rows.append({
            "id": item["id"], "tier": item["tier"], "mode": payload.get("mode"),
            "block": payload.get("block"), "ids": ids[:6], "gold_id": item["gold_id"],
            "in_window": hit, "answer_chars": len(answer),
            # "raw"/"partial" modes answer with evidence + a canned sentence, so the
            # probe must be looked for in the whole payload, not in `answer`.
            "probe_in_payload": bool(probe) and probe in blob,
            "abstained": "NOT FOUND" in answer.upper() or payload.get("mode") == "partial",
            "answer_head": answer[:160],
            "elapsed_s": round(elapsed, 1),
            "bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
        })
        print(f"  {item['id']} {payload.get('mode'):9s} {elapsed:5.1f}s "
              f"window={'yes' if hit else 'no' if hit is not None else '-'} "
              f"answer={len(answer):5d}c", flush=True)
    return rows


def report(totals: dict, semantic: list[dict] | None) -> None:
    print("\n== keyword (no model) ==")
    for key, value in totals.items():
        print(f"  {key:22s} {value}")
    if semantic:
        lit = [r for r in semantic if r["tier"] == "L"]
        para = [r for r in semantic if r["tier"] == "P"]
        neg = [r for r in semantic if r["tier"] == "N"]
        size_bad = [r["id"] for r in semantic if r["bytes"] > TOOL_CLIP]
        modes: dict[str, int] = {}
        for r in semantic:
            modes[str(r["mode"])] = modes.get(str(r["mode"]), 0) + 1
        print("\n== semantic end-to-end ==")
        print(f"  runs                     {len(semantic)}  (L={len(lit)} P={len(para)} N={len(neg)})")
        print(f"  mode breakdown           {modes}")
        print(f"  L: gold in window        {sum(1 for r in lit if r['in_window'])}/{len(lit)}")
        print(f"  L: answer cites anchor   {sum(1 for r in lit if r['probe_in_payload'])}/{len(lit)}")
        print(f"  P: gold in window        {sum(1 for r in para if r['in_window'])}/{len(para)}")
        print(f"  P: answer cites evidence {sum(1 for r in para if r['probe_in_payload'])}/{len(para)}")
        print(f"  P: abstained             {sum(1 for r in para if r['abstained'])}/{len(para)}")
        print(f"  N: abstained             {sum(1 for r in neg if r['abstained'])}/{len(neg)}")
        print(f"  N: fabricated token      {sum(1 for r in neg if r['probe_in_payload'])}/{len(neg)}")
        print(f"  over 30,000-byte budget  {len(size_bad)}")
        print(f"  latency p50 / max        {statistics.median(r['elapsed_s'] for r in semantic):.1f}s"
              f" / {max(r['elapsed_s'] for r in semantic):.1f}s")


def build_paraphrase(count: int, seed: int = 11) -> list[dict]:
    """Questions written ABOUT a record, with its literal identifiers withheld.

    This is the tier that measures routing rather than keyword luck: the query
    shares no rare token with the gold record, so `store.search` cannot be
    expected to find it and the summary tree has to earn its keep.
    """
    import uniichat as plugin  # noqa: PLC0415

    load_key()
    store = load_store(SNAPSHOT)
    records = [r for r in store.iter_messages()
               if len(r["text"]) >= 500 and any(len(a) >= 6 for a in anchors(r["text"]))]
    rng = random.Random(seed)
    rng.shuffle(records)
    out: list[dict] = []
    attempts = failures = 0
    for record in records:
        if len(out) >= count or attempts >= count * 3 or failures >= 3:
            break
        attempts += 1
        found = [a for a in anchors(record["text"]) if len(a) >= 6]
        check = max(found, key=len)
        question = plugin._chat(
            "Read this transcript record. Write ONE question that a colleague would ask weeks "
            "later and that can only be answered from this record. Paraphrase: do not quote its "
            "file paths, commit hashes, function or flag names, or numbers. No preamble, output "
            "only the question.\n\n---\n" + record["text"][:4000] + "\n---", max_tokens=120
        )
        if not question or "?" not in question:
            failures += 1
            continue
        if check.lower() in question.lower():
            failures += 1
            continue
        failures = 0
        out.append({
            "id": f"P{len(out) + 1:03d}", "tier": "P", "query": question.strip(),
            "gold_id": record["i"], "check": check, "date": record.get("date", ""),
            "chars": len(record["text"]),
        })
        print(f"  P{len(out):03d} <- rec {record['i']}: {question.strip()[:90]}")
    return out


def ask_model(model: str, prompt: str, max_tokens: int = 500, timeout: float = 60) -> str:
    """One direct completion against the Surplus endpoint, for judging."""
    import json as _json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    load_key()
    body = _json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(
        "https://api.surplusintelligence.ai/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {os.environ['HERMES_CUSTOM_API_SURPLUSINTELLIGENCE_AI_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        data = _json.loads(handle.read())
    return (data["choices"][0]["message"].get("content") or "").strip()


def judge_answer(question: str, record: str, answer: str, model: str = "deepseek-v4.1-flash") -> bool:
    """A different model decides whether the answer states the record's fact."""
    if not answer.strip() or "NOT FOUND" in answer.upper():
        return False
    verdict = ask_model(model, (
        "A memory system answered a question from raw log records. Decide whether the "
        "answer states the KEY FACT contained in the reference record. Ignore wording, "
        "citations, and extra detail. An answer that only says it cannot find it, or "
        "that talks about unrelated topics, is WRONG.\n\n"
        f"Question: {question}\n\nReference record:\n{record[:2000]}\n\nAnswer:\n{answer[:2000]}\n\n"
        "Reply with exactly CORRECT or WRONG."
    ), max_tokens=1500)
    return verdict.strip().upper().startswith("CORRECT")


def run_paraphrase_eval(gold: list[dict], overrides: dict, answer_model: str) -> list[dict]:
    """Drive the real provider tool for each paraphrase item, then judge the answer."""
    import uniichat as plugin  # noqa: PLC0415

    load_key()
    patch_answer_model(answer_model)
    provider = plugin.UniichatMemoryProvider(config={"root": str(SNAPSHOT), **overrides})
    provider.initialize("bench")
    store = provider.store
    rows = []
    for item in [g for g in gold if g["tier"] == "P"]:
        started = time.monotonic()
        payload = json.loads(provider.handle_tool_call("uniichat_recall", {"query": item["query"]}))
        elapsed = time.monotonic() - started
        answer = str(payload.get("answer", ""))
        gold_text = next(
            (m["text"] for m in store.iter_leaves(item["gold_id"], item["gold_id"] + 1)), "")
        ids = payload.get("message_ids") or []
        verdict = judge_answer(item["query"], gold_text, answer)
        rows.append({
            "id": item["id"], "mode": payload.get("mode"), "block": payload.get("block"),
            "gold_in_window": item["gold_id"] in ids, "judged_correct": verdict,
            "answer_chars": len(answer), "elapsed_s": round(elapsed, 1),
            "bytes": len(json.dumps(payload, ensure_ascii=False).encode()),
            "answer_head": answer[:200],
        })
        print(f"  {item['id']} {str(payload.get('mode')):9s} block={payload.get('block')!s:12s} "
              f"{elapsed:5.1f}s judge={'CORRECT' if verdict else 'WRONG'}", flush=True)
    return rows


def report_paraphrase_eval(label: str, rows: list[dict]) -> dict:
    return {
        "label": label, "n": len(rows),
        "semantic_mode": sum(1 for r in rows if r["mode"] == "semantic"),
        "gold_in_window": sum(1 for r in rows if r["gold_in_window"]),
        "judged_correct": sum(1 for r in rows if r["judged_correct"]),
        "median_latency_s": round(statistics.median(r["elapsed_s"] for r in rows), 1),
        "over_budget": sum(1 for r in rows if r["bytes"] > TOOL_CLIP),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="re-freeze the live log into bench/snapshot")
    ap.add_argument("--build", type=int, metavar="N", help="build N literal gold items + 20 negatives (OVERWRITES gold.jsonl)")
    ap.add_argument("--build-paraphrase", type=int, metavar="N",
                    help="append N paraphrase items; pair with --model glm-4.7-flash")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--semantic", action="store_true", help="run the real provider end-to-end (model calls)")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--model", metavar="NAME",
                    help="override the model behind _chat (e.g. glm-4.7-flash)")
    ap.add_argument("--paraphrase-eval", action="store_true",
                    help="A/B the semantic path on tier P: menu routing vs candidate windows")
    ap.add_argument("--json", metavar="PATH", help="write full results as JSON")
    args = ap.parse_args()

    if args.snapshot or not SNAPSHOT.exists():
        freeze()
        print(f"snapshot: {SNAPSHOT}")
        if not args.build and not args.semantic and not GOLD.exists():
            return 0
    if args.build:
        gold = build_gold(args.build, args.seed)
        GOLD.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n")
        print(f"wrote {len(gold)} gold items -> {GOLD}")
        return 0
    if args.model:
        patch_answer_model(args.model)
    if args.build_paraphrase:
        rows = build_paraphrase(args.build_paraphrase, args.seed)
        with GOLD.open("a") as handle:
            handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"appended {len(rows)} paraphrase items -> {GOLD}")
        return 0
    if not GOLD.exists():
        ap.error("no gold set: run with --build 120 first")

    gold = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]
    if args.paraphrase_eval:
        answer_model = args.model or "glm-4.7-flash"
        results = []
        for label, overrides in (("menu routing (candidates_first=false)",
                                  {"candidates_first": False}),
                                 ("candidate windows (default)", {})):
            print(f"\n== {label}")
            rows = run_paraphrase_eval(gold, overrides, answer_model)
            results.append({"config": overrides, "rows": rows,
                            "summary": report_paraphrase_eval(label, rows)})
        print("\n== summary (tier P, judged by deepseek-v4.1-flash)")
        print(f"  {'config':38s} {'semantic':>9s} {'gold@win':>9s} {'correct':>8s} {'p50 s':>6s}")
        for entry in results:
            m = entry["summary"]
            print(f"  {m['label']:38s} {m['semantic_mode']}/{m['n']:>6} "
                  f"{m['gold_in_window']}/{m['n']:>6} {m['judged_correct']}/{m['n']:>4} "
                  f"{m['median_latency_s']:>6}")
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False))
            print(f"\nwrote {args.json}")
        return 0
    rows, latency = run_keyword(gold)
    totals = summarize(rows, latency)
    semantic = run_semantic(gold, args.sample) if args.semantic else None
    report(totals, semantic)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"totals": totals, "rows": rows, "semantic": semantic}, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())