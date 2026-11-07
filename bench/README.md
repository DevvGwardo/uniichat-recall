# Benchmarking Uniichat memory

`bench.py` measures the provider's two retrieval paths against real records, on a
frozen copy of the log. It answers three different questions that "does memory
work?" usually conflates:

| Tier | Question it answers | Model calls |
|------|--------------------|-------------|
| `L` literal | Can the exact-match path find a record from its own identifiers? | none |
| `P` paraphrase | Can the summary tree route a question that shares no rare token with the record? | route + answer |
| `N` negative | Does it refuse a question whose answer is not in the log? | route (+ agy) |

## Run

```sh
PY=$HOME/.hermes/hermes-agent/.venv/bin/python
PP=$HOME/.hermes/hermes-agent:$HOME/.hermes/plugins
cd ~/.hermes/plugins/uniichat

PYTHONPATH=$PP $PY bench/bench.py --snapshot              # freeze the live log
PYTHONPATH=$PP $PY bench/bench.py --build 120             # 120 L items + 20 N items
PYTHONPATH=$PP $PY bench/bench.py --build-paraphrase 6 --model glm-4.7-flash
PYTHONPATH=$PP $PY bench/bench.py --json bench/result-keyword.json
PYTHONPATH=$PP $PY bench/bench.py --semantic --sample 6 \
    --model glm-4.7-flash --json bench/result-semantic.json
```

`--model` monkeypatches the module-level `_chat` so routing/answering can be
scored separately from the shipped model's latency. Without it the run measures
the model, not the retrieval — see "Findings" below.

## Scoring

- **strict hit@k** — the gold record's `i` is in the top k of `store.search`.
  Honest but harsh here: the log repeats the same topics across sessions, so a
  rank-2 record often contains the same evidence.
- **lenient hit@k** — any top-k record shares ≥2 content words with the gold
  record. Reported alongside strict, never instead of it.
- **strong-gate rate** — fraction of queries whose candidates would take the
  no-model `raw` path (exact phrase or all query terms present).
- **gate false positive** — a query with no answer in the log that nevertheless
  trips the strong gate. This is the failure that *silently* answers from the
  wrong records, so it matters more than hit rate.
- **probe in payload** — a distinctive token from the gold record appears in the
  returned payload. Note: in `raw` and `partial` modes `answer` is a canned
  sentence and the evidence lives in `messages[]`, so grep the whole payload.
- **abstained** — `NOT FOUND` in the answer, or `mode == "partial"`.

## Measurement pitfalls (learned the hard way)

1. **Snapshot the log.** `store.search` reads the live append-only log; a run
   that spans a turn changes its own denominator.
2. **`answer` is not the answer** in raw/partial mode. Score the payload.
3. **The gold set must not leak its own literals.** Paraphrase questions are
   rejected at build time if they contain the record's check token, otherwise
   tier P silently collapses into tier L.
4. **Model latency belongs in the budget test.** The provider gives routing,
   answering and the agy second opinion one shared 90 s deadline; a slow answer
   model makes routing look broken.
5. **Duplicate topics are the norm, not noise.** Report strict *and* lenient, or
   the number understates retrieval and overstates fix difficulty.
6. **A single `agy` second opinion can flip a `NOT FOUND` into an answer** mid-run;
   record the raw first answer if you need to compare across dates.

## First run — 2026-10-26, snapshot of 1,043 records (Oct 8–26)

Keyword tier, 120 literal + 20 negative items:

```
strict hit@1   0.858   strict hit@3  0.967   strict hit@6  1.0
lenient hit@k  1.0 at every k
strong-gate rate 1.0     gate false positives 0/20 negatives, 0/6 paraphrases
latency p50 16.5 ms      p95 28 ms
```

Semantic tier, 11 runs with `--model glm-4.7-flash`:

```
mode breakdown        raw 6, partial 5
L: gold in window     6/6      (all answered by the keyword gate, 0.0 s)
P: gold in window     3/3      (via partial candidates, not routing)
P: abstained          3/3      routing returned NO block, 3/3
N: abstained          2/2      N: fabricated 0/2
over 30 KB budget     0
latency p50 0.0 s     max 2.8 s
```

## Findings from that run

1. **`_route` selects a block 0/5 times — the semantic layer is currently a
   no-op.** The level-2 menu is 17 lines covering **64 messages each** (450 chars
   per line). A stage-2 question ("verify the rules are sent by the wallet
   endpoint") gets the reply `NONE`, `_route` returns `None`, and recall degrades
   to "Partial candidates are not a verified answer." Confirmed on the live tool,
   not just the harness: `uniichat_recall` returned `mode: partial`, six
   candidates that did not contain the answer, in 61.3 s. The zoom machinery
   never gets a chance to narrow 64 → 32 → 16 → 8 because it must pick the right
   64-message block on the first try.
2. **The answering model is the wrong one for a 90 s shared budget.**
   `_chat` uses `glm-5.3-flash`, a reasoning model. Measured 2026-10-26:
   86 s / 264-char answer at `max_tokens=1500`; **0 chars, `finish_reason=length`
   at `max_tokens=200`** — the exact failure the summarizer's own comment
   documents. `glm-4.7-flash` answered the same prompt in 0.5 s. Routing needs
   2–3 calls, so with the shipped model the budget is spent before an answer.
3. **The tree trails the log.** `tree_valid: False`, `indexed_count 1037` vs
   `message_count 1043` on the snapshot — expected between rebuilds, but it means
   tier-P recall measures an index that is systematically behind the newest
   records the test questions came from.

Next measurement, in order: rebuild the tree with `glm-4.7-flash` and re-run
tier P to separate "routing is broken" from "the index is stale"; then test a
smaller top-level menu (8–16 messages per line via `menu_blocks(level=0)`) to see
whether the first hop is the bottleneck.

## 2026-10-27 — is it the model? (deepseek-v4.1-flash)

Same 11-run tier with `--model deepseek-v4.1-flash` (in scope for the Surplus key):

```
mode breakdown   raw 6, partial 5      block selected 0/5
P: gold in window 3/3   P: abstained 3/3   N: abstained 2/2
latency p50 0.1 s      max 52.8 s  (glm-4.7-flash: max 2.8 s)
```

Routing still returned no block on every semantic query, so **the model is not the
defect**. Two things the swap did establish:

1. **First-hop accuracy, same 17×64 menu, 3 questions:** deepseek `1 hit, 1 miss,
   1 empty`; glm-4.7-flash `NONE, NONE, 1 miss`. Neither model reliably picks one
   of 17 lines. `menu_blocks(level=0)` returns the *same* 17 lines at 7,754 bytes:
   the 8,192-byte menu cap, not the level argument, sets the granularity. The first
   hop is structurally a 64-message guess.
2. **Empty answers below ~1000 tokens.** deepseek returned `content: ""` with
   `finish_reason=length` at `max_tokens=200` and `700`, and 4–15 s answers at 1500.
   Any caller that shrinks `max_tokens` gets a silent empty answer, not an error —
   this is the same trap as `glm-5.3-flash`, only at a different threshold.

Also ruled out as the fix: seeding a window from the top keyword hit and answering
from ±12 raw messages. Top hits were long review dumps (`929`, `925`, `440`) that
outrank the real gold record, so 4/6 windows did not contain the answer at all.

**Where the leverage actually is:** `store.search` ranking. Long records win on
term overlap (`strict hit@1 0.858` while `hit@6 1.0`), which starves the semantic
path of good candidates before routing is even attempted. A length-normalised
(BM25-style) score is a pure, model-free change that moves the one weak number,
and tier P can be re-run unchanged to show whether routing improves once the
candidates do.

## 2026-10-27 — market price sweep (10 models, same 7,754-byte routing prompt)

Prices from `GET api.surplusintelligence.ai/v1/models` (489 models, live
`pricing.prompt`/`pricing.completion`); the `/markets` HTML page only carries
aggregate stats. Cost per recall assumes 2,500 in + 330 out for routing and
2,500 in + 300 out for the answer.

| model | $/Mtok in/out | per recall | latency | reply |
|---|---|---|---|---|
| glm-4.7-flash | 0.0605 / 0.40 | $0.00055 | **0.6 s** | clean `NONE`, 0 reasoning |
| openai-gpt-oss-120b | 0.07 / 0.30 | $0.00054 | 1.2 s | clean `NONE`, 339 tok |
| qwen3-next-80b-a3b-instruct | 0.09 / 1.10 | $0.00114 | 1.1 s | clean `NONE` |
| deepseek-v4.1-flash | 0.14 / 0.42 | $0.00096 | 14.3 s | **empty**, 1500 tok of reasoning |
| openai-gpt-oss-20b | 0.018 / 0.09 | — | 2.4 s | empty, truncated |
| qwen3.5-flash | unpriced | — | 13.6 s | `NONE`, 1129 tok reasoning |
| minimax-m2.7-highspeed | unpriced | — | 14.2 s | raw ` thinking` text |
| kimi-k2.6 | 0.65 / 3.41 | $0.0054 | 42.7 s | `NONE` |

**Zero of ten returned a valid tag.** Price is noise at this workload (~1,800
recalls per dollar); the only models that are fast *and* never return an empty
body are glm-4.7-flash, openai-gpt-oss-120b and qwen3-next-80b. Keep
glm-4.7-flash for summarise + route + answer. `openai-gpt-oss-120b` is the one
swap worth remembering if answer quality ever binds (same cost and speed class,
stronger model).

## 2026-10-27 — tree census, and why a bigger summary budget won't fix routing

```
nodes 2081    summarized 788 (38%)    fallback 1293
L0 span=1    0 / 1037 summarized   <- per-message leaves, fallback by design
L1           342 / 519
L2           207 / 260
L3..L9       224 / 234
menu level (17 lines x 64 msgs): 15 / 17 summarized
```

Two corrections to the note above:

1. The routing menu is **mostly real summaries** (15/17) — the junk
   `session start … x4` nodes are at L1/L2, not in the first hop. So summary
   sparsity is *not* why ten models failed to route.
2. The first hop is the wrong retrieval unit regardless: one 450-character line
   standing for 64 messages across several sessions cannot match a specific
   question. Fix the unit (session/day boundaries, or candidate-seeded windows),
   not the summarizer.

Rebuild ceiling, measured: `build_tree(force=True)` with `max_summaries=64`
produced **64 nodes in 89 s** and left 2030 fallbacks with 986 never attempted.
Filling 2,094 nodes at that budget is ~33 rebuilds (~50 min) of model calls, so
the inbox-cadence rebuild can never catch up on a tree this size — a backlog
needs a one-off bulk build, and a bulk build still won't fix the routing unit.

## 2026-10-27 — ranking fix in `store.search` (shipped)

`search` ranked on `(phrase, matched, newest)`, so any long record that merely
mentions every query term outranked the short specific one. Two variants tested
against the same 120 L + 20 N + 6 P gold set:

| variant | hit@1 | hit@3 | hit@6 | strong-gate | gate FP | p50 |
|---|---|---|---|---|---|---|
| shipped `(phrase, matched, i)` | 0.858 | 0.967 | 1.0 | 1.0 | 0 | 16.5 ms |
| A: pure BM25-lite (coverage folded into the score) | 0.742 | 0.875 | 0.925 | 0.925 | 0 | 18.0 ms |
| **B: coverage primary, BM25 specificity as tie-break** | **0.933** | **0.975** | 1.0 | 1.0 | 0 | 18.3 ms |

B ships. Coverage stays the first key so a one-rare-term record can never
outrank a full-coverage one — variant A proved that failure mode (a 36-character
`session start …` note beating records that matched all four terms). The idf +
length-normalised weight now only separates candidates with equal coverage, which
is exactly the long-dump case. No test changes were needed: 49 unit tests green,
`hit@6` flat at 1.0 and gate false positives still 0, so the recall contract is
untouched.

**Result: +7.5 points of hit@1 for ~2 ms.** Paraphrase keyword hit@6 rose
0.667 → 0.833 as a side effect.

### Candidate-seeded window: still inconclusive

With the fixed ranking, "top-1 hit ± 12 raw messages, answer from that" put the
gold record first for **2/6** paraphrases (was 1/6) and produced two substantively
correct answers (P003, P005). The literal-probe score was still 0/6 because a good
answer paraphrases the withheld token (a path or hash) instead of quoting it —
**the probe metric is the wrong judge for free-text answers**, not necessarily the
window being wrong. Next time this is measured, score it with an LLM judge or with
word-level check tokens ("rate limiting", "PONG"), never with a path.

## 2026-10-27 — candidate-seeded routing (shipped) and its A/B evaluation

`_route()`'s menu walk is skipped when `candidates_first` is on (the default):
`store.search` already ranks the records, so the provider reads disjoint windows
around the top hits, one window per session, and answers from those. The menu path
stays as the fallback when there are no candidates, and `candidates_first: false`
restores it exactly.

Tier P (6 paraphrase items), same answer model, answers judged by a *different*
model (deepseek-v4.1-flash, "does the answer state the record's key fact"):

| configuration | answered | gold in window | judged correct | median latency |
|---|---|---|---|---|
| menu routing (`candidates_first: false`) | 0/6 | 5/6 | **0/6** | 0.8 s |
| seeds ±6, one even 20 KB split | 6/6 | 2/6 | 2/6 | 13.0 s |
| seeds ±16 | 6/6 | 0/6 | 0/6 | 15.8 s |
| seeds ±30 | 6/6 | 0/6 | 0/6 | 14.4 s |
| **two-tier evidence (14 KB seeds + 6 KB context)** | **6/6** | 3/6 | **3/6** | 7.1 s |
| **two-tier, agy second opinion only on the menu path** | **6/6** | 3/6 | **3/6** | **0.9 s** |
| **same, shipped `_chat` default (`glm-4.7-flash`, unpatched)** | **6/6** | 3/6 | **3/6** | **0.86 s** |

`_chat` now defaults to `glm-4.7-flash` instead of `glm-5.3-flash` (same model the
summarizer already used). Verified with the shipped code path, no monkeypatch: the
last row is the production configuration, identical accuracy, sub-second median.
P001's winning answer is the literal `2112821688`, which is what makes the judge
call credible rather than generous.

What each step taught:

1. **Answering, not retrieval, was the dead part.** The menu path returned five
   gold-containing candidate lists across six questions and still answered none of
   them (`mode: partial`, canned sentence). Seeding produced an answer 6/6.
2. **Widening the window makes it worse.** ±16 and ±30 scored 0/6 because
   `excerpts()` splits a fixed 20 KB budget across every record in the window; more
   records means each one is truncated before the fact. Coverage is not the lever,
   depth is.
3. **Two tiers of budget fixes that:** the matched candidates get 14 KB, the
   surrounding context 6 KB, so the record the query named arrives whole.
4. **The agy second opinion never rescued a candidate-path miss** (0/7 over two
   runs) while costing 12–24 s each, so it now runs only on the menu path. That is
   where the 13.0 s → 0.9 s improvement comes from.

Result artifacts for this section: `bench/result-paraphrase-eval.json` holds the
two configurations that ran together (menu routing, seeds ±6);
`bench/result-paraphrase-2tier.json` holds the shipped two-tier run. Anything else
under `bench/` is a superseded or interrupted log — check the JSON is complete
before quoting a number from it.

Budget hazard on the menu path: with the shipped `glm-5.3-flash`, a single routing
call on the 7.7 KB menu measured 86 s against a 90 s shared deadline, so the
menu-config half of this suite can appear to hang for minutes (one run was killed
after >4 min still on item 6). A candidates-first run finishes the same suite in
seconds. That asymmetry is the practical argument for the new default.

The three remaining misses are genuine coverage misses: the window around the top
hits simply does not contain the record, and the answer is an honest `NOT FOUND`
rather than a wrong one.

**Wider retry, measured and removed (2026-10-27).** On a miss the provider re-read
4 sessions at ±12 (up from 2 at ±6) with the same two-tier budget. It rescued **0 of
3** misses, so the records that stay missing do not sit in a nearby-but-unread
window — no keyword hit reaches their session at all. Reverted for the same reason
agy was scoped out: it added a call to every miss and bought nothing. That result is
the ceiling statement for this mechanism: **window geometry cannot fix a candidate
set that never names the right session.**

Next lever, if this path is worth more: **query expansion** — one cheap call that
rewrites the question into 2–3 search phrasings, then seed from the union of their
hits. That can reach a session no literal match names, which is exactly the failure
the retry could not touch, and it is testable with the same tier in about six extra
calls.
## 2026-10-27 — miss log (shipped) and the first thing it caught

Unanswered recalls now append to `<root>/misses.jsonl`
(`{ts, query, mode, block, ids, latency_s}`, best effort, `log_misses: false` to
disable). Read it with:

```sh
$HOME/.hermes/hermes-agent/.venv/bin/python bench/misses.py --last 20
```

The six-item tier had stopped moving, so real misses are now the source of the next
fix. Two unit tests pin it: a miss is recorded (no-tree partial *and* semantic
`NOT FOUND`), and an answered recall writes nothing.

**First capture, and a failed fix worth recording.** `uniichat recall ranking bm25`
returned `NOT FOUND` while record 1058 — 2,059 characters, containing both `bm25`
and `0.933` — sat whole in the window. Not a retrieval miss: a prompt-shaped one, a
bare keyword query is not a question. Rewriting the prompt to "report what the
material says about those terms" was measured and **reverted**:

| check | original prompt | "report what the material says" |
|---|---|---|
| P tier correct | 3/6 | 3/6 (no change) |
| the keyword query | `NOT FOUND` | `NOT FOUND` (**not fixed**) |
| N tier abstention | **20/20** (full tier) | **18/20** (N005, N018 answered) |

The rewrite bought nothing on the case it targeted and lost abstention on two
negatives. Sequencing matters for honesty here: the 20/20 was measured on the
reverted prompt (identical code to the pre-experiment prompt) *after* the 18/20
result, so the comparison is apples-to-apples but not simultaneous — and an earlier
draft of this note quoted a 20/20 that had not yet been run. `bench/result-negative-baseline.json`
holds the full run; the full negative tier is the abstention number to quote, not a
two-item sample.

Against that baseline the shipped prompt abstains on **20/20** negatives while
answering 3/6 paraphrases correctly — the abstention side is solid, the recall side
is the open problem. So the keyword-query refusal stays an open, evidence-backed miss
class. The likely mechanism, once ~20 misses accumulate: when a query carries no
interrogative, return the windows with an extractive summary instead of posing it
as a question — a different shape of answer, not a sterner instruction.
