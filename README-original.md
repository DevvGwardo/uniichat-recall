# Uniichat memory

The append-only JSONL log is the evidence. The summary tree is a derived index,
not a replacement for the raw records.

## Storage and sharing

The existing default fleet log stays in the default Hermes root's `uniichat/`
directory. Set `plugins.uniichat.root` to choose another store, including a
profile-specific one. Existing records are never silently moved or rewritten.
New records carry their session ID. Credentials always come from the calling
profile's secret scope, even when several profiles share a log.

The store uses macOS/POSIX advisory file locks for cross-process appends and
tree builds. Tree files are replaced atomically. A log append during a slow
summary invalidates the snapshot rather than blocking behind the model call.
Completed user/assistant pairs share one append lock and fsync. This prevents
interleaving, but JSONL is not a crash-atomic whole-turn transaction.

Hot appends and counts track complete-line offsets, counts, and maximum IDs,
not full historical message bodies. Other writers' appended suffixes are read
under the shared append lock before allocating IDs. File replacement,
truncation, and same-size edits invalidate that file's metadata. Cold starts
reconstruct it from the raw log; growing files follow the append-only contract.
Interrupted trailing bytes are preserved and separated from the next append.

## Recall and compaction

- Keyword recall searches raw records, including newly appended, unindexed
  records. It returns the original text and record/session IDs without a model
  call. A complete phrase or all content keywords are required to bypass
  semantic routing. Incidental partial matches do not suppress semantic recall;
  if routing is unavailable they remain explicitly marked partial evidence.
  Responses are bounded to **30,000 serialized UTF-8 bytes**, including JSON
  escaping, metadata, and IDs. Sources get fair excerpt shares, original-string
  offsets, truncation flags, and continuation pointers. Reopen a citation with
  `uniichat_recall(query="", message_id=<id>, excerpt_start=0)`, then follow
  `next_excerpt_start` while `has_more` is true. These exact-source reads never
  call a model; offsets count characters in the original string, not bytes.
  Exact phrases and whole tokens anchor excerpts
  before incidental substrings; Unicode casefold expansion cannot shift them.
- Both per-turn and routing menus stay within **8,192 UTF-8 bytes**. The store
  chooses coarser blocks to cover the whole history rather than clipping away
  older records. Semantic routing validates actual tags and zooms to smaller
  blocks before reading raw evidence. Its generated answer includes the exact
  excerpts supplied to the model; verify them before treating it as proof.
  Routing, answering, and second-opinion verification share a **90-second**
  model-call budget instead of receiving independent full timeouts. An
  exhausted budget or unavailable answer leaves explicitly partial evidence.
  This is not a hard end-to-end wall-clock deadline: local I/O, network socket
  timeout semantics, and process cleanup can add time.
- Completed user and assistant turns are stored in full. Tool outputs remain
  in Hermes's session database, not automatically in this turn-only memory log.
- Recalled content is historical evidence, never permission to resume an old
  task or ignore the latest user instruction.

The tree's manifest fingerprints all log files. Appends, another process's
rebuild, and an odd partial block gaining a right child invalidate the relevant
cached state. Unchanged summary nodes are reused at every level.
At most 16 formatted menus are cached, invalidated by tree replacement.
Cache reads and invalidation share a lock. Leaf counts and narrow zooms use
the existing sorted level metadata rather than rescanning the entire tree.
Ordinary maintenance batches ten new records rather than rebuilding every
turn. Empty valid trees do not launch workers; background builds skip a held
cross-process tree lock and acquire their local guard before spawning.

`plugins.uniichat.max_summaries` bounds model calls per rebuild (default **8**).
Remaining nodes use balanced raw excerpts and can be upgraded on later builds.
Least-attempted pending nodes get priority across levels, so an early failing
node cannot indefinitely consume the entire retry budget on an unchanged log.
Successful summaries are distinguished from fallback/unknown nodes, so a
temporary API failure or a cheap build cannot permanently poison the cache.
Unfinished quality upgrades retry no sooner than five minutes when the log is
unchanged. Credential/profile context travels with the background worker.
Cold builds retain clipped leaves, not every original turn body, and stream
serialized tree records to the atomic publisher. A first summarized build
publishes a complete cheap index before calling a model; its manifest honestly
marks quality enrichment as pending. Malformed derived cache records are
discarded and rebuilt from the raw evidence.

An old tree without a manifest rebuilds once. To refresh locally without model
calls, instantiate `UniichatStore(root, summarize=False)` and call `build_tree()`.
Unchanged legacy summaries are retained when their child inputs still match.

## Offline checks

Use a Hermes test interpreter with this checkout on `PYTHONPATH`:

```sh
python -m unittest discover -s tests -v
```

Tests use temporary logs and mock model calls. They cover freshness,
concurrency, odd partial blocks, raw recall, full-turn persistence, profile
scope propagation, credential isolation, Unicode offsets, serialized response
budgets, adaptive routing, summary-call budgets, and interrupted writes.
They also cover first-index availability during a blocked model call, retry
fairness, malformed caches, concurrent invalidation, exact large-record
continuation, metadata-based zooms, and shared semantic-call deadlines.
