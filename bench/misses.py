#!/usr/bin/env python3
"""Summarise <root>/misses.jsonl — the real misses that should drive the next fix.

  PY=$HOME/.hermes/hermes-agent/.venv/bin/python
  $PY bench/misses.py [--root ~/.hermes/uniichat] [--last 20]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(pathlib.Path.home() / ".hermes" / "uniichat"))
    ap.add_argument("--last", type=int, default=20)
    args = ap.parse_args()

    path = pathlib.Path(args.root).expanduser() / "misses.jsonl"
    if not path.exists():
        print(f"no misses yet ({path})")
        return 0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    modes = collections.Counter(row.get("mode") for row in rows)
    queries = collections.Counter(row.get("query") for row in rows)
    print(f"{len(rows)} misses  {dict(modes)}  since {rows[0].get('ts')}")
    repeated = [(q, n) for q, n in queries.most_common(5) if n > 1]
    if repeated:
        print("\nrepeated (a strong signal the retrieval mechanism, not the window, is wrong):")
        for query, count in repeated:
            print(f"  {count}x  {query[:90]}")
    print(f"\nlast {min(args.last, len(rows))}:")
    for row in rows[-args.last:]:
        print(f"  {row.get('ts')}  {str(row.get('mode')):9s} {str(row.get('block')):12s} "
              f"{str(row.get('latency_s')):>5}s  {str(row.get('query'))[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
