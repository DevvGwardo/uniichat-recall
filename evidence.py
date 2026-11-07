"""Budgeted historical evidence with original-coordinate excerpts and honest provenance."""

from __future__ import annotations

import json
from itertools import islice

from .store import match_info, query_terms


def encoded_size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def excerpts(messages, query: str, *, budget: int, limit: int = 12, start: int | None = None) -> list[dict]:
    """Fair per-record SERIALIZED byte shares; never join complete turn bodies."""
    selected = list(islice(messages, limit))
    share = budget // max(1, len(selected))
    terms = query_terms(query)
    records = []
    for message in selected:
        text = message["text"]
        allowance = share - encoded_size(message["i"]) - 4  # also budget the response's ID list
        record = {key: message[key] for key in ("i", "kind", "date", "session_id") if key in message}
        record.update(text="", excerpt_start=0, original_chars=len(text), truncated=True,
                      has_more=False, next_excerpt_start=None)
        if encoded_size(record) > allowance:
            continue
        # Full probes are bounded by the budget even for enormous raw turns.
        begin = start if start is not None else 0
        full = {**record, "text": text[begin:], "excerpt_start": begin, "truncated": begin != 0} \
            if len(text) - begin <= allowance else None
        if full is not None and encoded_size(full) <= allowance:
            records.append(full)
            continue
        if start is None:
            _, _, position = match_info(text, query, terms)
            begin = max(0, position - 100)
        record["excerpt_start"] = begin
        low, high = 0, min(allowance, len(text) - begin)
        while low < high:
            middle = (low + high + 1) // 2
            record.update(text=text[begin:begin + middle], has_more=begin + middle < len(text),
                          next_excerpt_start=begin + middle if begin + middle < len(text) else None)
            if encoded_size(record) <= allowance:
                low = middle
            else:
                high = middle - 1
        record.update(text=text[begin:begin + low], has_more=begin + low < len(text),
                      next_excerpt_start=begin + low if begin + low < len(text) else None)
        records.append(record)
    return records


def response(payload: dict, *, budget: int) -> str:
    """Keep valid JSON and evidence; truncate an oversized generated answer explicitly."""
    if encoded_size(payload) > budget and "answer" in payload:
        answer = str(payload["answer"])
        payload = {**payload, "answer": "", "answer_truncated": True}
        allowance = max(0, budget - encoded_size(payload))
        low, high = 0, min(len(answer), allowance)
        while low < high:
            middle = (low + high + 1) // 2
            payload["answer"] = answer[:middle]
            if encoded_size(payload) <= budget:
                low = middle
            else:
                high = middle - 1
        payload["answer"] = answer[:low]
    return json.dumps(payload, ensure_ascii=False)
