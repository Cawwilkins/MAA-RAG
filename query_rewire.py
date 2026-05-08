"""
Question text utilities for the RAG pipeline.

- ``normalize_query``: whitespace cleanup shared everywhere.
- ``append_clarification_turn``: one follow-up reply appended to the pending question.
- ``rewrite_query_for_rate_aliases``: Navy rate / alias OR-groups for better hybrid recall.
- ``normalize_then_expand_rate_aliases``: normalize + rate expansion → string used as
  ``QueryBundle.query_str`` (after upstream LLM consolidation).
"""

from __future__ import annotations

import re
from typing import Any

from config import RATE_ALIASES, pipeline_debug_log


def log_canonical_question_and_retrieval_query(
    canonical_question: str,
    hybrid_retriever_query_str: str,
) -> None:
    """Console trace: consolidated QA question vs final hybrid retriever input (after rate OR-groups)."""
    print(
        "\n"
        + "=" * 72
        + "\nHYBRID RETRIEVER (QueryBundle.query_str)\n"
        + "-" * 72
        + f"\nCanonical question (evidence + final-answer LLM): {canonical_question!r}\n"
        f"After Navy rate / alias expansion: {hybrid_retriever_query_str!r}\n"
        + "=" * 72
        + "\n",
        flush=True,
    )


def normalize_query(text: str) -> str:
    """Strip and collapse whitespace for consistent checks and prompts."""
    if not text:
        return ""
    t = str(text).strip()
    return re.sub(r"\s+", " ", t)


def append_clarification_turn(pending_question: str, user_clarification_reply: str) -> str:
    """
    Combine the stored pending question with the user's single clarification reply.

    The UI stores ``pending_question`` on a vagueness block; the next message is the reply.
    """
    b = normalize_query(pending_question)
    r = normalize_query(user_clarification_reply)
    if not r:
        return b
    combined = f"{b}\n\nAdditional detail from user: {r}"
    pipeline_debug_log(
        "append_clarification_turn",
        f"pending_question={b!r}\nuser_reply={r!r}\ncombined={combined!r}",
    )
    return combined


def normalize_then_expand_rate_aliases(question: str) -> str:
    """
    Normalize whitespace, then replace recognized Navy rate mentions with OR-groups
    from ``config.RATE_ALIASES``. Result is the string passed to the hybrid retriever.
    """
    normalized = normalize_query(question)
    expanded = rewrite_query_for_rate_aliases(normalized)
    if expanded != normalized:
        pipeline_debug_log(
            "normalize_then_expand_rate_aliases (OR-groups applied)",
            f"before={normalized!r}\nafter={expanded!r}",
        )
    else:
        pipeline_debug_log(
            "normalize_then_expand_rate_aliases (no rate matches)",
            f"query={expanded!r}",
        )
    return expanded


def rewrite_query_for_rate_aliases(
    question: str,
    rate_aliases: dict[str, list[Any]] | None = None,
) -> str:
    """
    If the question mentions any rate or alias from ``RATE_ALIASES``, replace each
    matched span with a parenthesized ``term1 OR term2 OR ...`` group (canonical
    title plus all aliases) so retrieval can match documents that use different
    wording (e.g. ``FN`` → ``(Fireman OR fireman OR fn)``).

    Matching is case-insensitive and uses word boundaries for single-token phrases;
    multi-word phrases allow flexible whitespace between words.

    If nothing matches, returns the original string unchanged.
    """

    def or_clause(canonical: str, aliases: list[Any]) -> str:
        parts: list[str] = []
        seen_exact: set[str] = set()
        for raw in [canonical, *(aliases or [])]:
            t = str(raw).strip()
            if not t or t in seen_exact:
                continue
            seen_exact.add(t)
            parts.append(t)
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"

    def pattern_for_phrase(phrase: str) -> str | None:
        phrase = str(phrase).strip()
        if not phrase:
            return None
        if " " in phrase:
            chunks = phrase.split()
            core = r"\s+".join(re.escape(c) for c in chunks)
            return rf"\b{core}\b"
        return rf"\b{re.escape(phrase)}\b"

    def build_specs(ra: dict[str, list[Any]]) -> list[tuple[int, re.Pattern[str], str]]:
        specs: list[tuple[int, re.Pattern[str], str]] = []
        for canonical, aliases in ra.items():
            if not isinstance(aliases, list):
                continue
            clause = or_clause(canonical, aliases)
            if not clause:
                continue
            phrases: list[str] = [canonical, *aliases]
            uniq: list[str] = []
            ph_seen: set[str] = set()
            for p in phrases:
                p = str(p).strip()
                if not p:
                    continue
                k = p.lower()
                if k in ph_seen:
                    continue
                ph_seen.add(k)
                uniq.append(p)
            uniq.sort(key=len, reverse=True)
            for p in uniq:
                pat = pattern_for_phrase(p)
                if pat:
                    specs.append((len(p), re.compile(pat, re.IGNORECASE), clause))
        specs.sort(key=lambda x: (-x[0], x[1].pattern))
        return specs

    ra = rate_aliases if rate_aliases is not None else RATE_ALIASES
    if not question or not ra:
        return question

    cache: dict[int, list[tuple[int, re.Pattern[str], str]]] = getattr(
        rewrite_query_for_rate_aliases, "_spec_cache", {}
    )
    key = id(ra)
    if key not in cache:
        cache[key] = build_specs(ra)
        setattr(rewrite_query_for_rate_aliases, "_spec_cache", cache)

    specs = cache[key]
    if not specs:
        return question

    text = question
    i = 0
    n = len(text)
    out: list[str] = []
    while i < n:
        best_start: int | None = None
        best_end: int | None = None
        best_clause: str | None = None
        for _plen, rx, clause in specs:
            m = rx.search(text, i)
            if m is None:
                continue
            s, e = m.start(), m.end()
            if s < i:
                continue
            if (
                best_start is None
                or s < best_start
                or (s == best_start and best_end is not None and e > best_end)
            ):
                best_start, best_end, best_clause = s, e, clause
        if best_start is None or best_end is None or best_clause is None:
            out.append(text[i:])
            break
        if best_start > i:
            out.append(text[i:best_start])
        out.append(best_clause)
        i = best_end

    return "".join(out)
