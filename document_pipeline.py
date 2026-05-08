"""
LLM steps and helpers for the document QA pipeline.

Pre-retrieval (order in ``RAGService.ask``):
  1. ``run_vagueness_check`` — boolean: too vague? If yes, UI asks user; next turn uses
     ``append_clarification_turn`` in ``query_rewire``.
  2. ``run_hybrid_retrieval_draft_llm`` — LLM shapes text for hybrid (dense + BM25) search.
  3. ``run_canonical_question_llm`` — LLM merges original question + clarification into one
     string used for evidence extraction, final answer, and (after rate expansion) retrieval.

Post-retrieval:
  ``build_evidence_context``, ``run_evidence_extraction``, ``run_final_answer``, node helpers.
"""

from __future__ import annotations

import numbers
import re
from typing import Any

from config import (
    CLARIFYING_QUESTION_MAX_NEW_TOKENS,
    CLARIFYING_QUESTION_PROMPT,
    EVIDENCE_EXTRACTION_MAX_NEW_TOKENS,
    EVIDENCE_EXTRACTION_PROMPT,
    FINAL_ANSWER_MAX_NEW_TOKENS,
    FINAL_ANSWER_PROMPT,
    FINAL_QUERY_REWRITE_MAX_NEW_TOKENS,
    FINAL_QUERY_REWRITE_PROMPT,
    MAX_EVIDENCE_CONTEXT_CHARS,
    RETRIEVAL_QUERY_EXPANSION_MAX_NEW_TOKENS,
    RETRIEVAL_QUERY_EXPANSION_PROMPT,
    VAGUENESS_BOOLEAN_PROMPT,
    VAGUENESS_CHECK_MAX_NEW_TOKENS,
    pipeline_debug_log,
)
from query_rewire import normalize_query


def _node_score(node: Any) -> float:
    score = getattr(node, "score", None)
    if isinstance(score, numbers.Real) and not isinstance(score, bool) and score == score:
        return float(score)
    meta = getattr(node, "metadata", {}) or {}
    for key in ("score", "retrieval_score", "relevance_score"):
        value = meta.get(key)
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and value == value:
            return float(value)
    inner_node = getattr(node, "node", None)
    inner_meta = getattr(inner_node, "metadata", {}) or {}
    for key in ("score", "retrieval_score", "relevance_score"):
        value = inner_meta.get(key)
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and value == value:
            return float(value)
    return float("-inf")


def dedupe_nodes_by_id(nodes: list[Any]) -> list[Any]:
    """Keep first occurrence order; when duplicate node_id appears, keep higher-scored node."""
    by_id: dict[str, Any] = {}
    order_keys: list[str] = []
    for nws in nodes:
        node = getattr(nws, "node", nws)
        nid = getattr(node, "node_id", None) or getattr(node, "id_", None) or str(id(node))
        if nid not in by_id:
            order_keys.append(nid)
            by_id[nid] = nws
        elif _node_score(nws) > _node_score(by_id[nid]):
            by_id[nid] = nws
    out = [by_id[k] for k in order_keys]
    pipeline_debug_log(
        "dedupe_nodes_by_id",
        f"input_nodes={len(nodes)} unique_after_dedupe={len(out)}",
    )
    return out


def build_evidence_context(nodes: list[Any], max_chars: int = MAX_EVIDENCE_CONTEXT_CHARS) -> str:
    parts: list[str] = []
    total = 0
    for i, nws in enumerate(nodes, 1):
        node = getattr(nws, "node", nws)
        text = (getattr(node, "text", "") or "").strip()
        meta = getattr(node, "metadata", {}) or {}
        title = meta.get("title") or meta.get("source_id") or "unknown"
        block = f"--- Snippet S{i} | {title} ---\n{text}\n"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 0:
                parts.append(block[:remain] + "\n[... truncated ...]")
            break
        parts.append(block)
        total += len(block)
    ctx = "\n".join(parts).strip()
    pipeline_debug_log(
        "build_evidence_context (string passed into evidence-extraction prompt as excerpts)",
        ctx,
    )
    return ctx


def llm_complete(llm: Any, prompt: str, max_new_tokens: int | None = None) -> str:
    kwargs: dict[str, Any] = {}
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    resp = llm.complete(prompt, **kwargs)
    return (getattr(resp, "text", None) or str(resp)).strip()


def _parse_too_vague_boolean(raw: str) -> bool:
    """``true`` in model output => question is too vague (per ``VAGUENESS_BOOLEAN_PROMPT``)."""
    s = (raw or "").strip().lower()
    if not s:
        return False
    m = re.search(r"\b(true|false)\b", s)
    if m:
        return m.group(1) == "true"
    if s.startswith("true"):
        return True
    if s.startswith("false"):
        return False
    return False


def _format_clarifying_questions_output(raw: str) -> str:
    lines_out: list[str] = []
    for ln in (raw or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        low = ln.lower()
        if low.startswith("output format:") or low.startswith("output:"):
            continue
        if ln.lstrip().startswith("-"):
            ln = ln.lstrip()[1:].lstrip()
        if ln:
            lines_out.append(ln)
    text = "\n".join(lines_out).strip()
    if not text:
        return (
            "Could you add a bit more detail—for example which person, record, vessel, "
            "or time period you mean?"
        )
    return text


def run_clarifying_questions(llm: Any, normalized_query: str) -> str:
    """User-facing follow-up when ``run_vagueness_check`` says the query is too vague."""
    q = (normalized_query or "").strip()
    prompt = CLARIFYING_QUESTION_PROMPT.format(question=q)
    pipeline_debug_log(
        "clarifying_question LLM prompt",
        f"question={q!r}\n\n{prompt}",
    )
    raw = llm_complete(llm, prompt, CLARIFYING_QUESTION_MAX_NEW_TOKENS)
    pipeline_debug_log("clarifying_question LLM raw response", raw)
    return _format_clarifying_questions_output(raw)


def run_vagueness_check(llm: Any, normalized_query: str) -> tuple[bool, str]:
    """
    Returns ``(specific_enough, user_message_if_blocked)``.

    If ``specific_enough`` is False, show ``user_message_if_blocked`` and store
    ``pending_question``; the next user message should be merged with
    ``append_clarification_turn`` in ``query_rewire``.
    """
    q = (normalized_query or "").strip()
    prompt = VAGUENESS_BOOLEAN_PROMPT.format(question=q)
    pipeline_debug_log(
        "vagueness_boolean LLM prompt",
        f"question={q!r}\n\n{prompt}",
    )
    raw = llm_complete(llm, prompt, VAGUENESS_CHECK_MAX_NEW_TOKENS)
    pipeline_debug_log("vagueness_boolean LLM raw response", raw)

    if _parse_too_vague_boolean(raw):
        clar = run_clarifying_questions(llm, q)
        return False, clar
    return True, ""


def _parse_retrieval_expansion_response(raw: str) -> str:
    m = re.search(r"(?mis)retrieval_query\s*:\s*(.*)", raw or "")
    if m:
        return (m.group(1) or "").strip()
    return (raw or "").strip()


def _parse_final_query_rewrite_response(raw: str) -> str:
    m = re.search(r"(?mis)finalized\s*query\s*:\s*(.*)", raw or "")
    if m:
        return (m.group(1) or "").strip()
    return (raw or "").strip()


def run_canonical_question_llm(
    llm: Any,
    original_question: str,
    clarification: str,
    *,
    draft_fallback_if_empty: str,
) -> str:
    """
    Single consolidated question from original + optional clarification reply.

    Used as ``canonical_question`` for evidence/final LLM and as input to
    ``normalize_then_expand_rate_aliases`` for hybrid retrieval.
    """
    o = normalize_query(original_question)
    c_raw = normalize_query(clarification)
    c = c_raw if c_raw else "(none — no additional clarification provided.)"
    prompt = FINAL_QUERY_REWRITE_PROMPT.format(question=o, clarification=c)
    pipeline_debug_log(
        "canonical_question LLM prompt",
        f"original={o!r}\nclarification_field={c!r}\n\n{prompt}",
    )
    raw = llm_complete(llm, prompt, FINAL_QUERY_REWRITE_MAX_NEW_TOKENS)
    pipeline_debug_log("canonical_question LLM raw response", raw)
    parsed = normalize_query(_parse_final_query_rewrite_response(raw))
    fb = normalize_query(draft_fallback_if_empty)
    return parsed if parsed else fb


def run_hybrid_retrieval_draft_llm(llm: Any, session_question: str) -> str:
    """
    First retrieval-oriented LLM pass: wording tuned for hybrid search.

    Output is fed into ``run_canonical_question_llm`` as ``draft_fallback_if_empty`` if
    the consolidation model returns nothing usable.
    """
    q = (session_question or "").strip()
    if not q:
        return ""
    prompt = RETRIEVAL_QUERY_EXPANSION_PROMPT.format(question=q)
    pipeline_debug_log(
        "hybrid_retrieval_draft LLM prompt",
        f"session_question={q!r}\n\n{prompt}",
    )
    raw = llm_complete(llm, prompt, RETRIEVAL_QUERY_EXPANSION_MAX_NEW_TOKENS)
    pipeline_debug_log("hybrid_retrieval_draft LLM raw response", raw)
    expanded = normalize_query(_parse_retrieval_expansion_response(raw))
    return expanded if expanded else normalize_query(q)


def run_evidence_extraction(llm: Any, context_str: str, canonical_question: str) -> str:
    prompt = EVIDENCE_EXTRACTION_PROMPT.format(
        canonical_question=canonical_question,
        snippets=context_str,
    )
    pipeline_debug_log(
        "evidence_extraction LLM prompt",
        f"canonical_question={canonical_question!r}\n\n{prompt}",
    )
    out = llm_complete(llm, prompt, EVIDENCE_EXTRACTION_MAX_NEW_TOKENS)
    pipeline_debug_log("evidence_extraction LLM raw response", out)
    return out


def run_final_answer(llm: Any, evidence_str: str, canonical_question: str) -> str:
    prompt = FINAL_ANSWER_PROMPT.format(
        canonical_question=canonical_question,
        evidence=evidence_str,
    )
    pipeline_debug_log(
        "final_answer LLM prompt",
        f"canonical_question={canonical_question!r}\n\n{prompt}",
    )
    out = llm_complete(llm, prompt, FINAL_ANSWER_MAX_NEW_TOKENS)
    pipeline_debug_log("final_answer LLM raw response", out)
    return out
