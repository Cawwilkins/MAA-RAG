import re
import numbers
from typing import Any, Callable

from llama_index.core import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models.LLM_Header_File import HuggingFaceLLM

from config import (
    EMBED_MODEL_CONFIG,
    GENERATIVE_MODEL_CONFIG,
    QA_TEMPLATE,
    RETRIEVE_ONLY_DEFAULT_TOP_K,
    RETRIEVE_ONLY_DEFAULT_RATIO,
    pipeline_debug_log,
)

# Change this import if your main backend file has a different name.
# Example: if your file is Interface.py, keep this.
from Interface import load_index, initialize_query_engine, updateModel

from llama_index.core import QueryBundle

from query_rewire import (
    append_clarification_turn,
    log_canonical_question_and_retrieval_query,
    normalize_query,
    normalize_then_expand_rate_aliases,
)
from document_pipeline import (
    build_evidence_context,
    dedupe_nodes_by_id,
    run_canonical_question_llm,
    run_evidence_extraction,
    run_final_answer,
    run_hybrid_retrieval_draft_llm,
    run_vagueness_check,
)


def apply_postprocessors(query_engine, nodes, question: str):
    query_bundle = QueryBundle(query_str=question)
    postprocessors = getattr(query_engine, "_node_postprocessors", [])

    for postprocessor in postprocessors:
        nodes = postprocessor.postprocess_nodes(
            nodes,
            query_bundle=query_bundle,
        )

    return nodes


def format_used_nodes(nodes) -> str:
    def compact_text(value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    lines = []
    lines.append("Here are the sources I used to generate the answer:")

    for i, node in enumerate(nodes, start=1):
        text = (getattr(node, "text", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        score = getattr(node, "score", None)
        meta = getattr(node, "metadata", {}) or {}
        title = compact_text(meta.get("title")) or "Untitled source"
        page = compact_text(meta.get("page"))

        header = f"--- Source {i}: {title}"
        if page:
            header += f" (p. {page})"
        lines.append(header + " ---")

        if score is not None:
            if isinstance(score, float):
                lines.append(f"Relevance Score: {score:.4f}")
            else:
                lines.append(f"Relevance Score: {score}")

        lines.append("Excerpt:")
        lines.append(text if text else "(No excerpt text.)")
        lines.append("")

    return "\n".join(lines).strip()


def _node_score(node: Any) -> float:
    # Primary score location for NodeWithScore
    score = getattr(node, "score", None)
    if isinstance(score, numbers.Real) and not isinstance(score, bool) and score == score:
        return float(score)

    # Some node wrappers may carry score in metadata instead.
    meta = getattr(node, "metadata", {}) or {}
    for key in ("score", "retrieval_score", "relevance_score"):
        value = meta.get(key)
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and value == value:
            return float(value)

    # If this is a wrapper, check inner node metadata.
    inner_node = getattr(node, "node", None)
    inner_meta = getattr(inner_node, "metadata", {}) or {}
    for key in ("score", "retrieval_score", "relevance_score"):
        value = inner_meta.get(key)
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and value == value:
            return float(value)

    return float("-inf")

class RAGService:
    """
    UI-facing entry for one ``ask`` turn.

    Question pipeline (after optional vagueness + clarification):
      ``session_question`` → ``run_hybrid_retrieval_draft_llm`` →
      ``run_canonical_question_llm`` → ``normalize_then_expand_rate_aliases`` →
      hybrid retrieve → evidence + final answer LLMs (both use ``canonical_question``).
    """

    def __init__(self):
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(**EMBED_MODEL_CONFIG)

        self.index = load_index()

        self.response_model = HuggingFaceLLM(**GENERATIVE_MODEL_CONFIG)
        Settings.llm = self.response_model

        self.current_max_tokens = GENERATIVE_MODEL_CONFIG["max_new_tokens"]
        self.template_current = "q"

        self.qe, self.hybrid = initialize_query_engine(
            self.index,
            template=QA_TEMPLATE,
        )

    def ask(
        self,
        question: str,
        response_length: str = "medium",
        template_choice: str = "q",
        metadata_filters: dict | None = None,
        references_only: bool = False,
        show_used_nodes: bool = False,
        retrieve_only: bool = False,
        retrieve_max_results: int = RETRIEVE_ONLY_DEFAULT_TOP_K,
        retrieve_relevance_ratio: float = RETRIEVE_ONLY_DEFAULT_RATIO,
        progress_callback: Callable[[str], None] | None = None,
        clarification_reply: str | None = None,
        pending_question: str | None = None,
    ):
        pipeline_debug_log(
            "RAGService.ask entry",
            (
                f"question={question!r}\n"
                f"retrieve_only={retrieve_only} references_only={references_only} "
                f"show_used_nodes={show_used_nodes}\n"
                f"pending_question={pending_question!r}\n"
                f"clarification_reply={clarification_reply!r}"
            ),
        )

        metadata_filters = dict(metadata_filters or {})

        if references_only:
            metadata_filters["section"] = "references"

        template_map = {
            "q": QA_TEMPLATE,
        }

        selected_template = template_map.get(template_choice, QA_TEMPLATE)

        # --- Build session text, then LLM retrieval draft → canonical question → rate OR-groups ---
        if pending_question and clarification_reply:
            # Second turn: user replied to vagueness prompt; merge with stored pending question.
            original_question = normalize_query(pending_question)
            clarification_text = normalize_query(clarification_reply)
            session_question = append_clarification_turn(pending_question, clarification_reply)
        else:
            normalized = normalize_query(question)
            if progress_callback:
                progress_callback("Checking whether your question is specific enough…")
            ok, clar_msg = run_vagueness_check(Settings.llm, normalized)
            if not ok:
                pipeline_debug_log(
                    "RAGService.ask needs_clarification (no retrieval yet)",
                    f"pending_question will be={normalized!r}\nclarification_message={clar_msg!r}",
                )
                return {
                    "answer": "",
                    "sources": [],
                    "metadata_filters": metadata_filters,
                    "references_only": references_only,
                    "used_nodes_text": "",
                    "needs_clarification": True,
                    "pending_question": normalized,
                    "clarification_message": clar_msg,
                    "retrieve_only": retrieve_only,
                }
            original_question = normalized
            clarification_text = ""
            session_question = normalized

        if progress_callback:
            progress_callback("Optimizing your question for search…")
        hybrid_retrieval_draft = run_hybrid_retrieval_draft_llm(Settings.llm, session_question)

        if progress_callback:
            progress_callback("Finalizing your question for search and answering…")
        canonical_question = run_canonical_question_llm(
            Settings.llm,
            original_question,
            clarification_text,
            draft_fallback_if_empty=hybrid_retrieval_draft,
        )

        hybrid_retriever_query_str = normalize_then_expand_rate_aliases(canonical_question)
        log_canonical_question_and_retrieval_query(canonical_question, hybrid_retriever_query_str)

        if retrieve_only:
            if progress_callback:
                progress_callback("Searching through docs...")
            retrieve_max_results = max(1, int(retrieve_max_results))
            retrieve_relevance_ratio = float(retrieve_relevance_ratio)
            self.qe, self.hybrid = initialize_query_engine(
                self.index,
                template=selected_template,
                metadata_filters=metadata_filters if metadata_filters else None,
                similarity_top_k=retrieve_max_results,
                mixed_top_k=retrieve_max_results,
                rerank_top_n=retrieve_max_results,
                include_threshold_postprocessor=False,
                include_reorder_postprocessor=False,
            )

            retriever = getattr(self.qe, "_retriever")
            nodes = retriever.retrieve(QueryBundle(query_str=hybrid_retriever_query_str))
            pipeline_debug_log(
                "retrieve_only after retriever.retrieve",
                f"nodes={len(nodes)} query_str={hybrid_retriever_query_str!r}",
            )
            nodes = apply_postprocessors(self.qe, nodes, hybrid_retriever_query_str)
            pipeline_debug_log(
                "retrieve_only after postprocessors",
                f"nodes={len(nodes)}",
            )

            if not references_only:
                nodes = [
                    node for node in nodes
                    if (getattr(node, "metadata", {}) or {}).get("section") != "references"
                ]
                pipeline_debug_log(
                    "retrieve_only after non-references filter",
                    f"nodes={len(nodes)}",
                )

            ranked_nodes = sorted(nodes, key=_node_score, reverse=True)
            candidate_nodes = ranked_nodes[:retrieve_max_results]
            score_pairs = [(node, _node_score(node)) for node in candidate_nodes]
            scored_nodes = [(node, score) for node, score in score_pairs if score != float("-inf")]
            missing_score_count = len(score_pairs) - len(scored_nodes)

            if not scored_nodes:
                # Score-based threshold requires numeric scores.
                # If unavailable, keep candidates and surface diagnostics.
                top_score = float("-inf")
                bottom_score = float("-inf")
                threshold = float("-inf")
                kept_nodes = list(candidate_nodes)
            else:
                scores = [score for _, score in scored_nodes]
                top_score = max(scores)
                bottom_score = min(scores)
                score_span = top_score - bottom_score

                # Range-based threshold:
                # ratio=0.00 -> keep everything (threshold at bottom)
                # ratio=1.00 -> keep only top-scored nodes (threshold at top)
                threshold = bottom_score + (score_span * retrieve_relevance_ratio)
                kept_nodes = [node for node, score in scored_nodes if score >= threshold]

                # Safety fallback: never return zero if retrieval found candidates.
                if not kept_nodes and candidate_nodes:
                    kept_nodes = candidate_nodes[: min(5, len(candidate_nodes))]

            kept_nodes = dedupe_nodes_by_id(kept_nodes)

            dropped_count = len(candidate_nodes) - len(kept_nodes)
            kept_count = len(kept_nodes)
            answer = (
                f"Retrieve-only mode: returning {kept_count} result(s) out of "
                f"{len(candidate_nodes)} reranked candidate(s)."
            )

            print(
                "[retrieve-only debug] "
                f"requested={retrieve_max_results} "
                f"retrieved={len(candidate_nodes)} "
                f"scored={len(scored_nodes)} "
                f"missing_scores={missing_score_count} "
                f"top={top_score} bottom={bottom_score} "
                f"ratio={retrieve_relevance_ratio} threshold={threshold} "
                f"kept={kept_count} cut={dropped_count}"
            )
            if candidate_nodes:
                raw_scores_preview = []
                for node in candidate_nodes[:5]:
                    raw = getattr(node, "score", None)
                    raw_scores_preview.append(f"{type(raw).__name__}:{raw}")
                print("[retrieve-only debug] raw_score_preview=", raw_scores_preview)

            return {
                "answer": answer,
                "sources": kept_nodes,
                "metadata_filters": metadata_filters,
                "references_only": references_only,
                "used_nodes_text": "",
                "retrieve_only": True,
                "retrieve_top_k": retrieve_max_results,
                "retrieve_ratio": retrieve_relevance_ratio,
                "retrieve_threshold": threshold,
                "retrieve_top_score": top_score,
                "retrieve_bottom_score": bottom_score,
                "retrieve_kept_count": kept_count,
                "retrieve_cut_count": dropped_count,
                "retrieve_candidates_count": len(candidate_nodes),
                "retrieve_requested_count": retrieve_max_results,
                "retrieve_scored_count": len(scored_nodes),
                "retrieve_missing_score_count": missing_score_count,
            }

        self.response_model, new_qe, new_hybrid, self.current_max_tokens, self.template_current = updateModel(
            self.index,
            template_choice,
            self.template_current,
            response_length,
            self.response_model,
            self.current_max_tokens,
        )
        Settings.llm = self.response_model

        self.qe, self.hybrid = initialize_query_engine(
            self.index,
            template=selected_template,
            metadata_filters=metadata_filters if metadata_filters else None,
        )

        if progress_callback:
            progress_callback("Searching through docs…")
        retriever = getattr(self.qe, "_retriever")
        nodes = retriever.retrieve(QueryBundle(query_str=hybrid_retriever_query_str))
        pipeline_debug_log(
            "full pipeline after retriever.retrieve",
            f"nodes={len(nodes)} query_str={hybrid_retriever_query_str!r}",
        )
        nodes = apply_postprocessors(self.qe, nodes, hybrid_retriever_query_str)
        pipeline_debug_log(
            "full pipeline after postprocessors",
            f"nodes={len(nodes)}",
        )

        if not references_only:
            nodes = [
                node for node in nodes
                if (getattr(node, "metadata", {}) or {}).get("section") != "references"
            ]
            pipeline_debug_log(
                "full pipeline after non-references filter",
                f"nodes={len(nodes)}",
            )

        if not nodes:
            return {
                "answer": "No results found with the current settings.",
                "sources": [],
                "metadata_filters": metadata_filters,
                "references_only": references_only,
                "used_nodes_text": "",
            }

        pipeline_debug_log(
            "full pipeline before dedupe",
            f"nodes={len(nodes)} canonical_question={canonical_question!r}",
        )
        nodes = dedupe_nodes_by_id(nodes)

        if progress_callback:
            progress_callback("Extracting evidence from retrieved excerpts…")
        evidence_context = build_evidence_context(nodes)
        evidence_answer = run_evidence_extraction(Settings.llm, evidence_context, canonical_question)

        if progress_callback:
            evidence_body = (evidence_answer or "").strip() or "(No evidence text returned by the model.)"
            progress_callback(
                "This is the evidence I've extracted from the documents. "
                "I'm thinking through an answer now…\n\n"
                + evidence_body
            )

        answer_text = run_final_answer(Settings.llm, evidence_answer, canonical_question)

        used_nodes_text = format_used_nodes(nodes) if show_used_nodes else ""

        return {
            "answer": answer_text,
            "sources": nodes,
            "metadata_filters": metadata_filters,
            "references_only": references_only,
            "used_nodes_text": used_nodes_text,
            "evidence_answer": evidence_answer,
            "canonical_question": canonical_question,
        }
