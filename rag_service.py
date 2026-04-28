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
    SUMMARY_TEMPLATE,
    TIMELINE_TEMPLATE,
    RETRIEVE_ONLY_DEFAULT_TOP_K,
    RETRIEVE_ONLY_DEFAULT_RATIO,
)

# Change this import if your main backend file has a different name.
# Example: if your file is Interface.py, keep this.
from Interface import load_index, initialize_query_engine, updateModel

from llama_index.core import QueryBundle


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

    def compact_value(value: Any) -> str:
        if isinstance(value, list):
            items = [compact_text(item) for item in value if compact_text(item)]
            return ", ".join(items)
        return compact_text(value)

    def add_if_exists(lines, label, value):
        clean_value = compact_value(value)
        if clean_value:
            lines.append(f"{label}: {clean_value}")

    lines = []
    lines.append("Here are the sources I used to generate the answer:")

    for i, node in enumerate(nodes, start=1):
        text = compact_text(getattr(node, "text", "") or "")
        score = getattr(node, "score", None)
        meta = getattr(node, "metadata", {}) or {}

        lines.append(f"--- Source {i} ---")

        if score is not None:
            if isinstance(score, float):
                lines.append(f"Relevance Score: {score:.4f}")
            else:
                lines.append(f"Relevance Score: {score}")

        add_if_exists(lines, "Title", meta.get("title"))
        add_if_exists(lines, "File Path", meta.get("file_path"))
        add_if_exists(lines, "Page", meta.get("page"))
        add_if_exists(lines, "Document Type", meta.get("doc_type"))
        add_if_exists(lines, "Section", meta.get("section"))
        add_if_exists(lines, "Source", meta.get("source"))
        add_if_exists(lines, "Source Quality", meta.get("source_quality"))
        add_if_exists(lines, "Source ID", meta.get("source_id"))
        add_if_exists(lines, "Job Number", meta.get("job_number"))
        add_if_exists(lines, "Ships Mentioned", meta.get("ships"))
        add_if_exists(lines, "Ship Classes Mentioned", meta.get("ship_classes"))
        add_if_exists(lines, "Years Mentioned", meta.get("years_mentioned"))
        add_if_exists(lines, "Rates Mentioned", meta.get("rates_mentioned"))
        add_if_exists(lines, "Shipyards Mentioned", meta.get("shipyards_mentioned"))

        lines.append("Relevant Excerpt:")
        lines.append(text)
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
    Thin wrapper between the UI and your existing RAG backend.

    The UI should call this class instead of calling input()/print()-based functions.
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
    ):
        metadata_filters = dict(metadata_filters or {})

        metadata_filters = dict(metadata_filters or {})

        if references_only:
            metadata_filters["section"] = "references"

        template_map = {
            "q": QA_TEMPLATE,
            "s": SUMMARY_TEMPLATE,
            "t": TIMELINE_TEMPLATE,
        }

        selected_template = template_map.get(template_choice, QA_TEMPLATE)

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
            nodes = retriever.retrieve(QueryBundle(query_str=question))
            nodes = apply_postprocessors(self.qe, nodes, question)

            if not references_only:
                nodes = [
                    node for node in nodes
                    if (getattr(node, "metadata", {}) or {}).get("section") != "references"
                ]

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
            progress_callback("Searching through docs...")
        retriever = getattr(self.qe, "_retriever")
        nodes = retriever.retrieve(QueryBundle(query_str=question))
        nodes = apply_postprocessors(self.qe, nodes, question)

        if not references_only:
            nodes = [
                node for node in nodes
                if (getattr(node, "metadata", {}) or {}).get("section") != "references"
            ]

        if not nodes:
            return {
                "answer": "No results found with the current settings.",
                "sources": [],
                "metadata_filters": metadata_filters,
                "references_only": references_only,
                "used_nodes_text": "",
            }

        if progress_callback:
            progress_callback("Generating an answer...")
        response = self.qe._response_synthesizer.synthesize(
            query=question,
            nodes=nodes,
        )

        used_nodes_text = format_used_nodes(nodes) if show_used_nodes else ""

        return {
            "answer": str(response),
            "sources": nodes,
            "metadata_filters": metadata_filters,
            "references_only": references_only,
            "used_nodes_text": used_nodes_text,
        } 
