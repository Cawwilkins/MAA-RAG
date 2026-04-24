from typing import Any

from llama_index.core import Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from models.LLM_Header_File import HuggingFaceLLM

from config import (
    EMBED_MODEL_CONFIG,
    GENERATIVE_MODEL_CONFIG,
    QA_TEMPLATE,
    SUMMARY_TEMPLATE,
    EXPOSURE_ANALYSIS_TEMPLATE,
    TIMELINE_TEMPLATE,
    REFERENCE_EVIDENCE_TEMPLATE,
    COMPARISON_TEMPLATE,
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
    def add_if_exists(lines, label, value):
        if value is not None and value != []:
            lines.append(f"{label}: {value}")

    lines = []
    lines.append("Here are the sources I used to generate the answer:\n")

    for i, node in enumerate(nodes, start=1):
        text = getattr(node, "text", "") or ""
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

        lines.append("\nRelevant Excerpt:")
        lines.append(text.strip())
        lines.append("")

    return "\n".join(lines)

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
    ):
        metadata_filters = dict(metadata_filters or {})

        metadata_filters = dict(metadata_filters or {})

        if references_only:
            metadata_filters["section"] = "references"

        self.response_model, new_qe, new_hybrid, self.current_max_tokens, self.template_current = updateModel(
            self.index,
            template_choice,
            self.template_current,
            response_length,
            self.response_model,
            self.current_max_tokens,
        )

        Settings.llm = self.response_model

        # Always rebuild query engine here so UI filters are applied.
        template_map = {
            "q": QA_TEMPLATE,
            "s": SUMMARY_TEMPLATE,
            "e": EXPOSURE_ANALYSIS_TEMPLATE,
            "t": TIMELINE_TEMPLATE,
            "r": REFERENCE_EVIDENCE_TEMPLATE,
            "c": COMPARISON_TEMPLATE,
        }

        selected_template = template_map.get(template_choice, QA_TEMPLATE)

        self.qe, self.hybrid = initialize_query_engine(
            self.index,
            template=selected_template,
            metadata_filters=metadata_filters if metadata_filters else None,
        )

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
