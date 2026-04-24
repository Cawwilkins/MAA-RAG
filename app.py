"""
app.py

Streamlit UI for the MAA Assistant.

Run:
    streamlit run app.py

Place this file beside rag_service.py and your existing Interface.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from config import METADATA_FACETS_PATH
from rag_service import RAGService, TEMPLATE_LABELS


st.set_page_config(
    page_title="MAA Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def get_rag_service() -> RAGService:
    return RAGService()


@st.cache_data
def load_metadata_facets() -> dict[str, dict[str, int]]:
    path = Path(METADATA_FACETS_PATH)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_filter_dropdowns(facets: dict[str, dict[str, int]]) -> dict[str, Any]:
    metadata_filters: dict[str, Any] = {}

    if not facets:
        st.caption("No metadata facets file found yet.")
        return metadata_filters

    for field, values in facets.items():
        if not isinstance(values, dict) or not values:
            continue

        options = ["Any"] + sorted(values.keys())
        selected = st.selectbox(
            label=field.replace("_", " ").title(),
            options=options,
            index=0,
        )

        if selected != "Any":
            metadata_filters[field] = selected

    return metadata_filters


def render_source(source_node: Any, index: int) -> None:
    node = getattr(source_node, "node", source_node)
    score = getattr(source_node, "score", None)
    meta = getattr(node, "metadata", {}) or {}
    text = getattr(node, "text", "") or ""

    title = meta.get("title", "Untitled source")
    page = meta.get("page")
    section = meta.get("section")

    label_parts = [f"Source {index}: {title}"]
    if page is not None:
        label_parts.append(f"page {page}")
    if section:
        label_parts.append(f"section: {section}")

    with st.expander(" | ".join(label_parts)):
        if score is not None:
            st.caption(f"Relevance score: {score}")

        cols = st.columns(2)
        with cols[0]:
            st.write("**Metadata**")
            for key in [
                "job_number",
                "doc_type",
                "section",
                "source",
                "source_quality",
                "file_path",
                "source_id",
            ]:
                value = meta.get(key)
                if value not in {None, "", []}:
                    st.write(f"**{key}:** {value}")

        with cols[1]:
            st.write("**Detected Fields**")
            for key in [
                "ships",
                "ship_classes",
                "rates_mentioned",
                "shipyards_mentioned",
                "years_mentioned",
            ]:
                value = meta.get(key)
                if value not in {None, "", []}:
                    st.write(f"**{key}:** {value}")

        st.write("**Relevant excerpt**")
        st.text(text[:3000])


rag = get_rag_service()
facets = load_metadata_facets()

st.title("MAA Assistant")
st.caption("Ask questions over your indexed reports, references, and document metadata.")

with st.sidebar:
    st.header("Response Settings")

    response_length = st.radio(
        "Response length",
        options=["short", "medium", "long"],
        index=1,
        horizontal=True,
    )

    template_choice = st.selectbox(
        "Response type",
        options=list(TEMPLATE_LABELS.keys()),
        format_func=lambda key: TEMPLATE_LABELS[key],
        index=0,
    )

    references_only = st.toggle("References only", value=False)

    st.divider()
    st.header("Metadata Filters")
    metadata_filters = build_filter_dropdowns(facets)

    if references_only:
        metadata_filters["section"] = "references"

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about the documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = rag.ask(
                question=question,
                response_length=response_length,
                template_choice=template_choice,
                metadata_filters=metadata_filters,
                references_only=references_only,
            )

        st.write(result.answer)

        if result.source_nodes:
            st.divider()
            st.write("**Sources used**")
            for i, source_node in enumerate(result.source_nodes, start=1):
                render_source(source_node, i)

    st.session_state.messages.append({"role": "assistant", "content": result.answer})
