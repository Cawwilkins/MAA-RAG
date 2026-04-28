from pathlib import Path
import qdrant_client
import torch
import time
from typing import List, Set, Dict, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
    Settings,
    get_response_synthesizer,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage import StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LongContextReorder, SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from models.LLM_Header_File import HuggingFaceLLM, HybridRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from setup_index.create_index import create_index
from config import EMBED_MODEL_CONFIG, DB_DIR, STORAGE_DIR, DOCS_DIR, COLLECTION_NAME, INDEX_ID, EXPOSURE_ANALYSIS_TEMPLATE, TIMELINE_TEMPLATE, REFERENCE_EVIDENCE_TEMPLATE, COMPARISON_TEMPLATE, SUMMARY_TEMPLATE, QA_TEMPLATE, RERANK_MODEL_CONFIG, GENERATIVE_MODEL_CONFIG, SIMILARITY_TOP_K, MIXED_TOP_K, SCORE_RATIO, DOCS_STORE


def print_final_context(query_engine, nodes, question: str):
    """
    Applies postprocessors to nodes and prints the exact context
    that will be sent to the LLM.
    
    Returns:
        processed_nodes (list): nodes after postprocessing
    """
    def print_if_exists(label, value):
        if value is not None and value != []:
            print(f"{label}: {value}")

    query_bundle = QueryBundle(query_str=question)

    # Apply postprocessors (same order as query_engine)
    postprocessors = getattr(query_engine, "_node_postprocessors", [])
    for postprocessor in postprocessors:
        nodes = postprocessor.postprocess_nodes(
            nodes,
            query_bundle=query_bundle,
        )

    print("\n> MAA Assistant: Here's the information I'm using to answer your question.\n")
    print("> MAA Assistant: I've selected the most relevant sources and included key details from each.\n")

    full_context = ""

    for i, node in enumerate(nodes):
        text = getattr(node, "text", "") or ""
        score = getattr(node, "score", None)
        meta = getattr(node, "metadata", {}) or {}

        print(f"\n--- Source {i + 1} ---")

        if score is not None:
            print(f"Relevance Score: {score:.4f}" if isinstance(score, float) else f"Relevance Score: {score}")

        # --- Metadata ---
        print_if_exists("Title", meta.get("title"))
        print_if_exists("File Path", meta.get("file_path"))
        print_if_exists("Page", meta.get("page"))
        print_if_exists("Document Type", meta.get("doc_type"))
        print_if_exists("Source", meta.get("source"))
        print_if_exists("Source Quality", meta.get("source_quality"))
        print_if_exists("Source ID", meta.get("source_id"))

        # --- Case-specific ---
        print_if_exists("Job Number", meta.get("job_number"))

        # --- Domain-specific ---
        print_if_exists("Ships Mentioned", meta.get("ships"))
        print_if_exists("Ship Classes Mentioned", meta.get("ship_classes"))
        print_if_exists("Years Mentioned", meta.get("years_mentioned"))
        print_if_exists("Rates Mentioned", meta.get("rates_mentioned"))
        print_if_exists("Shipyards Mentioned", meta.get("shipyards_mentioned"))

        # --- Content ---
        print("\nRelevant Excerpt:")
        print(text.strip())

        full_context += text + "\n\n\n"

    print("\n> MAA Assistant: These are the sources I'm using to generate your answer.\n")

    return nodes


def show_nodes(nodes, show_score=True):
    def print_if_exists(label, value):
        if value is not None and value != []:
            print(f"{label}: {value}")

    for i, node in enumerate(nodes[:5]):
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", {}) or {}

        print(f"\n--- Node {i + 1} ---")

        if show_score:
            score = getattr(node, "score", None)
            if score is not None:
                print("Score:", score)

        print("Node Information:")

        # --- Core file info ---
        print_if_exists("Title is", meta.get("title"))
        print_if_exists("File Path is", meta.get("file_path"))
        print_if_exists("Page is", meta.get("page"))
        print_if_exists("Source is", meta.get("source"))
        print_if_exists("Source Quality is", meta.get("source_quality"))
        print_if_exists("Doc Type is", meta.get("doc_type"))
        print_if_exists("Source ID is", meta.get("source_id"))

        # --- Case-specific ---
        print_if_exists("Job Number is", meta.get("job_number"))

        # --- Ships ---
        print_if_exists("Ships Mentioned are", meta.get("ships"))

        # --- Ship Classes ---
        print_if_exists("Ship Classes Mentioned are", meta.get("ship_classes"))

        # --- Years ---
        print_if_exists("Years Mentioned are", meta.get("years_mentioned"))

        # --- Rates ---
        print_if_exists("Rates Mentioned", meta.get("rates_mentioned"))

        # --- Shipyards ---
        print_if_exists("Shipyards Mentioned are", meta.get("shipyards_mentioned"))

        # --- Text preview ---
        print("\nNode Content:")
        print(text)


# Creates an index if one doesnt exist, opens existing index and vector store
def load_index():
    storage_dir = STORAGE_DIR

    # Check if the storage exists and if not, create a new one
    if not storage_dir.exists() or not DOCS_STORE.exists():
        print("No valid persisted vector db found, creating a new one...")
        create_index(DOCS_DIR)
        return load_index()

    # Open the vector store
    print("Loading existing vector db...")
    client = qdrant_client.QdrantClient(path=str(DB_DIR))
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(storage_dir),
    )
    
    index = load_index_from_storage(storage_context, index_id=INDEX_ID)
    return index


class ScoreThresholdFilter(BaseNodePostprocessor):
    ratio: float = SCORE_RATIO
    debug: bool = False

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        nodes = sorted(nodes, key=lambda x: x.score or 0, reverse=True)
        top_score = nodes[0].score or 0
        threshold = top_score * self.ratio

        if self.debug:
            print("\n--- BEFORE FILTER ---")
            for i, node in enumerate(nodes):
                print(f"{i+1}. score={node.score}")

        filtered = [
            node for node in nodes
            if node.score is not None and node.score >= threshold
        ]

        if self.debug:
            print(f"\nThreshold: {threshold}")
            print("--- AFTER FILTER ---")
            for i, node in enumerate(filtered):
                print(f"{i+1}. score={node.score}")
            print(f"Kept {len(filtered)} / {len(nodes)} nodes\n")

        return filtered if filtered else nodes[:1]
    

def initialize_query_engine(
    index,
    template=QA_TEMPLATE,
    metadata_filters=None,
    similarity_top_k: int | None = None,
    mixed_top_k: int | None = None,
    score_ratio: float | None = None,
    rerank_top_n: int | None = None,
    include_threshold_postprocessor: bool = True,
    include_reorder_postprocessor: bool = True,
):
    print("Docstore keys:", len(index.docstore.docs))
    similarity_top_k = similarity_top_k or SIMILARITY_TOP_K
    mixed_top_k = mixed_top_k or MIXED_TOP_K
    score_ratio = score_ratio if score_ratio is not None else SCORE_RATIO
    rerank_top_n = rerank_top_n or RERANK_MODEL_CONFIG.get("top_n", 5)

    llama_filters = build_metadata_filters(metadata_filters)
    vec_retriever=VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            embed_model=Settings.embed_model,
            filters=llama_filters
    )
    hybrid = HybridRetriever(
        vec_retriever = vec_retriever,

        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.storage_context.docstore,
            similarity_top_k=similarity_top_k,
        ),

        final_top_k=mixed_top_k,   # let reranker see more candidates
        rrf_k=60,
        debug=False,
    )

    synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=template,
        response_mode="compact",
        streaming=False, #turned off because it apparently increases lag and wasnt working for me
    )

    if metadata_filters:
        retriever = vec_retriever   # filtered only
    else:
        retriever = hybrid          # normal hybrid

    rerank_config = dict(RERANK_MODEL_CONFIG)
    rerank_config["top_n"] = rerank_top_n
    node_postprocessors = [
        SentenceTransformerRerank(**rerank_config),
    ]
    if include_threshold_postprocessor:
        node_postprocessors.append(ScoreThresholdFilter(ratio=score_ratio))
    if include_reorder_postprocessor:
        node_postprocessors.append(LongContextReorder())

    query_engine = RetrieverQueryEngine(
        retriever = retriever,
        response_synthesizer = synthesizer,
        node_postprocessors = node_postprocessors
    )
    return query_engine, hybrid


def ask_question(query_engine, hybrid, question: str, see_results):
    if not question:
        print("> MAA Assistant: Please provide a question.")
        return

    question = question.strip()

    total_start = time.time()

    print("> MAA Assistant: Retrieving relevant information...")
    retrieval_start = time.time()

    #semantic_nodes = hybrid._vec.retrieve(QueryBundle(query_str=question))
    #keyword_nodes = hybrid._bm25.retrieve(QueryBundle(query_str=question))
    hybrid_nodes = hybrid.retrieve(QueryBundle(query_str=question))

    retrieval_time = time.time() - retrieval_start

    #print(f"> MAA Assistant: Retrieved {len(semantic_nodes)} semantic nodes.")
    #print(f"> MAA Assistant: Retrieved {len(keyword_nodes)} keyword nodes.")
    print(f"> MAA Assistant: Retrieved {len(hybrid_nodes)} fused candidate nodes.")
    print(f"> MAA Assistant: Retrieval took {retrieval_time:.2f}s")

    #if see_results.lower().strip() == "y":
        #print("\n--- SEMANTIC RESULTS ---")
        #show_nodes(semantic_nodes)

        #print("\n--- KEYWORD RESULTS ---")
        #show_nodes(keyword_nodes)

        #print("\n--- HYBRID FUSED RESULTS ---")
        #show_nodes(hybrid_nodes)

    if not hybrid_nodes:
        print("> MAA Assistant: No relevant results found.")
        return

    print("> MAA Assistant: Working on response...")
    gen_start = time.time()

    if see_results.lower().strip() == "y":
        print_final_context(query_engine, hybrid_nodes, question)

    response = query_engine.query(question)


    gen_time = time.time() - gen_start
    #total_time = time.time() - total_start

    if response:
        print(f"> MAA Assistant: {response}")
    else:
        print("> MAA Assistant: Sorry, I don't have an answer for that")

    print(f"> MAA Assistant: Generation took {gen_time:.2f}s")

def build_metadata_filters(metadata_filters: dict | None):
    if not metadata_filters:
        return None

    return MetadataFilters(
        filters=[
            ExactMatchFilter(key=key, value=value)
            for key, value in metadata_filters.items()
            if value not in {None, "", "Any"}
        ]
    )

def updateModel(index, template_choice: str, template_current: str, response_length: str, response_model, current_max_tokens: int):
    config = GENERATIVE_MODEL_CONFIG.copy()

    if response_length == "short":
        response_tokens = 100
    elif response_length == "long":
        response_tokens = 600
    else:
        response_tokens = 300

    template_map = {
        "q": ("Q/A", QA_TEMPLATE),
        "s": ("Summary", SUMMARY_TEMPLATE),
        "e": ("Exposure Analysis", EXPOSURE_ANALYSIS_TEMPLATE),
        "t": ("Timeline", TIMELINE_TEMPLATE),
        "r": ("Reference", REFERENCE_EVIDENCE_TEMPLATE),
        "c": ("Comparison", COMPARISON_TEMPLATE),
    }

    if template_choice not in template_map:
        print("> MAA Assistant: Invalid template choice. Using Q/A template.\n")
        template_choice = "q"

    template_name, selected_template = template_map[template_choice]

    template_changed = template_choice != template_current
    tokens_changed = current_max_tokens != response_tokens

    if not template_changed and not tokens_changed:
        print("> MAA Assistant: Model settings unchanged.\n")
        return response_model, None, None, current_max_tokens, template_current

    print(f"> MAA Assistant: Setting maximum response length to {response_tokens} tokens.")
    print(f"> MAA Assistant: Using {template_name} template.\n")

    config["max_new_tokens"] = response_tokens
    response_model = HuggingFaceLLM(**config)
    Settings.llm = response_model

    qe, hybrid = initialize_query_engine(
        index,
        template=selected_template,
        metadata_filters=None
    )

    current_max_tokens = response_tokens
    template_current = template_choice

    return response_model, qe, hybrid, current_max_tokens, template_current


def main():
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(**EMBED_MODEL_CONFIG)
    current_max_tokens = GENERATIVE_MODEL_CONFIG["max_new_tokens"]
    template_current = "q"

    # If refreshing DB
    start_or_refresh = ""
    while start_or_refresh not in {"s", "r"}:
        start_or_refresh = input("> MAA Assistant: Hello, would you like to start the system or first refresh the index (s for start, r for refresh): ").strip("\n")
        start_or_refresh = start_or_refresh.lower()
    if start_or_refresh == "r":
        create_index(DOCS_DIR)
        print("> MAA Assistant: Index refreshed.")

    # Load Index and Initialize QE
    print("Loading index...")
    index = load_index()
    print("Docstore keys after reload:", len(index.docstore.docs))
    response_model = HuggingFaceLLM(**GENERATIVE_MODEL_CONFIG)
    Settings.llm = response_model
    qe, hybrid = initialize_query_engine(index, template=QA_TEMPLATE)
    choice = ""

    # Main Loop
    while choice not in {"Exit", "exit"}:
        choice = input(
            "> MAA Assistant: Hello, what would you like to do? "
            "(Ask a question (q) or Exit): "
        ).strip("\n")

        if choice == "q":
            question = input("> MAA Assistant: What is your question? ").strip("\n")
            see_results = input("> MAA Assistant: Would you like to see the returned results? ")
            response_length = input( "> MAA Assistant: Response length? (short / medium / long): " ).strip().lower()
            template_choice = input(
                "> MAA Assistant: Which template would you like to use? "
                "(q = Q/A, s = summary, e = exposure, t = timeline, r = reference, c = comparison): "
            ).strip().lower()

            new_model, new_qe, new_hybrid, current_max_tokens, template_current = updateModel(
                index,
                template_choice,
                template_current,
                response_length,
                response_model,
                current_max_tokens
            )
            response_model = new_model
            Settings.llm = response_model

            if new_qe is not None:
                qe = new_qe
                hybrid = new_hybrid

            ask_question(qe, hybrid, question, see_results)
        elif choice == "Exit":
            print("> MAA Assistant: Goodbye!")
        else:
            print("> MAA Assistant: Invalid choice. Please try again.")


if __name__ == "__main__":
    main()