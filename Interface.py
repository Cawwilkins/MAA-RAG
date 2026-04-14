from pathlib import Path
import qdrant_client
import torch
import time
from typing import List, Set, Dict
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
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.prompts import PromptTemplate
from models.LLM_Header_File import HuggingFaceLLM, HybridRetriever
from setup_index.create_index import create_index
from config import EMBED_MODEL_CONFIG, DB_DIR, STORAGE_DIR, DOCS_DIR, COLLECTION_NAME, INDEX_ID, QA_TEMPLATE, RERANK_MODEL_CONFIG, GENERATIVE_MODEL_CONFIG, SIMILARITY_TOP_K, MIXED_TOP_K


def show_nodes(nodes, show_score=True):
    for i, node in enumerate(nodes[:5]):
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", {}) or {}

        # Print the metadata you care about (safe .get calls)
        print(f"\n--- Node {i + 1} ---")
        if (show_score): print("Score: ", getattr(node, "score", None))
        print("Node Information:")
        print("Title:", meta.get("title"))
        print("OCR or Text-Based PDF:", meta.get("source"))
        print("File Path:", meta.get("file_path"))
        print("Document Page #:", meta.get("page"))
        print("   Text_preview:", repr(text))


# Creates an index if one doesnt exist, opens existing index and vector store
def load_index():
    storage_dir = STORAGE_DIR

    # Check if the storage exists and if not, create a new one
    if not storage_dir.exists():
        print("No vector db found, creating a new one...")
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


def initialize_query_engine(index):

    print("Docstore keys:", len(index.docstore.docs))
    hybrid = HybridRetriever(
        vec_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=SIMILARITY_TOP_K,
            embed_model=Settings.embed_model,
        ),

        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.storage_context.docstore,
            similarity_top_k=SIMILARITY_TOP_K,
        ),

        final_top_k=MIXED_TOP_K,   # let reranker see more candidates
        rrf_k=60,
        debug=False,
    )

    synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=QA_TEMPLATE,
        response_mode="compact",
        streaming=False, #turned off because it apparently increases lag and wasnt working for me
    )

    query_engine = RetrieverQueryEngine(
        retriever = hybrid,
        response_synthesizer = synthesizer,
        node_postprocessors = [
            SentenceTransformerRerank(**RERANK_MODEL_CONFIG),
            LongContextReorder(),
        ]
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

    semantic_nodes = hybrid._vec.retrieve(question)
    keyword_nodes = hybrid._bm25.retrieve(question)
    hybrid_nodes = hybrid.retrieve(question)

    retrieval_time = time.time() - retrieval_start

    print(f"> MAA Assistant: Retrieved {len(semantic_nodes)} semantic nodes.")
    print(f"> MAA Assistant: Retrieved {len(keyword_nodes)} keyword nodes.")
    print(f"> MAA Assistant: Retrieved {len(hybrid_nodes)} fused candidate nodes.")
    print(f"> MAA Assistant: Retrieval took {retrieval_time:.2f}s")

    if see_results.lower().strip() == "y":
        print("\n--- SEMANTIC RESULTS ---")
        show_nodes(semantic_nodes)

        print("\n--- KEYWORD RESULTS ---")
        show_nodes(keyword_nodes)

        print("\n--- HYBRID FUSED RESULTS ---")
        show_nodes(hybrid_nodes)

    if not hybrid_nodes:
        print("> MAA Assistant: No relevant results found.")
        return

    print("> MAA Assistant: Working on response...")
    gen_start = time.time()

    response = query_engine.query(question)

    gen_time = time.time() - gen_start
    total_time = time.time() - total_start

    if response:
        print(f"> MAA Assistant: {response}")
    else:
        print("> MAA Assistant: Sorry, I don't have an answer for that")

    print(f"> MAA Assistant: Generation took {gen_time:.2f}s")
    print(f"> MAA Assistant: Total time {total_time:.2f}s")


def main():
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(**EMBED_MODEL_CONFIG)

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
    Settings.llm = HuggingFaceLLM(**GENERATIVE_MODEL_CONFIG)
    qe, hybrid = initialize_query_engine(index)
    choice = ""

    # Main Loop
    while choice not in {"Exit", "exit"}:
        choice = input(
            "> MAA Assistant: Hello, what would you like to do? "
            "(Ask a question (q) or Exit): "
        ).strip("\n")

        if choice == "q":
            question = input("> MAA Assistant: What is your question? ").strip("\n")
            see_results = input("> MAA Assistant: Would you like to see the resturned results? ")
            ask_question(qe, hybrid, question, see_results)
        elif choice == "Exit":
            print("> MAA Assistant: Goodbye!")
        else:
            print("> MAA Assistant: Invalid choice. Please try again.")


if __name__ == "__main__":
    main()