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
from models.LLM_Header_File import HuggingFaceLLM
from setup_index.create_index import create_index


# -----------------------------
# Global model setup
# -----------------------------
Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r".\models\ai_models\bge-m3-st",
    max_length=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


BASE_DIR = Path(__file__).resolve().parent
db_dir = BASE_DIR / "vector_db"
DOCS_ROOT = BASE_DIR / "Docs"
INDEX_ID = "main_index"
COLLECTION_NAME = "test_store"

# -----------------------------
# Better hybrid retriever:
# uses Reciprocal Rank Fusion (RRF)
# -----------------------------
class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever using:
    - vector retrieval
    - BM25 retrieval
    - dedupe by node id
    - reciprocal rank fusion (RRF)

    This avoids unfairly favoring vector results and gives BM25
    a real chance to contribute before reranking.
    """
    def __init__(
        self,
        vec_retriever,
        bm25_retriever,
        final_top_k: int = 15,
        rrf_k: int = 60,
        debug: bool = True,
    ):
        super().__init__()
        self._vec = vec_retriever
        self._bm25 = bm25_retriever
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        self._debug = debug

    def _get_node_id(self, nws: NodeWithScore) -> str:
        node = getattr(nws, "node", nws)
        nid = (
            getattr(node, "node_id", None)
            or getattr(node, "id_", None)
            or getattr(nws, "id_", None)
        )

        if nid is None:
            text = getattr(node, "text", "") or (
                node.get_content() if hasattr(node, "get_content") else ""
            )
            nid = str(hash(text))

        return nid

    def _preview(self, nws: NodeWithScore, limit: int = 120) -> str:
        node = getattr(nws, "node", nws)
        text = getattr(node, "text", "") or (
            node.get_content() if hasattr(node, "get_content") else ""
        )
        return text[:limit].replace("\n", " ")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = query_bundle.query_str

        vec_nodes = self._vec.retrieve(q)
        bm25_nodes = self._bm25.retrieve(q)

        if self._debug:
            print("\nVECTOR RESULTS:")
            for i, n in enumerate(vec_nodes[:5]):
                print(i, n.score, self._preview(n))

            print("\nBM25 RESULTS:")
            for i, n in enumerate(bm25_nodes[:5]):
                print(i, n.score, self._preview(n))

        id_to_node: Dict[str, NodeWithScore] = {}
        fused_scores: Dict[str, float] = {}

        # Vector contribution
        for rank, nws in enumerate(vec_nodes, start=1):
            nid = self._get_node_id(nws)
            id_to_node[nid] = nws
            fused_scores[nid] = fused_scores.get(nid, 0.0) + 1.0 / (self._rrf_k + rank)

        # BM25 contribution
        for rank, nws in enumerate(bm25_nodes, start=1):
            nid = self._get_node_id(nws)
            if nid not in id_to_node:
                id_to_node[nid] = nws
            fused_scores[nid] = fused_scores.get(nid, 0.0) + 1.0 / (self._rrf_k + rank)

        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda nid: fused_scores[nid],
            reverse=True,
        )

        results: List[NodeWithScore] = []
        for nid in ranked_ids[: self._final_top_k]:
            nws = id_to_node[nid]
            results.append(NodeWithScore(node=nws.node, score=fused_scores[nid]))

        if self._debug:
            print("\nFUSED RESULTS:")
            for i, n in enumerate(results[:10]):
                print(i, n.score, self._preview(n))

        return results


qa_template = PromptTemplate(
    "You must answer using ONLY the context.\n"
    "If the context does not explicitly contain the answer, reply exactly:\n"
    "I cannot find this in the documents.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Do not add facts not supported by the context.\n"
    "Answer:"
)


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
        print("   Text_preview:", repr(text[:100]))


def load_index():
    storage_dir = Path(db_dir) / "storage"

    if not storage_dir.exists():
        print("No vector db found, creating a new one...")
        create_index(DOCS_ROOT)
        return load_index()

    print("Loading existing vector db...")
    client = qdrant_client.QdrantClient(path=str(db_dir))
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
            similarity_top_k=12,
            embed_model=Settings.embed_model,
        ),

        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.storage_context.docstore,
            similarity_top_k=12,
        ),

        final_top_k=15,   # let reranker see more candidates
        rrf_k=60,
        debug=False,
    )

    synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=qa_template,
        response_mode="compact",
        streaming=True,
    )

    query_engine = RetrieverQueryEngine(
        retriever = hybrid,
        response_synthesizer = synthesizer,
        node_postprocessors = [
            SentenceTransformerRerank(
                model=str(BASE_DIR / "models" / "ai_models" / "bge-reranker-base"),
                top_n=8
            ),
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


if __name__ == "__main__":
    Settings.llm = None
    start_or_refresh = ""
    while start_or_refresh not in {"s", "r"}:
        start_or_refresh = input("> MAA Assistant: Hello, would you like to start the system or first refresh the index (s for start, r for refresh): ").strip("\n")
        start_or_refresh = start_or_refresh.lower()

    if start_or_refresh == "r":
        create_index(DOCS_ROOT)
        print("> MAA Assistant: Index refreshed.")

    print("Loading index...")
    index = load_index()
    print("Docstore keys after reload:", len(index.docstore.docs))
    Settings.llm = HuggingFaceLLM(
        model_path=str(BASE_DIR / "models" / "ai_models" / "flan-t5-large"),
        temperature=0.1,
        do_sample=False,
        max_new_tokens=256,
        top_p=1.0,
        repetition_penalty=1.05,
        context_window=4096,
    )

    qe, hybrid = initialize_query_engine(index)
    choice = ""

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