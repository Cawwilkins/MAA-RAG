import os
from random import sample
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import traceback
import qdrant_client
import torch

from setup_index.feed_documents import feed_documents

# Hard-force offline mode (optional but recommended)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

EMBED_MODEL_PATH = r".\models\ai_models\bge-m3-st"

embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL_PATH,   # <-- local path works
    max_length=1024,               # bge-m3 supports long; pick what you can afford
    device="cuda" if torch.cuda.is_available() else "cpu",  # use gpu for embedding if available
)
    
Settings.llm = None  # Disable LLM calls in extractors for faster embedding; we just want raw keywords/summaries
Settings.embed_model = embed_model  # Use default embedding model; set to None to avoid unnecessary loading if not used in extractors

def debug_print_nodes(nodes, n: int = 3) -> None:

    print("\n==== NODE DEBUG SAMPLE ====")
    for i, node in enumerate(nodes[:n]):
        # Node ID varies a bit by LlamaIndex version; these are the common fields
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)

        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", {}) or {}

        # Relationships: prev/next etc (since include_prev_next_rel=True)
        rels = getattr(node, "relationships", {}) or {}

        print(f"\n--- Node {i} ---")
        print("node_id:", node_id)
        print("text_len:", len(text))
        print("text_preview:", repr(text[:250]))

        # Print the metadata you care about (safe .get calls)
        print("metadata:")
        print("  title:", meta.get("title"))
        print("  doc_type:", meta.get("doc_type"))
        print("  source:", meta.get("source"))
        print("  file_path:", meta.get("file_path"))
        print("  page:", meta.get("page"))

        # Keywords / summary often land in metadata; print anything extractor-like
        extractor_keys = [k for k in meta.keys() if "keyword" in k.lower() or "summary" in k.lower()]
        if extractor_keys:
            print("extractor_fields:")
            for k in extractor_keys:
                print(f"  {k}: {meta.get(k)}")

        # Show relationship keys and a compact view of IDs if present
        if rels:
            print("relationships keys:", list(rels.keys()))
            # Try to show any linked node ids without overprinting
            for rk, rv in rels.items():
                # relationship objects vary; attempt a readable id
                rid = getattr(rv, "node_id", None) or getattr(rv, "id_", None) or str(rv)
                print(f"  {rk}: {rid[:120]}")


def doc_embed_store(docs: list[Document]) -> VectorStoreIndex | None:
    Settings.llm = None  # Disable LLM calls in extractors for faster embedding; we just want raw keywords/summaries
    if not docs:
        print("No documents to embed.")
        return None

    try:
        print("Initializing Qdrant and storage context...")

        # === 1. Create storage directories ===
        db_dir = Path(r"C:\Users\Christian\Documents\Local_Code\MAA-RAG\Code\vector_db")
        storage_path = Path(r"C:\Users\Christian\Documents\Local_Code\MAA-RAG\Code\vector_db\storage")
        storage_path.mkdir(parents=True, exist_ok=True)
        print(f"Storage directories ensured at: {storage_path}\n")

        # === 2. Set up Qdrant and storage context ===
        client = qdrant_client.QdrantClient(path=str(db_dir))
        vector_store = QdrantVectorStore(client=client, collection_name="test_store")
        print("Qdrant client and vector store initialized.\n")

        # Create a fresh in-memory docstore (so it doesn't try to load a missing file)
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            vector_store=vector_store
        )
        print("Storage context created with in-memory docstore and Qdrant vector store.\n")

        # === 3. Run the ingestion pipeline (in-memory only) ===
        print("Running ingestion pipeline... (Enrich Phase)")
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    include_prev_next_rel=True,
                    paragraph_separator="\n\n"
                ),
            ],
            vector_store=vector_store,
        )
        print("Pipeline initialized with no transformations.\n")

        nodes = pipeline.run(documents=docs, show_progress=True)

        # Metadata keys: creation_date, doc_type, file_name, file_path, file_size, file_type, last_modified_date, page, source, title

        debug_print_nodes(nodes, n=5)
        print(f"✅ Pipeline completed. Generated {len(nodes)} nodes.")

        # === 4. Store the text nodes explicitly for BM25 retrieval ===
        print("Adding nodes to docstore...")
        storage_context.docstore.add_documents(nodes)

        print("Documents added to Docstore.\n")
        # === 5. Build and persist the vector index ===
        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context, 
            show_progress=True,
            embed_model=Settings.embed_model
        )
        print("VectorStoreIndex created with embedded nodes.\n")
        index.storage_context.persist(persist_dir=str(storage_path))

        print(f"✅ Docstore and index persisted successfully to: {storage_path}")
        return index

    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        traceback.print_exc()
        return None

    finally:
        try:
            client.close()
        except Exception:
            pass

if __name__ == "__main__":
    docs = feed_documents()
    doc_embed_store(docs)