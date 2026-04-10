import os
from llama_index.core import Document, VectorStoreIndex, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import traceback
import qdrant_client
from config import INDEX_ID, DB_DIR, STORAGE_DIR, EMBED_MODEL, COLLECTION_NAME

# Hard-force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

Settings.llm = None
Settings.embed_model = EMBED_MODEL

# Debug function to see node info
def debug_print_nodes(nodes, n: int = 3) -> None:
    print("\n==== NODE DEBUG SAMPLE ====")
    for i, node in enumerate(nodes[:n]):
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
        text = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", {}) or {}
        rels = getattr(node, "relationships", {}) or {}

        print(f"\n--- Node {i} ---")
        print("node_id:", node_id)
        print("text_len:", len(text))
        print("text_preview:", repr(text[:250]))
        print("metadata:")
        print("  title:", meta.get("title"))
        print("  doc_type:", meta.get("doc_type"))
        print("  source:", meta.get("source"))
        print("  file_path:", meta.get("file_path"))
        print("  page:", meta.get("page"))
        print("  source_id:", meta.get("source_id"))

        extractor_keys = [k for k in meta.keys() if "keyword" in k.lower() or "summary" in k.lower()]
        if extractor_keys:
            print("extractor_fields:")
            for k in extractor_keys:
                print(f"  {k}: {meta.get(k)}")

        if rels:
            print("relationships keys:", list(rels.keys()))
            for rk, rv in rels.items():
                rid = getattr(rv, "node_id", None) or getattr(rv, "id_", None) or str(rv)
                print(f"  {rk}: {rid[:120]}")


# Delete existing nodes whos id matches incoming ID
def delete_old_docstore_nodes(storage_context: StorageContext, source_ids_to_replace: set[str]) -> None:
    """
    Remove existing nodes from the docstore whose metadata.source_id matches
    one of the incoming source IDs.
    """
    docstore_docs = getattr(storage_context.docstore, "docs", {}) or {}
    node_ids_to_delete = []

    for node_id, node in docstore_docs.items():
        metadata = getattr(node, "metadata", {}) or {}
        if metadata.get("source_id") in source_ids_to_replace:
            node_ids_to_delete.append(node_id)

    if not node_ids_to_delete:
        print("No matching existing docstore nodes found to delete.")
        return

    print(f"Deleting {len(node_ids_to_delete)} old node(s) from docstore...")
    for node_id in node_ids_to_delete:
        try:
            storage_context.docstore.delete_document(node_id)
        except Exception as e:
            print(f"Warning: failed to delete docstore node {node_id}: {e}")


# Remove from qdrant any nodes from documents that have since been updated
def delete_old_qdrant_points(client: qdrant_client.QdrantClient, source_ids_to_replace: set[str]) -> None:
    """
    Remove existing vectors from Qdrant whose payload source_id matches
    one of the incoming source IDs.
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    for source_id in source_ids_to_replace:
        try:
            print(f"Deleting old Qdrant points for source_id={source_id}")
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(value=source_id),
                        )
                    ]
                ),
            )
        except Exception as e:
            print(f"Warning: failed to delete Qdrant points for {source_id}: {e}")


# Embeds documents and adds them to vector store
def doc_embed_store(docs: list[Document]) -> VectorStoreIndex | None:
    Settings.llm = None

    # Ensure Docs have actually been passed
    if not docs:
        print("No documents to embed.")
        return None

    #??
    client = None

    try:
        print("Initializing Qdrant and storage context...")

        # Check if sotrage_dir exists and if not make it
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Storage directory ensured at: {STORAGE_DIR}\n")

        # Collect source_ids from incoming docs
        source_ids_to_replace = {
            doc.metadata.get("source_id")
            for doc in docs
            if doc.metadata.get("source_id")
        }

        # Determine if anything to replace
        if not source_ids_to_replace:
            print("Warning: no source_id values found in incoming docs.")
        else:
            print(f"Incoming source_ids to upsert: {sorted(source_ids_to_replace)}")

        # Initialize Qdrant client
        client = qdrant_client.QdrantClient(path=str(DB_DIR))
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        print("Qdrant client and vector store initialized.\n")

        # Set Docstore path
        docstore_path = STORAGE_DIR / "docstore.json"


        # If it exists, load the storage_context
        if docstore_path.exists():
            existing_storage = True
            storage_context = StorageContext.from_defaults(
                persist_dir=str(STORAGE_DIR),
                vector_store=vector_store,
            )
            print("Loaded existing persisted storage context.\n")
            
            # Delete old versions before inserting new ones
            if source_ids_to_replace:
                delete_old_docstore_nodes(storage_context, source_ids_to_replace)
                delete_old_qdrant_points(client, source_ids_to_replace)

        
        # Otherwise make a new one
        else:
            existing_storage = False
            docstore = SimpleDocumentStore()
            storage_context = StorageContext.from_defaults(
                docstore=docstore,
                vector_store=vector_store,
            )
            print("Created new storage context.\n")

        # Initialize pipeline
        print("Running ingestion pipeline... (chunking phase)")
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=400,
                    chunk_overlap=100,
                    include_prev_next_rel=True,
                    paragraph_separator="\n\n",
                ),
            ],
        )
        print("Pipeline initialized with sentence splitter transformation.\n")

        # Transform chunks into nodes
        nodes = pipeline.run(documents=docs)

        debug_print_nodes(nodes, n=5)
        print(f"✅ Pipeline completed. Generated {len(nodes)} nodes.")

        # Add nodes to storage
        print("Adding nodes to docstore...")
        storage_context.docstore.add_documents(nodes)
        print("Docstore keys before persist:", len(storage_context.docstore.docs))
        print("Nodes added to docstore.\n")

        # Initialize vector store index
        print("Building vector index and embedding nodes...")
        if existing_storage == False:
            index = VectorStoreIndex(
                [],
                storage_context=storage_context,
                embed_model=Settings.embed_model,
            )
            index.set_index_id(INDEX_ID)
            index.insert_nodes(nodes)
            print("New VectorStoreIndex created and nodes inserted.\n")

        else:
            index = load_index_from_storage(storage_context, index_id=INDEX_ID)
            index.insert_nodes(nodes)
            print("Existing VectorStoreIndex loaded and nodes inserted.\n")

        print("Persisting docstore and index to disk...")
        index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        print(f"✅ Docstore and index persisted successfully to: {STORAGE_DIR}")
        return index

    except Exception as e:
        print(f"❌ Error during embedding: {e}")
        traceback.print_exc()
        return None

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass