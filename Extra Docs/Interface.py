import os
import torch
from pathlib import Path
import qdrant_client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.storage import StorageContext
from setup_index.create_index import create_index

# Hard-force offline mode (optional but recommended)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Set LLM and embedding model
Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(
    model_name=r".\models\bge-m3-st",   # <-- local path works
    max_length=1024,               # bge-m3 supports long; pick what you can afford
    device="cuda" if torch.cuda.is_available() else "cpu",  # use gpu for embedding if available
)

# Path to database
db_dir = Path(r"C:\Users\Christian\Documents\Local_Code\MAA-RAG\Code\vector_db")

# Loads index from vector dB and creates it if it doesn't exist
def load_index():
    storage_dir = Path(db_dir) / "storage"

    if not storage_dir.exists():
        print("No vector db found, creating a new one...")
        index = create_index()
        return load_index()
    else:
        print("Loading existing vector db...")
        print("Creating vector store...")
        client = qdrant_client.QdrantClient(path=db_dir)
        vector_store = QdrantVectorStore(client=client, collection_name="test_store")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(Path(db_dir) / "storage"),   # <-- must match your build path
        )
        index = load_index_from_storage(storage_context)
    return index


# Asks a question using the query engine
def ask_question(index, question: str):
    if not question:
        print("> MAA Assistant: Please provide a question.")
        return
    question = question.strip()
    retriever = index.as_retriever(similarity_top_k=8)  # widen for recall
    nodes = retriever.retrieve(question)

    if not nodes:
        print("> MAA Assistant: No relevant results found.")
        return

    print("> MAA Assistant: Here are the most relevant passages:\n")
    for i, n in enumerate(nodes[:5], start=1):
        text = getattr(n, "text", None) or (n.get_content() if hasattr(n, "get_content") else "")
        meta = getattr(n, "metadata", {}) or {}
        score = getattr(n, "score", None)

        print(f"--- Result {i} ---")
        if score is not None:
            print(f"score: {score:.4f}")
        print(f"title: {meta.get('title')}")
        print(f"file_path: {meta.get('file_path')}")
        print(f"page: {meta.get('page')}")
        print(text[:500])
        print()


if __name__ == "__main__":
    print("Loading index...")
    index = load_index()
    choice = ""

    # Main interaction loop
    while choice != "Exit" and choice != "exit":
        choice = input("> MAA Assistant: Hello, what would you like to do? (Ask a question (q), refresh index, Exit): ").strip("\n")

        # Ask a question
        if choice == "q":
            question = input("> MAA Assistant: What is your question? ").strip("\n")
            ask_question(index, question)

        # Refresh the index
        elif choice == "refresh index":
            index = create_index()
            choice = ""
            print("> MAA Assistant: Index refreshed.")

        # Exit the program
        elif choice == "Exit":
            print("> MAA Assistant: Goodbye!")

        # Handle invalid input
        else:
            print("> MAA Assistant: Invalid choice. Please try again.")
