# Load and embed documents
import traceback
from llama_index.core import VectorStoreIndex
from setup_index.feed_documents import feed_documents
from setup_index.doc_embed_store import doc_embed_store


# Creates vector store index from docs in specified directory
def create_index() -> VectorStoreIndex | None:
    try:
        print ("Loading documents...    ")
        documents = feed_documents()
        try:
            print("Embedding and storing documents...    ")
            index = doc_embed_store(documents)
            print("Documents embedded and stored successfully.")
            return index
        except Exception as e:
            print(f"Error occurred while embedding documents: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error occurred while loading documents: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    index = create_index()
    if index is not None:
        print("Index created successfully.")
    else:
        print("Failed to create index.")