from pathlib import Path
import torch
import psutil
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate



BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "vector_db"
STORAGE_DIR = DB_DIR / "storage"
DOCS_DIR = BASE_DIR / "Docs"  #was docs_root
INDEX_ID = "main_index"
COLLECTION_NAME = "test_store"
STORED_FILES_PATH = BASE_DIR / "setup_index" / "stored_files.json"
DEFAULT_MAX_UPSERT_FILES = 20

# Model Paths
MODEL_PATH = BASE_DIR / "models" / "ai_models"
GENERATIVE_MODEL_PATH = MODEL_PATH / "flan-t5-large"
EMBED_MODEL_PATH = MODEL_PATH / "bge-m3-st"
RERANK_MODEL_PATH = MODEL_PATH / "bge-reranker-base"

# Model Tuning
MAX_NEW_TOKENS = 256 # has to match specs of the model
CONTEXT_WINDOW = 512 # has to match the specs of the model
MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.9
MODEL_DO_SAMPLE = False 

RERANK_MODEL = SentenceTransformerRerank(
    model=str(RERANK_MODEL_PATH),
    top_n=5
)

EMBED_MODEL = HuggingFaceEmbedding(
    model_name=str(EMBED_MODEL_PATH),
    max_length=1024,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

QA_TEMPLATE = PromptTemplate(
    "You must answer using ONLY the context.\n"
    "If the context does not explicitly contain the answer, reply exactly:\n"
    "I cannot find this in the documents.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Do not add facts not supported by the context.\n"
    "Answer:"
)


# Retrieval
SIMILARITY_TOP_K = 20
MIXED_TOP_K = 30

# MISC
PHYSICAL_CORES = psutil.cpu_count(logical=False)
OCR_MAX_WORKERS = max(1, PHYSICAL_CORES - 1)
OCR_THREAD_COUNT = 1