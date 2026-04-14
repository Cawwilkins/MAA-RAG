from pathlib import Path
import torch
import psutil
from llama_index.core.prompts import PromptTemplate




BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "vector_db"
STORAGE_DIR = DB_DIR / "storage"
DOCS_DIR = BASE_DIR / "Docs"  #was docs_root
INDEX_ID = "main_index"
COLLECTION_NAME = "test_store"
STORED_FILES_PATH = BASE_DIR / "setup_index" / "stored_files.json"
DEFAULT_MAX_UPSERT_FILES = 20
DOCS_STORE = STORAGE_DIR / "docstore.json"

# Model Paths
MODEL_PATH = BASE_DIR / "models" / "ai_models"
GENERATIVE_MODEL_PATH = MODEL_PATH / "flan-t5-large"
EMBED_MODEL_PATH = MODEL_PATH / "bge-m3-st"
RERANK_MODEL_PATH = MODEL_PATH / "bge-reranker-base"

# Model Tuning
MAX_NEW_TOKENS = 100 # has to match specs of the model (256 max), lower to improve speed
CONTEXT_WINDOW = 512 # has to match the specs of the model
MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.9
MODEL_DO_SAMPLE = False 

# Embed Model Tuning
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# Retrieval
SIMILARITY_TOP_K = 20
MIXED_TOP_K = 30

# MISC
PHYSICAL_CORES = psutil.cpu_count(logical=False)
OCR_MAX_WORKERS = max(1, PHYSICAL_CORES - 1)
OCR_THREAD_COUNT = 1
EMBED_MAX_WORKERS = OCR_MAX_WORKERS

RERANK_MODEL_CONFIG = {
    "model":str(RERANK_MODEL_PATH),
    "top_n":5,
    "device":"cpu",
    #keep_retrieval_score=True #may be good to implment so see side by side of scores
    #cache_folder, cross_encoder_kwargs, not necessary
}

EMBED_MODEL_CONFIG = {
    "model_name":str(EMBED_MODEL_PATH),
    "max_length":1024, #this may be larger than bge-m3-st actually accepts but already shrunk to 400 by sentence splitter
    "device":"cpu",
    "query_instruction":"Represent this question for searching: ",
    "text_instruction":"Represent this document for retrieval: ",
    "parallel_process":False, #will allow for processing across independed processes
    #num_workers=EMBED_MAX_WORKERS, #this is for some other version of llamaindex
    "show_progress_bar":True,
    "embed_batch_size":32,
}

SENTENCE_SPLITTER_CONFIG = {
    "chunk_size":CHUNK_SIZE,
    "chunk_overlap":CHUNK_OVERLAP,
    "include_prev_next_rel":True,
    "paragraph_separator":"\n\n",
}

GENERATIVE_MODEL_CONFIG = {
    "model_path":str(GENERATIVE_MODEL_PATH),
    "do_sample":False,
    "max_new_tokens":MAX_NEW_TOKENS,
    "repetition_penalty":1.05,
    "context_window":CONTEXT_WINDOW,
}

QA_TEMPLATE = PromptTemplate(
    "You must answer using ONLY the context.\n"
    "If the context does not explicitly contain the answer, reply exactly:\n"
    "I cannot find this in the documents.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Do not add facts not supported by the context.\n"
    "Answer:"
)


# add configs for the response synth and retrievers

#turning off streaming, and decreasing the new chucks to 100 significantly improved speed