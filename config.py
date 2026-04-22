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
WPD_PATH = BASE_DIR / "wpd_support" / "wpd2text" / "wpd2text.exe"

ACCEPTED_FILE_TYPES = {".pdf", ".wpd"}

# Model Paths
MODEL_PATH = BASE_DIR / "models" / "ai_models"
GENERATIVE_MODEL_PATH = MODEL_PATH / "llama-3.2-1b-instruct" #"phi-3-mini-4k-instruct"
EMBED_MODEL_PATH = MODEL_PATH / "bge-m3-st"
RERANK_MODEL_PATH = MODEL_PATH / "bge-reranker-base"

# Model Tuning
MAX_NEW_TOKENS = 100 # has to match specs of the model (256 max), lower to improve speed
CONTEXT_WINDOW = 8000 # has to match the specs of the model 4096 for phi
MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.9
MODEL_DO_SAMPLE = False 
REPETITION_PENALTY = 1.05 # 1.15 seems to be causing some hallicinations

# Embed Model Tuning
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# Retrieval
SIMILARITY_TOP_K = 25 # Testing with 25 and 40 to see if recall is better
MIXED_TOP_K = 40 # boosting to 40/40/60 makes it take twice as long
SCORE_RATIO = 0.85 # keep nodes that are at least 85% as good as first node

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
    "parallel_process":False, #will allow for processing across independed processes but buggy on my system
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
    "repetition_penalty":REPETITION_PENALTY, #could upgrade to 1.1
    "context_window":CONTEXT_WINDOW,
}

QA_TEMPLATE = PromptTemplate(
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Instructions:\n"
    "- Answer using only the provided context.\n"
    "- Do not use outside knowledge.\n"
    "- If the question asks about a specific job, rate, ship, ship class, year range, shipyard, or place, only report exact or clearly supported matches from the context.\n"
    "- Do not infer that a job, rate, ship, or exposure is present unless it is explicitly stated or clearly supported in the context.\n"
    "- If a document is identified by an internal job number in its title, treat that title as part of the evidence.\n"
    "- If the context does not contain the answer, say exactly: I cannot find this in the documents.\n\n"
    "Answer:"
)

SYSTEM_TEMPLATE = (
    "You are a document analysis assistant.\n"
    "Your task is to answer questions strictly from the provided document context.\n\n"
    "Rules:\n"
    "- Use only the provided context.\n"
    "- Do not use outside knowledge.\n"
    "- Report only what is explicitly stated or clearly supported in the context.\n"
    "- For questions about a specific job, rate, ship, ship class, year range, shipyard, or place, treat the exact requested term as critical.\n"
    "- Do not substitute related occupations or make unsupported connections.\n"
    "- Preserve distinctions between exact matches, partial matches, and unrelated entries when relevant.\n"
    "- If the answer is not in the context, say exactly: I cannot find this in the documents.\n"
)

#MetaData Extraction: 
EXTRACT_TEXT_WINDOW = 5000
EXTRACT_JOB_WINDOW = 2000
MAX_VALID_YEAR = 1999
MIN_VALID_YEAR = 1800

# Do Header, filter by metadata (json file)

#can later filter by metadata so will be important to add as much metadata as possible
# only include in header very important info bc ikts visitble to llm

#strugglnig to answer about the memos, when given job numbers
# need to expand window to more than just first 5k chars for metadata bc shipyards getting cut
#Can we see which nodes after filter are being returned
#keeps saying cant find in docs, assuming this is because of header or metadata added

#preingecting into embedded text - better recall but what if have to redo later, embeddings will need to be redone
#inject at retrieval - safer for future and not having to redo embeddings

# add configs for the response synth and retrievers

RATE_ALIASES = {
    "Fireman": [
        "fireman",
        "fn",
    ],
    "Fireman Apprentice": [
        "fireman apprentice",
        "fa",
    ],
    "Boiler Technician": [
        "boiler technician",
        "bt",
    ],
    "Machinist's Mate": [
        "machinist's mate",
        "machinists mate",
        "mm",
    ],
    "Electrician's Mate": [
        "electrician's mate",
        "electricians mate",
        "em",
    ],
    "Engineman": [
        "engineman",
        "en",
    ],
    "Boilerman": [
        "boilerman",
        "bt",  # only keep if this is actually right for your corpus; otherwise remove
    ],
}