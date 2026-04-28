from pathlib import Path
import torch
import psutil
from llama_index.core.prompts import PromptTemplate


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "vector_db"
STORAGE_DIR = DB_DIR / "storage"
DOCS_DIR = Path(r"C:\Users\Christian.DESKTOP-2DI7LJ6\Documents\Local_Code\MAA-RAG\Documents\Docs")  #was docs_root
INDEX_ID = "main_index"
COLLECTION_NAME = "test_store"
#STORED_FILES_PATH = BASE_DIR / "setup_index" / "stored_files.json"
STORED_FILES_PATH = DB_DIR / "stored_files.json"
DEFAULT_MAX_UPSERT_FILES = 20
DOCS_STORE = STORAGE_DIR / "docstore.json"
WPD_PATH = BASE_DIR / "wpd_support" / "wpd2text" / "wpd2text.exe"
METADATA_FACETS_PATH = DB_DIR / "metadata_facets.json"
METADATA_SOURCE_CONTRIBUTIONS_PATH = DB_DIR / "metadata_source_contributions.json"

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
RETRIEVE_ONLY_DEFAULT_TOP_K = 60
RETRIEVE_ONLY_MIN_TOP_K = 5
RETRIEVE_ONLY_MAX_TOP_K = 200
RETRIEVE_ONLY_DEFAULT_RATIO = SCORE_RATIO

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
    "Rules:\n"
    "- Use only the context.\n"
    "- Report only explicit or clearly supported facts.\n"
    "- If not found in context, say exactly: I cannot find this in the documents.\n\n"
    "Answer:"
)

SUMMARY_TEMPLATE = PromptTemplate(
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Rules:\n"
    "- Summarize only from context.\n"
    "- Focus on document purpose, key findings, and important details.\n"
    "- If insufficient context, say exactly: I cannot determine this from the provided context.\n\n"
    "Answer:"
)

TIMELINE_TEMPLATE = PromptTemplate(
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Rules:\n"
    "- Build chronology using only explicit dates/years/ranges from context.\n"
    "- Do not invent dates.\n"
    "- Put uncertain ordering under Undated or Unclear Events.\n\n"
    "Format:\n"
    "Timeline:\n"
    "- Date/Year:\n"
    "  Event:\n"

    "Answer:"
)

SYSTEM_TEMPLATE = (
    "You are a document analysis assistant.\n"
    "Use only provided context.\n"
    "Do not use outside knowledge.\n"
    "Report explicit or clearly supported facts only.\n"
)

#MetaData Extraction: 
EXTRACT_TEXT_WINDOW = 5000
EXTRACT_JOB_WINDOW = 2000
MAX_VALID_YEAR = 1999
MIN_VALID_YEAR = 1800

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
        "bt",
    ],
    "Seaman": [
        "seaman",
        "sn",
    ],
    "Airman": [
        "airman",
        "an",
    ],
    "Radarman": [
        "radarman",
        "rd",
    ],
    "Boatswain's Mate": [
        "boatswain's mate",
        "boatswains mate",
        "bm",
    ],
    "Quartermaster": [
        "quartermaster",
        "qm",
    ],
    "Sonarman": [
        "sonarman",
        "so",
    ],
    "Sonar Technician": [
        "sonar technician",
        "st",
        "sts",
        "stg",
    ],
    "Signalman": [
        "signalman",
        "sm",
    ],
    "Torpedoman's Mate": [
        "torpedoman's mate",
        "torpedomans mate",
        "tm",
    ],
    "Gunner's Mate": [
        "gunner's mate",
        "gunners mate",
        "gm",
    ],
    "Fire Control Technician": [
        "fire control technician",
        "ft",
        "ftm",
        "ftg",
    ],
    "Mineman": [
        "mineman",
        "mn",
    ],
    "Missile Technician": [
        "missile technician",
        "mt",
    ],
    "Electronics Technician": [
        "electronics technician",
        "et",
        "etr",
        "etn",
    ],
    "Data Systems Technician": [
        "data systems technician",
        "ds",
    ],
    "Instrumentman": [
        "instrumentman",
        "im",
    ],
    "Opticalman": [
        "opticalman",
        "om",
    ],
    "Postal Clerk": [
        "postal clerk",
        "pc",
    ],
    "Communications Technician": [
        "communications technician",
        "ct",
    ],
    "Communications Yeoman": [
        "communications yeoman",
        "cyn",
    ],
    "Yeoman": [
        "yeoman",
        "yn",
    ],
    "Personnelman": [
        "personnelman",
        "pn",
    ],
    "Machine Accountant": [
        "machine accountant",
        "ma",
    ],
    "Storekeeper": [
        "storekeeper",
        "sk",
    ],
    "Disbursing Clerk": [
        "disbursing clerk",
        "dk",
    ],
    "Commissaryman": [
        "commissaryman",
        "cs",
    ],
    "Ship's Serviceman": [
        "ship's serviceman",
        "ships serviceman",
        "sh",
        "shs",
        "shb",
        "sht",
        "shl",
        "shr",
    ],
    "Journalist": [
        "journalist",
        "jo",
    ],
    "Lithographer": [
        "lithographer",
        "li",
    ],
    "Illustrator Draftsman": [
        "illustrator draftsman",
        "dm",
    ],
    "Musician": [
        "musician",
        "mu",
    ],
    "Machinery Repairman": [
        "machinery repairman",
        "mr",
    ],
    "Interior Communications Electrician": [
        "interior communications electrician",
        "ic",
    ],
    "Shipfitter": [
        "shipfitter",
        "sf",
        "sfm",
        "sfp",
    ],
    "Damage Controlman": [
        "damage controlman",
        "dc",
    ],
    "Patternmaker": [
        "patternmaker",
        "pm",
    ],
    "Molder": [
        "molder",
        "ml",
    ],
    "Boilermaker": [
        "boilermaker",
        "br",
    ],
    "Engineering Aid": [
        "engineering aid",
        "ea",
        "eas",
        "ead",
    ],
    "Construction Electrician": [
        "construction electrician",
        "ce",
        "cew",
        "cep",
        "cet",
        "ces",
    ],
    "Equipment Operator": [
        "equipment operator",
        "eo",
        "eon",
        "eoh",
    ],
    "Construction Mechanic": [
        "construction mechanic",
        "cm",
        "cma",
        "cmh",
    ],
    "Builder": [
        "builder",
        "bu",
        "bul",
        "buh",
        "bur",
    ],
    "Steelworker": [
        "steelworker",
        "sw",
        "swe",
        "swf",
    ],
    "Utilitiesman": [
        "utilitiesman",
        "ut",
        "utp",
        "uta",
        "utb",
        "utw",
    ],
    "Aviation Maintenance Administrationman": [
        "aviation maintenance administrationman",
        "az",
    ],
    "Aviation Machinist's Mate": [
        "aviation machinist's mate",
        "aviation machinists mate",
        "ad",
        "adj",
        "adr",
    ],
    "Aviation Antisubmarine Warfare Technician": [
        "aviation antisubmarine warfare technician",
        "ax",
    ],
    "Aviation Electronics Technician": [
        "aviation electronics technician",
        "at",
        "atr",
        "atn",
    ],
    "Photographic Intelligenceman": [
        "photographic intelligenceman",
        "pt",
    ],
    "Aviation Ordnanceman": [
        "aviation ordnanceman",
        "ao",
    ],
    "Air Controlman": [
        "air controlman",
        "ac",
    ],
    "Aviation Boatswain's Mate": [
        "aviation boatswain's mate",
        "aviation boatswains mate",
        "ab",
        "abh",
        "abf",
        "abe",
    ],
    "Aviation Electrician's Mate": [
        "aviation electrician's mate",
        "aviation electricians mate",
        "ae",
    ],
    "Aviation Structural Mechanic": [
        "aviation structural mechanic",
        "am",
        "ams",
        "amh",
        "ame",
    ],
    "Parachute Rigger": [
        "parachute rigger",
        "pr",
    ],
    "Aerographer's Mate": [
        "aerographer's mate",
        "aerographers mate",
        "ag",
    ],
    "Tradewoman": [
        "tradewoman",
        "td",
    ],
    "Aviation Support Equipment Technician": [
        "aviation support equipment technician",
        "as",
        "ase",
        "ash",
        "asm",
    ],
    "Aviation Fire Control Technician": [
        "aviation fire control technician",
        "aq",
        "aqf",
        "aqb",
    ],
    "Aviation Storekeeper": [
        "aviation storekeeper",
        "ak",
    ],
    "Photographer's Mate": [
        "photographer's mate",
        "photographers mate",
        "ph",
    ],
    "Hospital Corpsman": [
        "hospital corpsman",
        "hm",
    ],
    "Dental Technician": [
        "dental technician",
        "dt",
    ],
    "Steward": [
        "steward",
        "sd",
    ],
}