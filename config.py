"""
Configuration file containing all reusable constants for the HybridRAG project
"""

# API Configuration
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BIT-User-Agent/1.0 (academic research;)"

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOKENIZER_MODEL = "bert-base-uncased"
LLM_RAG_MODEL_NAME = "google/flan-t5-base"

# Re-ranking Configuration
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


# Text Chunking Configuration
# Text Chunking Configuration
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
MAX_NEW_TOKENS_LONG = 512

# Data Processing Configuration
MAX_RECORDS_LIMIT = 10
BATCH_SIZE = 32

# Vector Database Configuration
PINECONE_API_KEY = "pcsk_2Esr7y_49UUndeEPqmXFyWG3CpRQXnd3Y2TAnKtia1GSDgKoEdqqS8P14AEs3aHL6uzFgo"
PINECONE_INDEX_NAME = "hybrid-rag"
PINECONE_HOST = "https://hybrid-rag-md4312k.svc.aped-4627-b74a.pinecone.io" # Optional, can be derived or left out if not using host directly usually


# Paths
DATA_FILES_PATH = "./files"
DYNAMIC_URLS_FILE = "dynamic_urls.json"
FIXED_URLS_FILE = "fixed_urls.json"

# BM25 Configuration
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25
RRF_WEIGHTS = {
    'dense': 1.0,
    'sparse': 3.0
}

# Retrieval Configuration
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.5

# API Request Configuration
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

CATEGORIES = [
        "Category:Computer security",
        "Category:Cryptography",
        "Category:Information privacy",
        "Category:Information sensitivity",
        "Category:Cybersecurity engineering"
    ]
TARGET_DYNAMIC_URLS = 300
MIN_WORDS = 200

BM25_INDEX_PATH = "./bm25_index/bm25_index.pkl"

# Evaluation Configuration
FORCE_REGENERATE_QA = True