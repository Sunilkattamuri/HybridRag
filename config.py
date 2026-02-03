"""
Configuration file containing all reusable constants for the HybridRAG project
"""

# API Configuration
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BIT-User-Agent/1.0 (academic research;)"

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "bert-base-uncased"
LLM_RAG_MODEL_NAME = "google/flan-t5-base"


# Text Chunking Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
MAX_NEW_TOKENS_LONG = 512

# Data Processing Configuration
MAX_RECORDS_LIMIT = 10
BATCH_SIZE = 32

# Vector Database Configuration
CHROMA_DB_PATH = "./chroma_data"
COLLECTION_NAME = "hybrid_rag_collection"

# Paths
DATA_FILES_PATH = "./files"
DYNAMIC_URLS_FILE = "dynamic_urls.json"
FIXED_URLS_FILE = "fixed_urls.json"

# BM25 Configuration
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

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