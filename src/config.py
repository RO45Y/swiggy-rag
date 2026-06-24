import os
from dotenv import load_dotenv

load_dotenv()

# Paths
PDF_PATH = "data/swiggy_annual_report.pdf"
VECTOR_STORE_PATH = "vector_store"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
CHUNKING_METHOD = "semantic"  # "recursive" or "semantic"
SEMANTIC_THRESHOLD = 0.65      # similarity threshold for sentence merging

# Embedding
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"   # better than small

# Retrieval
TOP_K = 18                     # retrieve more, then rerank
FINAL_K = 5                    # number of chunks to pass to LLM

# Reranker
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.0

# Multi‑query
NUM_QUERIES = 3                # number of alternative questions to generate