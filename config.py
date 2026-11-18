import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- API Keys ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

    # --- Vector DB ---
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "rag_docs"

    # --- Cohere Embedding Model ---
    # Options: embed-english-v3.0, embed-multilingual-v3.0, embed-english-light-v3.0
    EMBEDDING_MODEL = "embed-english-v3.0"
    EMBEDDING_INPUT_TYPE = "search_document"  # for ingest, use "search_query" for query

    # --- Groq LLM Model ---
    # Options: openai/gpt-oss-120b, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it
    LLM_MODEL = "openai/gpt-oss-120b"
    TEMPERATURE = 0.0
    MAX_TOKENS = 1024

    # --- RAG Settings ---
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K = 4

    # --- Cohere Reranker (Optional) ---
    USE_RERANKER = False
    RERANK_MODEL = "rerank-english-v3.0"
    RERANK_TOP_N = 3