import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://chatembedding.openai.azure.com/"
)

AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "2LRHLT3x4AAr5krCsayMfAg84DgmVKw623GE9bWbOol1Ardwu9wYJQQJ99BEACYeBjFXJ3w3AAABACOGMaBj"
)

AZURE_OPENAI_REGION = os.getenv("AZURE_OPENAI_REGION", "eastus")

# Model names (deployment names in your Azure OpenAI resource)
# Based on your Azure Portal: use "embed_chat" for embeddings and "gpt-4o-2" for chat
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed_chat")  # Your embedding deployment name
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-2")  # Your chat/vision deployment name

# API version - FIXED
AZURE_OPENAI_API_VERSION = "2023-05-06"  

# Embedding configuration
EMBEDDING_DIMENSION = 1536  # text-embedding-ada-002 dimension

# Text chunking configuration
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# RAG configuration
TOP_K_RESULTS = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.1  # Minimum similarity score (temporarily lowered for debugging)

# File upload configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
ALLOWED_EXTENSIONS = {'.pdf'}

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
PROCESSED_IMAGES_DIR = os.path.join(BASE_DIR, "data", "processed_images")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
