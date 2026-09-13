import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# NIM Configuration
# When the NIM container is running locally, use the local endpoint.
# Change to "https://integrate.api.nvidia.com/v1" to use NVIDIA API Cloud instead.
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")

# Embeddings run on the RTX 3090 for direct local execution. Containerized
# deployments can set this to "cpu" so the NIM service owns the GPU.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai_papers")

# Paths (Dynamic based on root directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(BASE_DIR, "chroma_db"))
STATE_DIR = os.getenv("STATE_DIR", BASE_DIR)

# Separate SQLite databases for separate concerns
SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH", os.path.join(STATE_DIR, "checkpoints.sqlite")
)
AGENT_MEMORY_PATH = os.getenv(
    "AGENT_MEMORY_PATH", os.path.join(STATE_DIR, "agent_memory.sqlite")
)

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
