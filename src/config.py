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

# Paths (Dynamic based on root directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Separate SQLite databases for separate concerns
SQLITE_DB_PATH = os.path.join(BASE_DIR, "checkpoints.sqlite")    # Ingestion state (ephemeral)
AGENT_MEMORY_PATH = os.path.join(BASE_DIR, "agent_memory.sqlite") # Chat history (persistent)

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
