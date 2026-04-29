import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
NGC_API_KEY = os.getenv("NGC_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Paths (Dynamic based on root directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "checkpoints.sqlite")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
