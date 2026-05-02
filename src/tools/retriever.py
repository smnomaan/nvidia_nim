"""
retriever.py — The ChromaDB RAG Search Tool

This module defines the only tool that the RAG subagent has access to.
The Supervisor/Coordinator cannot call this directly — it MUST delegate
to the rag-retriever subagent, which is the whole point of this design.

The tool:
  1. Takes a natural language query from the subagent.
  2. Encodes it using the BAAI embedding model on the GPU.
  3. Runs a similarity search against our pre-built ChromaDB vector store.
  4. Returns the top-k most relevant text chunks as a formatted string.
"""

# We defer the imports to prevent PyTorch from initializing in the main thread.

from src.config import CHROMA_DB_DIR

# We use lazy loading for the vectorstore to prevent PyTorch CUDA threading Segfaults.
# Streamlit executes UI interactions in separate background threads, and PyTorch
# will crash if a CUDA model is initialized in the main thread but inferenced elsewhere.
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cuda"},
        )
        _vectorstore = Chroma(
            collection_name="ai_papers",
            embedding_function=_embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
    return _vectorstore


def retrieve_from_papers(query: str, top_k: int = 5) -> str:
    """Search the ArXiv AI papers knowledge base for context relevant to the query.
    
    Use this tool when you need to find information about the 'Attention Is All You
    Need' transformer paper, the BERT paper, or the Llama 3 paper. Returns the most
    relevant text passages with their source metadata.

    Args:
        query: A natural language question or search query.
        top_k: Number of top results to return (default: 5).

    Returns:
        A formatted string containing the relevant document passages and metadata.
    """
    results = get_vectorstore().similarity_search(query, k=top_k)

    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"--- Result {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}"
        )

    return "\n\n".join(formatted)
