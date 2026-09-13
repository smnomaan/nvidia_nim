"""
retriever.py — The ChromaDB RAG Search Tool

This module defines the only tool that the RAG subagent has access to.
The Supervisor/Coordinator cannot call this directly — it MUST delegate
to the rag-retriever subagent, which is the whole point of this design.

The tool:
  1. Takes a natural language query from the subagent.
  2. Encodes it using the BAAI embedding model on the configured device.
  3. Runs a similarity search against our pre-built ChromaDB vector store.
  4. Returns the top-k most relevant text chunks as a formatted string.
"""

# We defer Chroma's heavy import until retrieval is first used.

import os
import re

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
)
from src.embeddings import create_embeddings

# Reuse one vector-store client and embedding model within the process.
_vectorstore = None

PAPER_QUERY_PREFIXES = (
    (("bert",), "BERT paper"),
    (("llama",), "Llama 3 paper"),
    (("transformer", "attention is all you need"), "Attention Is All You Need paper"),
)


def targeted_queries(query: str) -> list[str]:
    """Create one paper-specific query per explicitly named paper."""
    lowered = query.lower()
    expanded = lowered
    for phrase, expansion in (
        ("context length", "maximum supported context window sequence length"),
        ("optimizer and beta values", "optimizer beta hyperparameters training"),
        ("beta values", "optimizer beta1 beta2 hyperparameters training"),
        ("position-encoding approaches", "positional embeddings position encoding"),
        (
            "training objectives",
            "pretraining objective language model textual sequence next token prediction",
        ),
        ("model scale", "model parameters parameter count"),
    ):
        expanded = expanded.replace(phrase, expansion)
    if "optimizer" in expanded and "beta" in expanded:
        expanded = f"{expanded} beta1 beta2 optimizer hyperparameters"
    prefixes = [
        prefix
        for markers, prefix in PAPER_QUERY_PREFIXES
        if any(marker in lowered for marker in markers)
    ]
    if len(prefixes) < 2:
        return [expanded]
    topic = expanded
    for phrase in (
        "attention is all you need",
        "original transformer",
        "transformer architecture",
        "llama 3",
        "bert large",
        "bert",
        "compare",
        "described in",
        "papers",
        "paper",
    ):
        topic = topic.replace(phrase, " ")
    topic = re.sub(r"[^a-z0-9-]+", " ", topic)
    stopwords = {"a", "an", "and", "how", "in", "of", "on", "the", "to", "use"}
    topic = " ".join(
        word for word in topic.split() if word not in stopwords
    ).strip()
    return [f"{prefix} {topic}" for prefix in prefixes]

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma

        _embeddings = create_embeddings()
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
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
    results = []
    seen = set()
    for search_query in targeted_queries(query):
        for document in get_vectorstore().similarity_search(search_query, k=top_k):
            identity = document.metadata.get("chunk_id") or (
                document.metadata.get("source"),
                document.metadata.get("page"),
                document.page_content,
            )
            if identity in seen:
                continue
            seen.add(identity)
            results.append(document)

    if not results:
        return "No relevant documents found for this query."

    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_name") or os.path.basename(
            doc.metadata.get("source", "Unknown Source")
        )
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        formatted.append(
            f"--- Result {i} (Source: {source}, PDF Page: {page}) ---\n{doc.page_content}"
        )

    return "\n\n".join(formatted)
