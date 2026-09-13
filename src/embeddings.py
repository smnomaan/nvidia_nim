"""Shared construction for the local sentence-transformer embeddings."""

from __future__ import annotations

import importlib
import os

from src.config import EMBEDDING_DEVICE, EMBEDDING_MODEL


def create_embeddings():
    """Create the configured embedding model.

    On Windows, preloading the compiled scientific stack avoids an import-order
    crash observed when sentence-transformers is first initialized from the
    larger Streamlit/DeepAgents process. Linux containers do not need it.
    """
    if os.name == "nt":
        for module_name in ("pandas", "sklearn", "sentence_transformers"):
            importlib.import_module(module_name)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
