"""Resumable and retry-safe ingestion for the local paper collection."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import List, Protocol, TypedDict

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_DIR,
    DATA_DIR,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    SQLITE_DB_PATH,
)
from src.embeddings import create_embeddings


ARXIV_PAPERS = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert": "https://arxiv.org/pdf/1810.04805.pdf",
    "llama3": "https://arxiv.org/pdf/2407.21783.pdf",
}

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200
DOWNLOAD_TIMEOUT = (10, 120)
INGESTION_THREAD_ID = "paper_ingestion_v2"


class VectorStore(Protocol):
    """Small interface used by the graph and its failure-recovery tests."""

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> list[str]: ...


class IngestionState(TypedDict):
    documents_to_process: List[str]
    total_chunks: int
    chunks: List[str]
    chunk_ids: List[str]
    metadatas: List[dict]
    current_chunk_index: int
    ingestion_fingerprint: str
    status: str


def get_vectorstore():
    """Load the embedding model and persistent Chroma collection on demand."""
    from langchain_chroma import Chroma
    print(f"Initializing {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}...")
    embeddings = create_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )


def download_papers(
    papers: dict[str, str] | None = None,
    data_dir: str = DATA_DIR,
) -> list[str]:
    """Download missing PDFs atomically and reject unsuccessful responses."""
    paper_map = papers or ARXIV_PAPERS
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for name, url in paper_map.items():
        target = target_dir / f"{name}.pdf"
        if not target.exists():
            partial = target.with_suffix(".pdf.part")
            print(f"Downloading {name} from ArXiv...")
            try:
                with requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as response:
                    response.raise_for_status()
                    with partial.open("wb") as destination:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                destination.write(block)
                os.replace(partial, target)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        paths.append(str(target.resolve()))

    return paths


def _chunk_id(source: str, page: int | str, chunk_index: int, text: str) -> str:
    """Return a stable ID so replaying a write updates rather than duplicates it."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = f"{Path(source).name}|{page}|{chunk_index}|{content_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _ingestion_fingerprint(document_paths: list[str]) -> str:
    """Identify the corpus and chunking configuration used for a checkpoint."""
    digest = hashlib.sha256()
    digest.update(f"chunk_size={CHUNK_SIZE};overlap={CHUNK_OVERLAP}".encode())
    for path_string in sorted(document_paths, key=lambda value: Path(value).name):
        path = Path(path_string)
        digest.update(path.name.encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_and_chunk(state: IngestionState) -> dict:
    """Load the corpus once and create stable IDs for every resulting chunk."""
    print("Loading and chunking documents...")
    all_chunks: list[str] = []
    all_chunk_ids: list[str] = []
    all_metadatas: list[dict] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for document_path in state["documents_to_process"]:
        pages = PyPDFLoader(document_path).load()
        chunks = splitter.split_documents(pages)
        for chunk_index, chunk in enumerate(chunks):
            page = chunk.metadata.get("page", "unknown")
            identifier = _chunk_id(
                document_path,
                page,
                chunk_index,
                chunk.page_content,
            )
            metadata = {
                **chunk.metadata,
                "source": str(Path(document_path).resolve()),
                "source_name": Path(document_path).name,
                "chunk_index": chunk_index,
                "chunk_id": identifier,
            }
            all_chunks.append(chunk.page_content)
            all_chunk_ids.append(identifier)
            all_metadatas.append(metadata)

    return {
        "chunks": all_chunks,
        "chunk_ids": all_chunk_ids,
        "metadatas": all_metadatas,
        "total_chunks": len(all_chunks),
        "current_chunk_index": 0,
        "ingestion_fingerprint": _ingestion_fingerprint(
            state["documents_to_process"]
        ),
        "status": "processing",
    }


def create_ingestion_graph(vectorstore: VectorStore, checkpointer: SqliteSaver):
    """Build the graph with injected storage so recovery can be tested."""

    def embed_and_store(state: IngestionState) -> dict:
        index = state["current_chunk_index"]
        total = state["total_chunks"]
        if index >= total:
            return {}

        print(f"Embedding chunk {index + 1} of {total}...")
        vectorstore.add_texts(
            texts=[state["chunks"][index]],
            metadatas=[state["metadatas"][index]],
            ids=[state["chunk_ids"][index]],
        )
        return {"current_chunk_index": index + 1}

    def route_after_write(state: IngestionState):
        if state["current_chunk_index"] < state["total_chunks"]:
            return "embed_and_store"
        return "finalize"

    def finalize(_: IngestionState) -> dict:
        return {"status": "completed"}

    builder = StateGraph(IngestionState)
    builder.add_node("load_and_chunk", load_and_chunk)
    builder.add_node("embed_and_store", embed_and_store)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "load_and_chunk")
    builder.add_edge("load_and_chunk", "embed_and_store")
    builder.add_conditional_edges("embed_and_store", route_after_write)
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def run_ingestion() -> None:
    """Start or resume the configured ingestion thread."""
    paper_paths = download_papers()
    vectorstore = get_vectorstore()
    connection = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    graph = create_ingestion_graph(vectorstore, SqliteSaver(connection))
    config = {"configurable": {"thread_id": INGESTION_THREAD_ID}}
    state = graph.get_state(config)

    try:
        if not state.values:
            print("Starting new stateful ingestion pipeline...")
            initial_state: IngestionState = {
                "documents_to_process": paper_paths,
                "chunks": [],
                "chunk_ids": [],
                "metadatas": [],
                "total_chunks": 0,
                "current_chunk_index": 0,
                "ingestion_fingerprint": "",
                "status": "started",
            }
            result = graph.invoke(initial_state, config=config)
        elif state.values.get("status") != "completed":
            current = state.values.get("current_chunk_index", 0)
            total = state.values.get("total_chunks", 0)
            print(f"Resuming ingestion from chunk {current + 1} of {total}...")
            result = graph.invoke(None, config=config)
        else:
            print("Ingestion is already complete.")
            return

        if result.get("status") != "completed":
            raise RuntimeError("Ingestion stopped without a completed checkpoint")
        print(f"Ingestion complete. Vector store saved to {CHROMA_DB_DIR}")
    finally:
        connection.close()


if __name__ == "__main__":
    run_ingestion()
