import os
import sqlite3
import requests
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import DATA_DIR, CHROMA_DB_DIR, SQLITE_DB_PATH

# URLs of the foundational AI papers
ARXIV_PAPERS = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762.pdf",
    "bert": "https://arxiv.org/pdf/1810.04805.pdf",
    "llama3": "https://arxiv.org/pdf/2407.21783.pdf" 
}

print("Initializing Embedding Model on GPU (this only happens once)...")


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cuda'}
)

print("Connecting to ChromaDB...")
vectorstore = Chroma(
    collection_name="ai_papers",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

class IngestionState(TypedDict):
    documents_to_process: List[str]
    total_chunks: int
    chunks: List[str]
    metadatas: List[dict]
    current_chunk_index: int
    status: str

def download_papers():
    paths = []
    for name, url in ARXIV_PAPERS.items():
        filepath = os.path.join(DATA_DIR, f"{name}.pdf")
        if not os.path.exists(filepath):
            print(f"Downloading {name} from ArXiv...")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
        paths.append(filepath)
    return paths

def load_and_chunk(state: IngestionState):
    print("Loading and chunking documents...")
    all_chunks = []
    all_metadatas = []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for doc_path in state["documents_to_process"]:
        loader = PyPDFLoader(doc_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            all_chunks.append(chunk.page_content)
            all_metadatas.append(chunk.metadata)
            
    return {
        "chunks": all_chunks,
        "metadatas": all_metadatas,
        "total_chunks": len(all_chunks),
        "current_chunk_index": 0,
        "status": "processing"
    }

def embed_and_store(state: IngestionState):
    index = state["current_chunk_index"]
    total = state["total_chunks"]
    
    if index >= total:
        return {"status": "completed"}
        
    print(f"Embedding chunk {index + 1} of {total}...")
    chunk_text = state["chunks"][index]
    metadata = state["metadatas"][index]
    
    # Add single chunk to Chroma
    vectorstore.add_texts(texts=[chunk_text], metadatas=[metadata])
    
    return {
        "current_chunk_index": index + 1
    }

def routing_condition(state: IngestionState):
    if state["current_chunk_index"] < state["total_chunks"]:
        return "embed_and_store"
    return END

# Build the Graph
builder = StateGraph(IngestionState)
builder.add_node("load_and_chunk", load_and_chunk)
builder.add_node("embed_and_store", embed_and_store)

builder.add_edge(START, "load_and_chunk")
builder.add_edge("load_and_chunk", "embed_and_store")
builder.add_conditional_edges("embed_and_store", routing_condition)

conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    paper_paths = download_papers()
    config = {"configurable": {"thread_id": "paper_ingestion_1"}}
    state = graph.get_state(config)
    
    if not state.values:
        print("Starting new Stateful Ingestion Pipeline...")
        initial_state = {
            "documents_to_process": paper_paths,
            "chunks": [],
            "metadatas": [],
            "total_chunks": 0,
            "current_chunk_index": 0,
            "status": "started"
        }
        graph.invoke(initial_state, config=config)
        print(f"Ingestion Complete! Vector store saved to {CHROMA_DB_DIR}")
    else:
        current_idx = state.values.get('current_chunk_index', 0)
        total = state.values.get('total_chunks', 0)
        if current_idx < total:
            print(f"Resuming ingestion from chunk {current_idx + 1} of {total}...")
            graph.invoke(None, config=config)
            print(f"Ingestion Complete! Vector store saved to {CHROMA_DB_DIR}")
        else:
            print("Ingestion is already complete!")
