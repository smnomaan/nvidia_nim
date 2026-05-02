# NVIDIA NIM — Multi-Agent RAG Chat

A multi-agent RAG system built with NVIDIA NIM, LangGraph, DeepAgents, and ChromaDB. Includes a Streamlit UI that shows live agent delegation traces as the system works.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

The system answers research questions about three foundational AI papers:

- Attention Is All You Need (Transformers)
- BERT: Pre-training of Deep Bidirectional Transformers
- The Llama 3 Herd of Models

A Coordinator agent receives the user query, delegates retrieval to a RAG Retriever subagent, optionally delegates fact verification to a Fact Checker subagent, then synthesizes a final answer. The Coordinator has no direct access to ChromaDB — it must go through the subagents.

---

## Architecture

```
User Query
    |
    v
+-----------------------------+
|      Coordinator Agent       |  <- Llama 3.1 8B via NVIDIA NIM
|  (no custom tools, plans     |
|   and synthesizes answers)   |
+------------+----------------+
             |  delegates via DeepAgents task tool
    +--------+---------+
    |                  |
    v                  v
+--------------+  +------------------+
| RAG Retriever|  |  Fact Checker    |
|  (ChromaDB   |  |  (LLM-as-Judge,  |
|   search)    |  |  verify_claims)  |
+--------------+  +------------------+
```

---

## Features

- Multi-agent orchestration via DeepAgents + LangGraph
- Vector search with ChromaDB and `BAAI/bge-base-en-v1.5` embeddings (GPU)
- LLM-as-a-Judge fact checking (SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED)
- Real-time trace panel in the UI showing live delegation steps
- Per-session chat memory via SQLite checkpointer
- Resumable ingestion pipeline — recovers from crashes at the exact chunk it stopped at
- Docker Compose setup for running NIM + app together in containers

---

## Project Structure

```
nvidia_nim/
├── app.py                  # Streamlit UI
├── src/
│   ├── agent.py            # Terminal chat loop with ANSI traces
│   ├── config.py           # Env vars and path constants
│   ├── ingest.py           # PDF ingestion pipeline (LangGraph)
│   ├── agents/
│   │   ├── coordinator.py  # Supervisor agent
│   │   └── subagents.py    # RAG Retriever and Fact Checker
│   └── tools/
│       ├── retriever.py    # ChromaDB similarity search
│       └── fact_checker.py # Claim verification tool
├── data/                   # Downloaded ArXiv PDFs
├── chroma_db/              # ChromaDB vector store
├── agent_memory.sqlite     # Chat history
├── checkpoints.sqlite      # Ingestion state
├── Dockerfile              # Image for the Streamlit app
├── docker-compose.yml      # NIM container + app container
└── pyproject.toml          # Dependencies (uv)
```

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU (for local embeddings via PyTorch CUDA)
- Either a local NIM container (NGC API key + Docker) or an NVIDIA Cloud API key

---

## Setup — NVIDIA Cloud (no local NIM container)

### 1. Clone and install

```bash
git clone https://github.com/your-username/nvidia_nim.git
cd nvidia_nim
uv sync
```

### 2. Create `.env`

```env
# Get from https://build.nvidia.com
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx

NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=meta/llama-3.1-8b-instruct

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=nvidia-nim-rag
```

### 3. Ingest the papers

Downloads the PDFs and builds the ChromaDB vector store. Run once.

```bash
uv run python -m src.ingest
```

### 4. Start the UI

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Setup — Local NIM Container (Docker)

Runs Llama 3.1 8B and the Streamlit app in containers. Requires Docker with NVIDIA Container Toolkit.

```bash
docker compose up
```

- NIM API: `http://localhost:8000/v1`
- Streamlit: `http://localhost:8501`

The compose file pulls `nvcr.io/nim/meta/llama-3.1-8b-instruct:latest`, caches model weights in a Docker volume, and starts the app after NIM is ready.

---

## Terminal Mode

Test the agent without the UI:

```bash
uv run python -m src.agent
```

Prints ANSI-colored delegation traces in real time.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | Required. NGC or NVIDIA Cloud key |
| `NIM_BASE_URL` | `http://localhost:8000/v1` | LLM endpoint |
| `NIM_MODEL` | `meta/llama-3.1-8b-instruct` | Model name |
| `LANGCHAIN_API_KEY` | — | Optional. Enables LangSmith tracing |

---

## Ingestion Pipeline

`src/ingest.py` is a LangGraph state machine with a SQLite checkpointer.

1. Downloads PDFs from ArXiv if not already in `data/`
2. Splits each PDF into 1000-character chunks with 200-character overlap
3. Embeds and stores one chunk at a time into ChromaDB, saving state after each

If the process is interrupted, re-running it resumes from where it stopped.

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | NVIDIA NIM via `langchain-nvidia-ai-endpoints` |
| Multi-Agent | DeepAgents |
| Orchestration | LangGraph |
| Embeddings | `sentence-transformers` (CUDA) |
| Vector Store | ChromaDB |
| Chat Memory | LangGraph SQLite Checkpointer |
| UI | Streamlit |
| Package Manager | uv |

---

## License

MIT
