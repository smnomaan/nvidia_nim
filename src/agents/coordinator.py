"""
coordinator.py — The Supervisor Agent.

THE COORDINATOR PATTERN
-----------------------
The coordinator is the "brain" of the multi-agent system. It:
  - Receives the user's message.
  - PLANS what needs to be done.
  - DECIDES whether to delegate to the rag-retriever subagent.
  - SYNTHESIZES the subagent's findings into a final answer.
  - REMEMBERS conversation history via the SQLite checkpointer.

The coordinator has ZERO custom tools. It cannot search ChromaDB directly.
It MUST use the rag-retriever subagent for that. This makes the delegation
pattern explicit and demonstrable in LangSmith traces.

WHY NIM?
--------
We pass a pre-instantiated ChatNVIDIA object as the model. DeepAgents accepts
any BaseChatModel instance, so we bypass the "provider:model" string format
(which is only for known cloud providers) and plug our local NIM container
directly in.
"""

import sqlite3

from deepagents import create_deep_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import NVIDIA_API_KEY, AGENT_MEMORY_PATH, NIM_BASE_URL, NIM_MODEL
from src.agents.subagents import rag_retriever_subagent, fact_checker_subagent


COORDINATOR_SYSTEM_PROMPT = (
    "You are an expert AI research assistant with deep knowledge of foundational "
    "AI papers. You have access to a specialized knowledge base containing three "
    "seminal papers: 'Attention Is All You Need' (Transformers), 'BERT', and "
    "'Llama 3'.\n\n"
    "YOUR TEAM:\n"
    "  - 'rag-retriever': Searches the knowledge base for relevant passages. "
    "You MUST delegate all fact-finding to this subagent.\n"
    "  - 'fact-checker': Verifies claims against evidence using LLM-as-a-Judge. "
    "Use this AFTER retrieval to validate key findings.\n\n"
    "YOUR WORKFLOW:\n"
    "  1. PLAN: Before doing anything, briefly state your plan. Example: "
    "'I will search for information about X, then verify the key claims.'\n"
    "  2. RETRIEVE: Delegate to 'rag-retriever' with a precise search query.\n"
    "  3. VERIFY: Send the key claim(s) and retrieved evidence to 'fact-checker' "
    "for validation.\n"
    "  4. SYNTHESIZE: Combine verified findings with your reasoning to produce "
    "a clear, well-cited answer. Include the fact-checker's verdict.\n"
    "  5. RESPOND: Deliver a concise, accurate answer with source citations "
    "and verification status.\n\n"
    "IMPORTANT RULES:\n"
    "  - You CANNOT search the knowledge base directly — always use rag-retriever.\n"
    "  - You CANNOT judge facts yourself — always use fact-checker for verification.\n"
    "  - Always cite the source paper and page number in your final answer.\n"
    "  - If the question is conversational (e.g. 'hello', 'thanks'), answer directly "
    "without calling any subagent.\n"
    "  - Be concise. Aim for answers under 300 words unless depth is requested."
)


def create_coordinator(session_id: str = "default-session"):
    """Create and return the NIM-powered multi-agent RAG coordinator.

    Args:
        session_id: Unique identifier for the conversation session. Used as
                    the LangGraph thread_id so chat history is preserved per
                    user session. Defaults to 'default-session'.

    Returns:
        A tuple of (compiled_agent, config) ready to be invoked.
    """
    # Use the local NIM container as the LLM backend.
    # Falls back to NVIDIA API Cloud if NIM_BASE_URL is not a local endpoint.
    kwargs = {
        "model": NIM_MODEL,
        "api_key": NVIDIA_API_KEY,
        "temperature": 0.2,
        "max_completion_tokens": 1024,
    }
    
    # Only pass base_url if we are NOT using the official NVIDIA cloud.
    # Passing the cloud URL manually can sometimes break LangChain's auth routing.
    if NIM_BASE_URL and "integrate.api.nvidia.com" not in NIM_BASE_URL:
        kwargs["base_url"] = NIM_BASE_URL

    nim_llm = ChatNVIDIA(**kwargs)

    # Separate SQLite DB for chat memory (never mixed with ingestion checkpoints)
    conn = sqlite3.connect(AGENT_MEMORY_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)

    agent = create_deep_agent(
        model=nim_llm,
        tools=[],                          # Coordinator has ZERO custom tools
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        subagents=[rag_retriever_subagent, fact_checker_subagent],
        checkpointer=memory,
    )

    config = {"configurable": {"thread_id": session_id}}

    return agent, config
