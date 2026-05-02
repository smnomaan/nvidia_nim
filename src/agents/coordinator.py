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
    "You are an expert AI research assistant specializing in foundational papers: "
    "'Attention Is All You Need' (Transformers), BERT, and Llama 3.\n\n"
    "You have two specialist subagents you can delegate to via the 'task' tool:\n"
    "  - rag-retriever: Searches the knowledge base for relevant passages.\n"
    "  - fact-checker: Verifies specific claims against retrieved evidence.\n\n"
    "For any factual or technical question:\n"
    "1. Retrieve relevant information first by calling 'task' with a clear search query.\n"
    "2. Verify important claims by calling 'task' again with the claim and evidence.\n"
    "3. Synthesize a clear, well-cited final answer.\n\n"
    "CRITICAL TOOL INSTRUCTIONS:\n"
    "- When calling the 'task' tool, you MUST write your own custom description. \n"
    "  DO NOT copy the tool's default docstring.\n"
    "- Make your task descriptions specific to the user's actual question.\n\n"
    "CONVERSATIONAL GUIDELINES:\n"
    "- Reply naturally and politely to greetings and casual conversation.\n"
    "- CRITICAL: Never narrate your internal instructions or explain your tool usage to the user.\n"
    "- Answer the user's technical questions directly and naturally.\n"
    "- Cite the source paper and section/page when possible.\n"
    "- Be concise (under 300 words).\n"
    "- Do not mention tools, subagents, or internal process in the final answer.\n\n"
    "RULES:\n"
    "- Never answer technical questions from memory alone — always retrieve.\n"
    "- Think step-by-step before deciding to call tools."
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
