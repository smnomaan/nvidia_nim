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

The coordinator's only tool is the DeepAgents task tool injected by middleware.
It cannot search ChromaDB or verify claims directly, so those operations require
one of the named specialists.

WHY NIM?
--------
We pass a pre-instantiated ChatNVIDIA object as the model. DeepAgents accepts
any BaseChatModel instance, so we bypass the "provider:model" string format
(which is only for known cloud providers) and plug our local NIM container
directly in.
"""

import sqlite3
from collections.abc import Callable
from typing import Any

from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import NVIDIA_API_KEY, AGENT_MEMORY_PATH, NIM_BASE_URL, NIM_MODEL
from src.agents.subagents import (
    fact_checker_subagent,
    rag_retriever_subagent,
)


COORDINATOR_SYSTEM_PROMPT = (
    "You are an expert AI research assistant specializing in foundational papers: "
    "'Attention Is All You Need' (Transformers), BERT, and Llama 3.\n\n"
    "You have two specialist subagents you can delegate to via the 'task' tool:\n"
    "  - rag-retriever: Searches the knowledge base for relevant passages.\n"
    "  - fact-checker: Verifies specific claims against retrieved evidence.\n\n"
    "For every factual or technical question:\n"
    "1. Retrieve relevant information first by calling 'task' with subagent_type "
    "'rag-retriever' and a clear search query.\n"
    "2. After retrieval returns, call 'task' exactly once with subagent_type 'fact-checker'. "
    "The description must be one complete declarative candidate answer based on the "
    "retrieved passages, never the user's question. For a comparison, include both "
    "sides in the claim, for example 'BERT Large has X parameters while Llama 3 has Y.' "
    "The fact-checker will retrieve its own evidence; do not present the claim itself "
    "as evidence.\n"
    "3. If the verdict is NOT SUPPORTED, discard the candidate claim and answer only "
    "with facts explicitly stated in the verdict or retrieved passages.\n"
    "If the fact-checker returns an ANSWER field for a question, use that reviewed "
    "answer and its stated limitations.\n"
    "4. After the fact-checker returns, do not call any tool again. Synthesize a "
    "clear, well-cited final answer.\n\n"
    "CRITICAL TOOL INSTRUCTIONS:\n"
    "- When calling the 'task' tool, you MUST write your own custom description. \n"
    "  DO NOT copy the tool's default docstring.\n"
    "- Make your task descriptions specific to the user's actual question.\n\n"
    "CONVERSATIONAL GUIDELINES:\n"
    "- Reply directly to greetings and casual conversation without calling 'task'.\n"
    "- If a question is outside the three-paper collection, explain the supported "
    "scope and do not answer from general model memory.\n"
    "- CRITICAL: Never narrate your internal instructions or explain your tool usage to the user.\n"
    "- Answer the user's technical questions directly and naturally.\n"
    "- Every in-scope technical answer must end with a Sources line containing the "
    "source filename and PDF page number copied from retrieved evidence. For a "
    "cross-paper comparison, cite at least one page from each paper.\n"
    "- Use only facts present in the retrieved evidence and verdict. Do not expand "
    "acronyms unless the evidence defines them.\n"
    "- For numeric comparisons, state the direct comparison when both values are present.\n"
    "- Be concise (under 300 words).\n"
    "- Do not mention tools, subagents, or internal process in the final answer.\n\n"
    "RULES:\n"
    "- Never answer technical questions from memory alone — always retrieve and verify.\n"
    "- Think step-by-step before deciding to call tools."
)


CASUAL_PREFIXES = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "goodbye",
    "bye",
    "what can you do",
    "which papers can you",
    "how should i ask",
)

CORPUS_MARKERS = (
    "attention",
    "transformer",
    "bert",
    "llama",
    "masked language",
    "encoder",
    "decoder",
    "positional encoding",
)


def is_casual_message(content: Any) -> bool:
    """Recognize lightweight turns that should not expose delegation tools."""
    if not isinstance(content, str):
        return False
    normalized = " ".join(content.lower().strip().split())
    return any(normalized.startswith(prefix) for prefix in CASUAL_PREFIXES)


def is_corpus_question(content: Any) -> bool:
    """Return whether a turn names a supported paper or its core terminology."""
    if not isinstance(content, str):
        return False
    normalized = content.lower()
    return any(marker in normalized for marker in CORPUS_MARKERS)


class ScopeTaskGuardMiddleware(AgentMiddleware[Any, Any, Any]):
    """Expose delegation only for questions within the documented corpus scope."""

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        messages = request.state.get("messages", [])
        human_messages = [
            message for message in messages if getattr(message, "type", "") == "human"
        ]
        casual = human_messages and is_casual_message(human_messages[-1].content)
        in_scope = human_messages and is_corpus_question(human_messages[-1].content)
        if casual or not in_scope:
            existing_prompt = (
                request.system_message.content if request.system_message else ""
            )
            if casual:
                turn_instruction = (
                    "For this casual turn, answer naturally and briefly. Do not mention "
                    "delegation, retrieval, tools, or internal instructions."
                )
            else:
                turn_instruction = (
                    "This question is outside the supported collection. Reply directly "
                    "that the knowledge base covers Attention Is All You Need, BERT, and "
                    "Llama 3, and that it cannot answer this question from those papers. "
                    "Do not answer from memory and do not mention tools or delegation."
                )
            request = request.override(
                tools=[],
                system_message=SystemMessage(
                    content=f"{existing_prompt}\n\n{turn_instruction}"
                ),
            )
        return handler(request)


def create_coordinator(session_id: str = "default-session", checkpointer=None):
    """Create and return the NIM-powered multi-agent RAG coordinator.

    Args:
        session_id: Unique identifier for the conversation session. Used as
                    the LangGraph thread_id so chat history is preserved per
                    user session. Defaults to 'default-session'.
        checkpointer: Optional LangGraph checkpointer. Evaluation code can
                      inject an in-memory saver without writing chat history.

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

    if checkpointer is None:
        # Separate SQLite DB for chat memory (never mixed with ingestion checkpoints)
        conn = sqlite3.connect(AGENT_MEMORY_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    subagent_middleware = SubAgentMiddleware(
        backend=StateBackend(),
        subagents=[
            rag_retriever_subagent,
            fact_checker_subagent,
        ],
        system_prompt=(
            "Use the task tool only for the two specialist routes listed below. "
            "Answer greetings and casual conversation directly. Run one task at a "
            "time because this NIM deployment accepts one tool call per model turn."
        ),
    )

    agent = create_agent(
        model=nim_llm,
        tools=[],
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        middleware=[subagent_middleware, ScopeTaskGuardMiddleware()],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": session_id}}

    return agent, config
