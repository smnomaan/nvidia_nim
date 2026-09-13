"""Deterministic specialist workers exposed through DeepAgents task routing.

The coordinator decides when to delegate. Each compiled subagent then performs one
bounded operation so an 8B model cannot lose evidence, repeat tools indefinitely,
or send multiple tool calls to a NIM deployment that accepts one at a time.
"""

from deepagents import CompiledSubAgent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.tools.fact_checker import verify_claim_against_papers
from src.tools.retriever import retrieve_from_papers


def _delegated_text(state: dict) -> str:
    return str(state["messages"][-1].content)


def run_rag_retriever(state: dict) -> dict:
    """Return raw cited evidence for the delegated query."""
    evidence = retrieve_from_papers(_delegated_text(state), top_k=5)
    return {"messages": [AIMessage(content=evidence)]}


def run_fact_checker(state: dict) -> dict:
    """Retrieve independent evidence and judge the delegated claim once."""
    verdict = verify_claim_against_papers(_delegated_text(state))
    return {"messages": [AIMessage(content=verdict)]}


rag_retriever_subagent: CompiledSubAgent = {
    "name": "rag-retriever",
    "description": (
        "Returns cited passages from Attention Is All You Need, BERT, and Llama 3. "
        "Use first for every factual or technical paper question."
    ),
    "runnable": RunnableLambda(run_rag_retriever),
}


fact_checker_subagent: CompiledSubAgent = {
    "name": "fact-checker",
    "description": (
        "Independently retrieves paper evidence and verifies one factual claim. "
        "Use exactly once after rag-retriever for technical questions."
    ),
    "runnable": RunnableLambda(run_fact_checker),
}
