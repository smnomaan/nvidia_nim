"""
subagents.py — The specialized SubAgent definitions for the RAG system.

DESIGN PHILOSOPHY
-----------------
The Supervisor (coordinator) has ZERO custom tools. It cannot search the
knowledge base directly. This is deliberate — it forces the coordinator to
delegate to the rag-retriever subagent, making the multi-agent pattern clear
and demonstrable.

TWO-AGENT TEAM
--------------
1. rag-retriever: Searches ChromaDB for relevant passages from the papers.
2. fact-checker:  Uses LLM-as-a-Judge to verify claims against evidence.

The coordinator delegates retrieval first, then optionally sends the findings
to the fact-checker for verification before synthesizing the final answer.
"""

from deepagents import SubAgent
from src.tools.retriever import retrieve_from_papers
from src.tools.fact_checker import verify_claims


rag_retriever_subagent: SubAgent = {
    "name": "rag-retriever",
    "description": (
        "Searches the ArXiv AI papers knowledge base (Attention Is All You Need, "
        "BERT, and Llama 3) for relevant passages. Use this subagent whenever the "
        "user asks a question that may be answered by the research papers."
    ),
    "system_prompt": (
        "You are a specialist research retriever. Your only job is to find relevant "
        "passages from the knowledge base to answer a given question.\n\n"
        "Instructions:\n"
        "1. Use the retrieve_from_papers tool to search the knowledge base.\n"
        "2. Make 1-2 targeted searches using different keywords if needed.\n"
        "3. Synthesize the retrieved passages into a concise set of bullet points.\n"
        "4. Always include the source paper name and page number for each finding.\n"
        "5. Keep your response under 400 words — the coordinator will handle synthesis.\n\n"
        "Output format:\n"
        "## Retrieved Context\n"
        "- Finding 1 (Source: <paper>, Page: <N>)\n"
        "- Finding 2 (Source: <paper>, Page: <N>)\n"
        "## Sources\n"
        "- <paper name>"
    ),
    "tools": [retrieve_from_papers],
}


fact_checker_subagent: SubAgent = {
    "name": "fact-checker",
    "description": (
        "Verifies factual claims against evidence using LLM-as-a-Judge. "
        "Use this subagent AFTER the rag-retriever has returned findings. "
        "Send it a specific claim and the evidence to get a structured verdict "
        "(SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED)."
    ),
    "system_prompt": (
        "You are a rigorous academic fact-checker operating as an LLM-as-a-Judge.\n\n"
        "Your job:\n"
        "1. Receive a claim and supporting evidence from the coordinator.\n"
        "2. Use the verify_claims tool to evaluate whether the evidence supports the claim.\n"
        "3. Return the structured verdict exactly as the tool provides it.\n\n"
        "Rules:\n"
        "- Be skeptical. Do NOT assume facts that are not in the evidence.\n"
        "- If the coordinator sends multiple claims, verify each one separately.\n"
        "- Always use the verify_claims tool — never judge claims from memory.\n"
        "- Keep your response structured and concise."
    ),
    "tools": [verify_claims],
}
