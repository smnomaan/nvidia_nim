"""
fact_checker.py — The LLM-as-a-Judge Verification Tool

This module provides the fact-checking tool used by the fact-checker subagent.
It takes a claim and the supporting evidence, then uses the SAME NIM-powered
LLM to evaluate whether the evidence actually supports the claim.

This is the "LLM-as-a-Judge" pattern — using one LLM call to critically
evaluate the output of another LLM call. The key insight is that the judge
uses a completely different system prompt (adversarial, skeptical) than the
agent that generated the original answer (helpful, informative).

The fact-checker subagent has exclusive access to this tool. The coordinator
cannot call it directly.
"""

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.config import NVIDIA_API_KEY, NIM_BASE_URL, NIM_MODEL


def _get_judge_llm() -> ChatNVIDIA:
    """Create a separate ChatNVIDIA instance for the judge.

    We use a low temperature (0.1) to make the judge's evaluations
    as deterministic and consistent as possible.
    """
    kwargs = {
        "model": NIM_MODEL,
        "api_key": NVIDIA_API_KEY,
        "temperature": 0.1,
        "max_completion_tokens": 512,
    }
    if NIM_BASE_URL and "integrate.api.nvidia.com" not in NIM_BASE_URL:
        kwargs["base_url"] = NIM_BASE_URL
    return ChatNVIDIA(**kwargs)


def verify_claims(claim: str, evidence: str) -> str:
    """Evaluate whether a claim is supported by the given evidence using LLM-as-a-Judge.

    This tool acts as a critical fact-checker. It takes a specific claim and the
    evidence that was retrieved from the knowledge base, then returns a structured
    verdict on whether the evidence actually supports the claim.

    Args:
        claim: The specific factual claim to verify (e.g. "Transformers use
               multi-head attention with 8 heads").
        evidence: The retrieved passages or context that supposedly support
                  the claim.

    Returns:
        A structured verdict with SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED
        status, along with reasoning.
    """
    judge_prompt = (
        "You are a rigorous academic fact-checker. Your job is to evaluate whether "
        "a given CLAIM is supported by the provided EVIDENCE.\n\n"
        "Be strict and skeptical. Do not assume anything that is not explicitly "
        "stated in the evidence.\n\n"
        f"CLAIM: {claim}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        "Respond in this exact format:\n"
        "VERDICT: [SUPPORTED | PARTIALLY SUPPORTED | NOT SUPPORTED]\n"
        "CONFIDENCE: [HIGH | MEDIUM | LOW]\n"
        "REASONING: [1-3 sentences explaining your judgment]\n"
        "MISSING: [What additional evidence would strengthen or refute this claim, "
        "if any]"
    )

    judge = _get_judge_llm()
    response = judge.invoke(judge_prompt)
    return response.content
