"""
agent.py — Entry point for the Multi-Agent RAG Chatbot.

Run this script directly to test the agent in the terminal before
the Streamlit UI is ready:

    uv run python -m src.agent

The agent will prompt you for input in a loop. Type 'exit' to quit.
Each session preserves conversation history via the SQLite checkpointer.

LIVE TRACE MODE
---------------
Instead of silently waiting for a final answer, this script streams
the LangGraph execution and prints each delegation hop in real time,
making the multi-agent orchestration visible in the terminal.
"""

import uuid
from src.agents.coordinator import create_coordinator

# ANSI color codes for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _get_display_name(node_name: str) -> tuple[str, str]:
    """Map internal LangGraph node names to clean display names + colors.

    Returns (display_name, ansi_color).
    """
    name_lower = node_name.lower()

    if "patchtoollcallsmiddleware" in name_lower or "before_agent" in name_lower:
        return "Coordinator", CYAN
    if name_lower == "model":
        return "Coordinator", CYAN
    if name_lower == "tools":
        return "Tools", YELLOW
    if "rag" in name_lower or "retriever" in name_lower:
        return "RAG Retriever", GREEN
    if "fact" in name_lower or "checker" in name_lower or "judge" in name_lower:
        return "Fact Checker", MAGENTA

    return node_name, YELLOW


# Track which subagent is currently active for better labeling
_active_subagent = None


def chat(session_id: str = None):
    """Run an interactive chat loop in the terminal with live delegation traces."""
    global _active_subagent

    if session_id is None:
        session_id = f"terminal-{uuid.uuid4().hex[:8]}"

    print(f"\n{BOLD}{'=' * 60}")
    print(f"  NIM Multi-Agent RAG Chatbot")
    print(f"  Powered by: DeepAgents + NVIDIA NIM + ChromaDB")
    print(f"  Knowledge Base: Transformers, BERT, Llama 3 papers")
    print(f"{'=' * 60}{RESET}")
    print(f"{DIM}NOTE: NIM container must be running on localhost:8000")
    print(f"Type 'exit' to quit.{RESET}\n")

    agent, config = create_coordinator(session_id=session_id)

    # Track message IDs across questions so replayed history is never shown.
    # LangGraph's stream() replays ALL messages from the checkpointer on every
    # call. Without this, old delegations would re-appear on new questions.
    seen_message_ids = set()

    while True:
        user_input = input(f"{BOLD}You:{RESET} ").strip()
        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print(f"\n{DIM}{'─' * 60}")
        print(f"  LIVE AGENT TRACE")
        print(f"{'─' * 60}{RESET}")

        _active_subagent = None
        seen_events = set()
        final_answer = None

        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_data in event.items():
                if not isinstance(node_data, dict):
                    continue

                display_name, color = _get_display_name(node_name)
                messages = node_data.get("messages", [])

                if hasattr(messages, "value"):
                    messages = messages.value
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    # Skip messages we've already processed (replayed history)
                    msg_id = getattr(msg, "id", None)
                    if msg_id:
                        if msg_id in seen_message_ids:
                            continue
                        seen_message_ids.add(msg_id)

                    if not hasattr(msg, "content") and not hasattr(msg, "tool_calls"):
                        continue

                    # === Tool Calls (Delegation) ===
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_name = tool_call.get("name", "").lower()
                            args = tool_call.get("args", {})

                            if tool_name in ("task", "transfer", "delegate"):
                                desc = str(args.get("description", args))[:120]
                                all_text = str(args).lower()

                                if any(k in all_text for k in ["fact", "check", "judge", "verif", "claim"]):
                                    _active_subagent = "Fact Checker"
                                    agent_color = MAGENTA
                                elif any(k in all_text for k in ["rag", "retriev", "search", "knowledge", "paper", "document"]):
                                    _active_subagent = "RAG Retriever"
                                    agent_color = GREEN
                                else:
                                    _active_subagent = "Subagent"
                                    agent_color = YELLOW

                                event_key = f"delegate-{_active_subagent}-{desc[:60]}"
                                if event_key not in seen_events:
                                    seen_events.add(event_key)
                                    print(f"  {CYAN}[Coordinator]{RESET} -> Delegating to {BOLD}{agent_color}{_active_subagent}{RESET}")
                                    if desc:
                                        print(f"    {DIM}\"{desc}\"{RESET}")

                            elif tool_name == "retrieve_from_papers":
                                query = args.get("query", "")[:80]
                                print(f"  {GREEN}[RAG Retriever]{RESET} -> Searching: {DIM}\"{query}\"{RESET}")
                            elif tool_name in ("verify_claims", "fact_check"):
                                claim = args.get("claim", "")[:80]
                                print(f"  {MAGENTA}[Fact Checker]{RESET} -> Judging: {DIM}\"{claim}\"{RESET}")

                    # === Tool Responses ===
                    elif hasattr(msg, "type") and msg.type == "tool":
                        content = str(msg.content)
                        if "VERDICT:" in content.upper():
                            print(f"  {MAGENTA}[Fact Checker]{RESET} << {BOLD}{content.strip()}{RESET}")
                        else:
                            label = _active_subagent or "Tools"
                            lcolor = GREEN if "rag" in label.lower() else MAGENTA
                            print(f"  {lcolor}[{label}]{RESET} << Received evidence")

                    # === Final Answer ===
                    elif hasattr(msg, "content") and msg.content and msg.content.strip():
                        if not getattr(msg, "tool_calls", None):
                            final_answer = msg.content

        print(f"{DIM}{'─' * 60}{RESET}")

        if final_answer:
            print(f"\n{BOLD}{CYAN}Agent:{RESET} {final_answer}\n")
        else:
            print(f"\n{YELLOW}Agent: (No response generated){RESET}\n")


if __name__ == "__main__":
    chat()
