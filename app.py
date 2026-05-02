import os
import sqlite3
import uuid

import streamlit as st

from src.agents.coordinator import create_coordinator
from src.config import AGENT_MEMORY_PATH, DATA_DIR, CHROMA_DB_DIR


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NIM RAG Chat",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# CUSTOM CSS - NVIDIA NIM ENTERPRISE THEME
# ==========================================
st.html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* ==========================================
   GLOBAL BASE
========================================== */

:root {
    --nvidia-green: #76B900;
    --nvidia-green-soft: rgba(118, 185, 0, 0.18);
    --bg-main: #0E1117;
    --bg-panel: rgba(22, 26, 33, 0.82);
    --bg-panel-light: rgba(38, 39, 48, 0.72);
    --border-soft: rgba(255, 255, 255, 0.08);
    --text-main: #FAFAFA;
    --text-muted: #A0A0A8;
    --text-subtle: #6B7280;
    --purple-soft: #C084FC;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(118, 185, 0, 0.08), transparent 34%),
        linear-gradient(135deg, #0E1117 0%, #090B10 100%) !important;
    color: var(--text-main) !important;
    font-family: 'Outfit', Inter, Roboto, sans-serif !important;
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1500px !important;
}

/* ==========================================
   TYPOGRAPHY
========================================== */

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    color: var(--text-main) !important;
    margin-bottom: 0.25rem !important;
}

h2, h3 {
    color: var(--text-main) !important;
    letter-spacing: -0.02em !important;
}

p {
    color: inherit;
}

.nim-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-top: -0.2rem;
    margin-bottom: 1.8rem;
}

.nim-green {
    color: var(--nvidia-green);
    font-weight: 700;
}

/* ==========================================
   SIDEBAR
========================================== */

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(22, 26, 33, 0.98), rgba(12, 14, 19, 0.98)) !important;
    border-right: 1px solid var(--border-soft);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    margin-top: 1rem !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-soft) !important;
    margin: 1.2rem 0 !important;
}

/* ==========================================
   BUTTONS
========================================== */

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid var(--border-soft) !important;
    background: rgba(255, 255, 255, 0.035) !important;
    color: var(--text-main) !important;
    font-weight: 600 !important;
    transition: all 0.18s ease-in-out !important;
}

.stButton > button:hover {
    border-color: rgba(118, 185, 0, 0.5) !important;
    background: rgba(118, 185, 0, 0.09) !important;
    color: var(--text-main) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #76B900, #5A9500) !important;
    color: #071000 !important;
    border: 1px solid rgba(118, 185, 0, 0.7) !important;
    box-shadow: 0 0 0 1px rgba(118, 185, 0, 0.18),
                0 10px 28px rgba(118, 185, 0, 0.18) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8AD600, #76B900) !important;
    color: #071000 !important;
}

/* ==========================================
   METRIC CARDS
========================================== */

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(38, 39, 48, 0.78), rgba(18, 21, 28, 0.88)) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 18px !important;
    padding: 1.15rem 1.25rem !important;
    box-shadow:
        0 12px 32px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.86rem !important;
}

div[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricDelta"] {
    color: var(--nvidia-green) !important;
}

/* ==========================================
   PANELS
========================================== */

.glass-panel {
    background: linear-gradient(180deg, rgba(38, 39, 48, 0.74), rgba(18, 21, 28, 0.88)) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 18px !important;
    padding: 1rem !important;
    box-shadow:
        0 18px 45px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
}

.panel-title {
    color: var(--nvidia-green);
    font-weight: 750;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
    margin-bottom: 0.5rem;
}

.panel-muted {
    color: var(--text-muted);
    font-size: 0.88rem;
}

/* Glow Animation for Active States */
@keyframes neon-pulse {
    0% { box-shadow: 0 0 5px rgba(118, 185, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.035); }
    50% { box-shadow: 0 0 25px rgba(118, 185, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.035); }
    100% { box-shadow: 0 0 5px rgba(118, 185, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.035); }
}

.trace-active {
    animation: neon-pulse 2.5s infinite ease-in-out !important;
    border-color: rgba(118, 185, 0, 0.5) !important;
}

/* ==========================================
   CHAT
========================================== */

[data-testid="stChatMessage"] {
    background: transparent !important;
}

[data-testid="stChatMessage"] > div {
    background: linear-gradient(180deg, rgba(38, 39, 48, 0.70), rgba(20, 23, 30, 0.88)) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px !important;
    padding: 0.9rem 1rem !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.20) !important;
}

[data-testid="stChatMessage"] p {
    line-height: 1.65 !important;
}

/* Chat input */
div[data-testid="stChatInput"] {
    border: 1px solid rgba(118, 185, 0, 0.38) !important;
    border-radius: 16px !important;
    background: rgba(14, 17, 23, 0.95) !important;
    box-shadow:
        0 0 0 1px rgba(118, 185, 0, 0.08),
        0 16px 42px rgba(0, 0, 0, 0.35) !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(118, 185, 0, 0.72) !important;
    box-shadow:
        0 0 0 1px rgba(118, 185, 0, 0.22),
        0 0 28px rgba(118, 185, 0, 0.10) !important;
}

textarea {
    color: var(--text-main) !important;
}

/* ==========================================
   TRACE PANEL
========================================== */

.trace-panel {
    min-height: 180px;
    max-height: 550px;
    overflow-y: auto;
}

.trace-step {
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.075);
}

.trace-step:last-child {
    border-bottom: none;
}

.trace-agent {
    color: var(--nvidia-green);
    font-weight: 750;
}

.trace-fact {
    color: var(--purple-soft);
    font-weight: 750;
}

.trace-desc {
    color: var(--text-muted);
    font-size: 0.86rem;
    line-height: 1.45;
}

.trace-status {
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* ==========================================
   STATUS BADGES
========================================== */

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    background: rgba(118, 185, 0, 0.12);
    color: var(--nvidia-green);
    border: 1px solid rgba(118, 185, 0, 0.24);
    font-size: 0.78rem;
    font-weight: 700;
}

.status-dot {
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 999px;
    background: var(--nvidia-green);
    box-shadow: 0 0 12px rgba(118, 185, 0, 0.8);
}

/* ==========================================
   SCROLLBARS
========================================== */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.03);
}

::-webkit-scrollbar-thumb {
    background: rgba(118, 185, 0, 0.38);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(118, 185, 0, 0.58);
}
/* ==========================================
   CUSTOM CHAT INPUT (replaces st.chat_input)
========================================== */

/* Nuke the default Streamlit form border completely */
div[data-testid="stForm"],
div[data-testid="stForm"] > div,
div[data-testid="stForm"] > div > div {
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* Style the actual input field */
div[data-testid="stForm"] div[data-testid="stTextInput"] input {
    background: rgba(14, 17, 23, 0.96) !important;
    border: 1.5px solid var(--nvidia-green) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
    font-family: 'Outfit', Inter, Roboto, sans-serif !important;
    font-size: 1rem !important;
    padding: 0.72rem 1.1rem !important;
    box-shadow: 0 0 14px rgba(118, 185, 0, 0.2) !important;
    transition: box-shadow 0.2s ease !important;
}

div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
    box-shadow: 0 0 22px rgba(118, 185, 0, 0.42) !important;
    outline: none !important;
}

div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder {
    color: var(--text-subtle) !important;
}

/* Remove the input wrapper border that Streamlit injects */
div[data-testid="stForm"] div[data-testid="stTextInput"] > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Submit button */
div[data-testid="stForm"] button[kind="formSubmit"] {
    background: rgba(118, 185, 0, 0.12) !important;
    border: 1.5px solid var(--nvidia-green) !important;
    border-radius: 10px !important;
    color: var(--nvidia-green) !important;
    font-size: 1.1rem !important;
    padding: 0.55rem !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stForm"] button[kind="formSubmit"]:hover {
    background: rgba(118, 185, 0, 0.26) !important;
    box-shadow: 0 0 12px rgba(118, 185, 0, 0.3) !important;
}

/* Shrink sidebar metric values */
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
}

/* ==========================================
   THINKING ANIMATION
========================================== */

@keyframes thinkingDot {
    0%, 80%, 100% { opacity: 0.15; transform: translateY(0); }
    40%            { opacity: 1;    transform: translateY(-4px); }
}

.thinking-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--text-muted);
    font-size: 0.95rem;
    font-style: italic;
    padding: 0.2rem 0;
}

.thinking-dots span {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--nvidia-green);
    animation: thinkingDot 1.4s ease-in-out infinite;
    opacity: 0.15;
}

.thinking-dots span:nth-child(2) { animation-delay: 0.18s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.36s; }

/* ==========================================
   CHAT MESSAGE AVATARS - NVIDIA STYLE
========================================== */

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #76B900, #5A9500) !important;
    border: 1px solid rgba(118, 185, 0, 0.85) !important;
    border-radius: 50% !important;
    box-shadow:
        0 0 0 1px rgba(118, 185, 0, 0.25),
        0 0 14px rgba(118, 185, 0, 0.45) !important;
}

/* Make emoji/icon sit nicely in the green circle */
[data-testid="stChatMessageAvatarUser"] div,
[data-testid="stChatMessageAvatarAssistant"] div {
    background: transparent !important;
    color: #071000 !important;
    font-weight: 800 !important;
}
</style>
"""
)


# ==========================================
# DATABASE HELPERS
# ==========================================
def get_historical_sessions() -> list[str]:
    """Query the LangGraph SQLite DB for distinct thread IDs."""
    if not os.path.exists(AGENT_MEMORY_PATH):
        return []

    try:
        with sqlite3.connect(AGENT_MEMORY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT thread_id
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY MAX(checkpoint_id) DESC
                """
            )
            return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []


# ==========================================
# AGENT HELPERS
# ==========================================
def load_agent():
    """
    Create the coordinator graph fresh on each Streamlit run.

    This keeps the existing behavior from your app and helps avoid
    SQLite/PyTorch threading issues during reruns.
    """
    agent, _ = create_coordinator(st.session_state.session_id)
    return agent


def hydrate_messages_from_checkpoint(agent, config: dict) -> list[dict]:
    """Restore human and final AI messages from LangGraph checkpoint state."""
    hydrated_messages = []

    try:
        state = agent.get_state(config)

        if state and hasattr(state, "values") and "messages" in state.values:
            for message in state.values["messages"]:
                if message.type == "human":
                    hydrated_messages.append(
                        {"role": "user", "content": message.content}
                    )
                elif (
                    message.type == "ai"
                    and message.content
                    and not getattr(message, "tool_calls", None)
                ):
                    hydrated_messages.append(
                        {"role": "assistant", "content": message.content}
                    )
    except Exception:
        pass

    return hydrated_messages


def get_trace_header(is_active=False):
    """Return the HTML for the trace panel header."""
    classes = "glass-panel trace-panel trace-active" if is_active else "glass-panel trace-panel"
    return f"""<div class="{classes}">
<div class="panel-title">Live Agent Traces</div>"""

def render_trace_waiting(trace_container):
    """Render the default trace panel state."""
    html = get_trace_header(is_active=False) + """
<span class="trace-status">Awaiting query...</span>
</div>"""
    trace_container.markdown(html, unsafe_allow_html=True)


def append_trace_step(trace_html: str, content: str) -> str:
    """Append one styled trace step to the trace HTML buffer."""
    return trace_html + f"""<div class="trace-step">\n{content}\n</div>"""


# ==========================================
# STATE INITIALIZATION
# ==========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"chat-{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = None

if "trace_html" not in st.session_state:
    st.session_state.trace_html = None


sessions = get_historical_sessions()

if st.session_state.session_id not in sessions:
    sessions.insert(0, st.session_state.session_id)


agent = load_agent()
config = {"configurable": {"thread_id": st.session_state.session_id}}

if st.session_state.messages is None:
    st.session_state.messages = hydrate_messages_from_checkpoint(agent, config)


# ==========================================
# LEFT SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## NVIDIA NIM")
    st.markdown(
        """<span class="status-badge">
<span class="status-dot"></span>
Multi-Agent RAG
</span>""",
        unsafe_allow_html=True,
    )

    if st.button("Start New Chat", use_container_width=True, type="primary"):
        st.session_state.session_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = None
        st.rerun()

    st.markdown("---")
    st.markdown("### Chat History")

    with st.container(height=400):
        for session_id in sessions:
            button_type = (
                "primary"
                if session_id == st.session_state.session_id
                else "secondary"
            )

            if st.button(
                session_id[:16],
                key=f"btn_{session_id}",
                use_container_width=True,
                type=button_type,
            ):
                if st.session_state.session_id != session_id:
                    st.session_state.session_id = session_id
                    st.session_state.messages = None
                    st.rerun()




# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 style="margin-bottom: 0.15rem;"><span class="nim-green">NVIDIA NIM</span> RAG Chat</h1>
        <div class="nim-subtitle">
            Multi-agent RAG research prototype powered by NVIDIA NIM &amp; Llama 3.1
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# TOP METRICS (Dynamic)
# ==========================================

# Count PDFs in the data directory
_pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]) if os.path.exists(DATA_DIR) else 0

# Count actual chunks in ChromaDB
_chunk_count = "—"
try:
    import chromadb
    _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    _collection = _client.get_collection("ai_papers")
    _chunk_count = f"{_collection.count():,}"
except Exception:
    pass

# Track average latency from session state
if "latencies" not in st.session_state:
    st.session_state.latencies = []

_avg_latency = "—"
if st.session_state.latencies:
    _avg = sum(st.session_state.latencies) / len(st.session_state.latencies)
    _avg_latency = f"{_avg:.1f}s"

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

metric_col_1.metric("Total Chats", len(sessions))
metric_col_2.metric("Knowledge Base", f"{_pdf_count} Papers")
metric_col_3.metric("Vector Chunks", _chunk_count)
metric_col_4.metric("Avg Latency", _avg_latency)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# MAIN LAYOUT
# ==========================================
chat_col, trace_col = st.columns([2.5, 1], gap="large")


# ==========================================
# RIGHT TRACE PANEL
# ==========================================
with trace_col:
    trace_container = st.empty()
    # Render persisted trace if available, otherwise just the header (empty)
    if st.session_state.trace_html:
        trace_container.markdown(st.session_state.trace_html, unsafe_allow_html=True)
    else:
        trace_container.markdown(
            get_trace_header(is_active=False) + "</div>",
            unsafe_allow_html=True,
        )


with chat_col:
    # ---- INPUT BOX pinned at the TOP of the chat column ----
    with st.form(key="chat_form", clear_on_submit=True):
        form_col1, form_col2 = st.columns([10, 1])
        with form_col1:
            prompt_input = st.text_input(
                "chat_input",
                placeholder="Ask about Transformers, BERT, or Llama 3...",
                label_visibility="collapsed",
            )
        with form_col2:
            submitted = st.form_submit_button("↑", use_container_width=True)

    prompt = prompt_input.strip() if submitted and prompt_input.strip() else None

    # ---- MESSAGE HISTORY: newest pair first, correct order within pair ----
    messages = st.session_state.messages
    pairs = []
    i = 0
    while i < len(messages):
        if i + 1 < len(messages):
            pairs.append((messages[i], messages[i + 1]))
            i += 2
        else:
            pairs.append((messages[i], None))
            i += 1

    # Container for the live streamed response (will appear at top, newest)
    new_message_container = st.container()

    # Render historical pairs newest-first
    for user_msg, assistant_msg in reversed(pairs):
        with st.chat_message(user_msg["role"], avatar="👨"):
            st.markdown(user_msg["content"])
        if assistant_msg:
            with st.chat_message(assistant_msg["role"], avatar="🤖"):
                st.markdown(assistant_msg["content"])
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with new_message_container:
        with st.chat_message("user", avatar="👨"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            response_container.markdown(
                """
<div class='thinking-dots'>
  Thinking
  <span></span><span></span><span></span>
</div>
""",
                unsafe_allow_html=True,
            )

    # Reset trace for the new question
    st.session_state.trace_html = get_trace_header(is_active=True)
    trace_html = st.session_state.trace_html
    trace_container.markdown(trace_html + "</div>", unsafe_allow_html=True)

    import time as _time
    _start_time = _time.time()

    final_response = ""
    seen_message_ids = set()

    try:
        for event in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_data in event.items():
                if not isinstance(node_data, dict):
                    continue

                messages = node_data.get("messages", [])

                if hasattr(messages, "value"):
                    messages = messages.value

                if not isinstance(messages, list):
                    messages = [messages]

                for message in messages:
                    message_id = getattr(message, "id", None)

                    if message_id:
                        if message_id in seen_message_ids:
                            continue
                        seen_message_ids.add(message_id)

                    # ------------------------------------------
                    # Tool Call / Delegation Trace
                    # ------------------------------------------
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            name = tool_call.get("name", "").lower()
                            args = tool_call.get("args", {})

                            desc = str(args.get("description", args.get("query", args.get("claim", str(args)))))[:90]
                            full_args_str = str(args).lower()

                            if name in ("rag-retriever", "fact-checker", "task", "delegate"):
                                # Search the entire args dict as a string for fact-checking intent
                                is_fact_checker = (
                                    "fact" in name or "check" in name or "verif" in name or
                                    "fact" in full_args_str or "check" in full_args_str or
                                    "verif" in full_args_str or "claim" in full_args_str or
                                    "judge" in full_args_str
                                )
                                subagent = "Fact Checker" if is_fact_checker else "RAG Retriever"
                                color_class = "trace-fact" if is_fact_checker else "trace-agent"
                                
                                trace_html = append_trace_step(
                                    trace_html,
                                    f"""<span class="trace-agent">Coordinator</span>
<span class="trace-desc">→ Delegating to</span>
<span class="{color_class}">{subagent}</span>
<br>
<span class="trace-desc">"{desc}..."</span>""",
                                )
                            elif name == "retrieve_from_papers":
                                # This is the RAG Retriever actually searching ChromaDB
                                query = args.get("query", str(args))[:90]
                                trace_html = append_trace_step(
                                    trace_html,
                                    f"""<span class="trace-agent">RAG Retriever</span>
<span class="trace-desc">→ Searching Vector DB for:</span>
<br>
<span class="trace-desc">"{query}..."</span>""",
                                )
                            elif name in ("verify_claims", "fact_check"):
                                # This is the Fact Checker actually running LLM-as-a-judge
                                claim = args.get("claim", str(args))[:90]
                                trace_html = append_trace_step(
                                    trace_html,
                                    f"""<span class="trace-fact">Fact Checker</span>
<span class="trace-desc">→ Judging claim:</span>
<br>
<span class="trace-desc">"{claim}..."</span>""",
                                )
                            else:
                                # Fallback
                                trace_html = append_trace_step(
                                    trace_html,
                                    f"""<span class="trace-agent">System</span>
<span class="trace-desc">→ Calling tool {name}</span>""",
                                )

                            st.session_state.trace_html = trace_html + "</div>"
                            trace_container.markdown(
                                st.session_state.trace_html,
                                unsafe_allow_html=True,
                            )

                    # ------------------------------------------
                    # Tool Result Trace
                    # ------------------------------------------
                    elif hasattr(message, "type") and message.type == "tool":
                        content = str(message.content)

                        if "VERDICT:" in content.upper():
                            try:
                                verdict = [
                                    line
                                    for line in content.split("\n")
                                    if "VERDICT:" in line.upper()
                                ][0]
                            except Exception:
                                verdict = content[:80]

                            trace_html = append_trace_step(
                                trace_html,
                                f"""<span class="trace-fact">Fact Checker</span>
<span class="trace-desc">→ {verdict}</span>""",
                            )
                        else:
                            chars = len(content)
                            trace_html = append_trace_step(
                                trace_html,
                                f"""<span class="trace-agent">RAG Retriever</span>
<span class="trace-desc">→ Retrieved {chars} characters of evidence</span>""",
                            )

                        st.session_state.trace_html = trace_html + "</div>"
                        trace_container.markdown(
                            st.session_state.trace_html,
                            unsafe_allow_html=True,
                        )

                    # ------------------------------------------
                    # Final Answer (collect only — render after stream ends)
                    # ------------------------------------------
                    elif (
                        hasattr(message, "content")
                        and message.content
                        and message.content.strip()
                        and not getattr(message, "tool_calls", None)
                        and getattr(message, "type", "") == "ai"
                    ):
                        # Always overwrite — we want the LAST AI message
                        final_response = message.content

        # ---- Stream finished: now render the final answer ----
        if final_response:
            import time
            def stream_words():
                for word in final_response.split(" "):
                    yield word + " "
                    time.sleep(0.04)

            with new_message_container:
                response_container.write_stream(stream_words)

        # Record latency
        _elapsed = _time.time() - _start_time
        st.session_state.latencies.append(_elapsed)

        if final_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )

            trace_html = append_trace_step(
                trace_html,
                """<span class="status-badge">
<span class="status-dot"></span>
Done
</span>""",
            )

            st.session_state.trace_html = trace_html + "</div>"
            trace_container.markdown(
                st.session_state.trace_html,
                unsafe_allow_html=True,
            )
        else:
            with new_message_container:
                response_container.error("No final response generated.")

            trace_html = append_trace_step(
                trace_html,
                "<span class='trace-status'>Error: no final response generated.</span>",
            )

            st.session_state.trace_html = trace_html + "</div>"
            trace_container.markdown(
                st.session_state.trace_html,
                unsafe_allow_html=True,
            )

    except Exception as exc:
        with new_message_container:
            response_container.error(
                "Something went wrong while running the agent."
            )

        trace_html = append_trace_step(
            trace_html,
            f"<span class='trace-status'>Error: {str(exc)}</span>",
        )

        st.session_state.trace_html = trace_html + "</div>"
        trace_container.markdown(
            st.session_state.trace_html,
            unsafe_allow_html=True,
        )

