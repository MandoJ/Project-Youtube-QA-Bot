"""
Tubot — Streamlit Web App
Day 5: Split-screen chat interface with video library, mic input, and dynamic ingestion.
"""

import os
import json
import tempfile
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent

import torch
import whisper as whisper_module
from youtube_transcript_api import YouTubeTranscriptApi
import re

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(".env"))

VECTORSTORE_DIR = Path("data/vectorstore")
TRANSCRIPTS_DIR = Path("data/transcripts")

CATEGORIES = ["all", "education", "tech_ai", "entertainment"]

KNOWN_VIDEOS = [
    {"video_id": "aircAruvnKk", "title": "neural_networks_explained",   "category": "education"},
    {"video_id": "WUvTyaaNkzM", "title": "gradient_descent_explained",  "category": "education"},
    {"video_id": "Ilg3gGewQ5U", "title": "transformers_explained",      "category": "education"},
    {"video_id": "hM_h0UA7upI", "title": "openai_gpt4_announcement",    "category": "tech_ai"},
    {"video_id": "nAmC7SoVLd8", "title": "langchain_crash_course",      "category": "tech_ai"},
    {"video_id": "qbIk7-JPB2c", "title": "rag_explained",               "category": "tech_ai"},
    {"video_id": "u-ZBLZATlhM", "title": "erwin_speech_aot",            "category": "entertainment"},
    {"video_id": "HyqEtb5HE50", "title": "bad_bunny_hot_ones",          "category": "entertainment"},
    {"video_id": "u37lG0BMFZk", "title": "fantano_igor_review",         "category": "entertainment"},
]

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tubot",
    page_icon="assets/favicon.png" if Path("assets/favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ---- CSS variables ---- */
:root {
    --bg:        #141414;
    --surface:   #1c1c1e;
    --surface2:  #242426;
    --border:    #2e2e30;
    --border2:   #3a3a3c;
    --accent:    #e5484d;
    --accent-dim: rgba(229,72,77,0.15);
    --text:      #f0f0f0;
    --text-2:    #a0a0a8;
    --text-3:    #606068;
    --radius:    10px;
    --radius-sm: 6px;
}

/* ---- Remove all default Streamlit padding ---- */
[data-testid="stAppViewContainer"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stMain"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stMain"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
}
.block-container {
    padding-top: 0 !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* ---- Reset & base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}

[data-testid="stMain"] {
    background-color: var(--bg) !important;
}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ---- App header ---- */
.app-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    padding: 1.25rem 1.75rem 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0;
    background: linear-gradient(180deg, #1a1a1c 0%, var(--bg) 100%);
}

.header-dot {
    width: 9px;
    height: 9px;
    background: var(--accent);
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 2.8s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(229,72,77,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 4px rgba(229,72,77,0); }
}

.header-label { color: var(--accent); font-weight: 700; }
.header-sub {
    font-size: 0.75rem;
    font-weight: 400;
    color: var(--text-3);
    margin-left: auto;
}

/* ---- Panel divider ---- */
.panel-divider {
    width: 1px;
    background: var(--border);
    align-self: stretch;
    margin: 0 0.25rem;
    min-height: 400px;
}

/* ---- Chat panel ---- */
.chat-messages {
    height: 70vh;
    overflow-y: auto;
    padding: 1.25rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    scrollbar-width: thin;
    scrollbar-color: var(--border2) transparent;
}

.chat-messages::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

.bubble-row {
    display: flex;
    width: 100%;
}
.bubble-row.user  { justify-content: flex-end; }
.bubble-row.bot   { justify-content: flex-start; }

.bubble {
    max-width: 80%;
    padding: 0.8rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.65;
}

.bubble.user {
    background: var(--accent);
    background: linear-gradient(135deg, #e5484d, #c0392b);
    border-radius: var(--radius) var(--radius) 2px var(--radius);
    color: #fff;
    font-weight: 400;
    box-shadow: 0 2px 12px rgba(229,72,77,0.25);
}

.bubble.bot {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) var(--radius) 2px;
    color: var(--text);
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.bubble-label {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 0.35rem;
}
.bubble-label.user { color: var(--accent); text-align: right; }

.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-2);
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    margin-top: 0.5rem;
    margin-right: 0.3rem;
}

/* ---- Text inputs ---- */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-3) !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(229,72,77,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ---- Section labels ---- */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-3);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ---- Video cards ---- */
.video-card {
    display: flex;
    gap: 0.85rem;
    align-items: center;
    padding: 0.75rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 0.6rem;
    background: var(--surface);
    transition: all 0.2s;
    cursor: default;
}
.video-card:hover {
    border-color: var(--border2);
    background: var(--surface2);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}

.video-thumb {
    width: 96px;
    height: 56px;
    object-fit: cover;
    border-radius: var(--radius-sm);
    flex-shrink: 0;
}

.video-thumb-placeholder {
    width: 96px;
    height: 56px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-3);
    font-size: 0.6rem;
}

.video-meta { flex: 1; min-width: 0; }
.video-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.4;
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.video-cat {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    display: inline-block;
}
.video-cat.education     { color: #60a5fa; background: rgba(96,165,250,0.1); }
.video-cat.tech_ai       { color: #34d399; background: rgba(52,211,153,0.1); }
.video-cat.entertainment { color: #c084fc; background: rgba(192,132,252,0.1); }

/* ---- Selectbox ---- */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ---- Status bar ---- */
.status-bar {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-3);
    padding: 0.35rem 0;
}
.status-ok  { color: #34d399; }
.status-err { color: var(--accent); }

/* ---- Scrollbar for right panel ---- */
section[data-testid="column"]:nth-child(3) {
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <div class="header-dot"></div>
    <span><span class="header-label">TUBOT</span>
    <span class="header-sub">RAG &middot; LangGraph &middot; Whisper &middot; GPT-4o-mini</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": "user"|"bot", "content": str, "sources": list}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "video_library" not in st.session_state:
    st.session_state.video_library = list(KNOWN_VIDEOS)

if "category_filter" not in st.session_state:
    st.session_state.category_filter = "all"

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = ""

# ---------------------------------------------------------------------------
# Backend loaders — cached so they only run once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vs = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embedding_model,
        collection_name="youtube_qa",
    )
    return vs, embedding_model


@st.cache_resource(show_spinner=False)
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper_module.load_model("medium", device=device)


def build_agent(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="youtube_transcript_search",
        description=(
            "Search YouTube video transcripts to answer questions about video content. "
            "ALWAYS use this tool first for any question — even if it seems general. "
            "The transcripts cover education (neural networks, transformers, gradient descent), "
            "tech/AI (GPT-4, LangChain, RAG), and entertainment (Bad Bunny Hot Ones, "
            "Erwin's speech from Attack on Titan, Fantano's IGOR review). "
            "Use this tool before considering any other source."
        ),
    )

    @tool
    def general_knowledge(query: str) -> str:
        """
        Answer questions that are completely unrelated to any YouTube video content.
        Only use this tool if youtube_transcript_search returned no relevant results
        and the question is clearly outside the scope of the video library.
        Do NOT use this tool if the question could plausibly relate to any video.
        """
        response = llm.invoke(query)
        return response.content

    system_prompt = """You are a helpful assistant with access to YouTube video transcripts.

Your primary job is to answer questions using the video transcripts. Follow these rules strictly:

1. ALWAYS call youtube_transcript_search first, for every question without exception.
2. Base your answer on what the transcripts actually say — do not invent or embellish.
3. Always tell the user which video title and category your answer came from.
4. If the transcript content is unclear or garbled, say so honestly rather than guessing.
5. Only use general_knowledge if the transcript search returns nothing relevant AND the
   question is completely unrelated to any video in the library.
6. Keep answers concise and grounded in the source material.
"""

    agent_executor = create_react_agent(
        model=llm,
        tools=[retriever_tool, general_knowledge],
        prompt=system_prompt,
    )

    def get_session_history(session_id: str):
        return st.session_state.chat_history

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="chat_history",
    )

    return agent_with_memory


# ---------------------------------------------------------------------------
# Helper: extract video ID from a YouTube URL
# ---------------------------------------------------------------------------

def extract_video_id(url: str):
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Helper: ingest a new video URL into the vectorstore
# ---------------------------------------------------------------------------

def ingest_video(url: str, title: str, category: str, vectorstore, embedding_model,
                 use_whisper: bool = False) -> bool:
    video_id = extract_video_id(url)
    if not video_id:
        st.session_state.ingestion_status = "error: could not parse video ID from URL"
        return False

    # Check duplicate
    existing_ids = [v["video_id"] for v in st.session_state.video_library]
    if video_id in existing_ids:
        st.session_state.ingestion_status = f"already in library: {video_id}"
        return False

    if use_whisper:
        # Download audio with yt-dlp then transcribe with Whisper
        try:
            import subprocess
            import sys

            audio_dir = Path("data/audio")
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{title}.mp3"

            yt_dlp_path = Path(sys.executable).parent / "yt-dlp"

            subprocess.run([
                str(yt_dlp_path),
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", str(audio_path),
                url,
            ], check=True, capture_output=True)

            whisper_model = st.session_state.whisper_model
            result = whisper_model.transcribe(
                str(audio_path),
                fp16=torch.cuda.is_available(),
            )
            text = result["text"].strip()

        except Exception as e:
            st.session_state.ingestion_status = f"error during Whisper transcription: {e}"
            return False
    else:
        # Fast path — fetch YouTube auto-captions
        try:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id)
            text = " ".join(s.text for s in fetched)
        except Exception as e:
            st.session_state.ingestion_status = f"error fetching transcript: {e}"
            return False

    # Chunk and embed
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "title":    title,
                "category": category,
                "video_id": video_id,
                "source":   url,
                "chunk_id": i,
            },
        )
        for i, chunk in enumerate(raw_chunks)
    ]

    vectorstore.add_documents(docs)

    # Save transcript to disk
    save_dir = TRANSCRIPTS_DIR / category
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{title}.json", "w") as f:
        json.dump({"video_id": video_id, "title": title, "category": category,
                   "transcript": text}, f, indent=2)

    # Update library in session
    st.session_state.video_library.append({
        "video_id": video_id,
        "title":    title,
        "category": category,
    })

    st.session_state.ingestion_status = f"ok: added {len(docs)} chunks for '{title}'"
    return True


# ---------------------------------------------------------------------------
# Helper: ask the agent
# ---------------------------------------------------------------------------

def ask_agent(question: str) -> dict:
    agent = st.session_state.agent
    if agent is None:
        return {"answer": "Agent not loaded yet.", "sources": []}

    response = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"session_id": "streamlit_session"}},
    )

    answer = response["messages"][-1].content

    # Pull source titles from intermediate tool messages
    sources = []
    for msg in response["messages"]:
        content = msg.content if isinstance(msg.content, str) else ""
        # Source titles appear as "title" in chunk metadata — scan for known ones
        for video in st.session_state.video_library:
            t = video["title"]
            if t in content and t not in sources:
                sources.append(t)

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Helper: transcribe with Whisper
# ---------------------------------------------------------------------------

def transcribe(audio_bytes: bytes) -> str:
    whisper_model = st.session_state.whisper_model
    if whisper_model is None:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    result = whisper_model.transcribe(
        tmp_path,
        fp16=torch.cuda.is_available(),
    )
    os.unlink(tmp_path)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Load resources on first run
# ---------------------------------------------------------------------------

with st.spinner("Loading vectorstore..."):
    vs, emb_model = load_vectorstore()
    st.session_state.vectorstore = vs

with st.spinner("Building agent..."):
    if st.session_state.agent is None:
        st.session_state.agent = build_agent(vs)

with st.spinner("Loading Whisper..."):
    if st.session_state.whisper_model is None:
        st.session_state.whisper_model = load_whisper()

# ---------------------------------------------------------------------------
# Layout — left chat panel | divider | right library panel
# ---------------------------------------------------------------------------

left_col, div_col, right_col = st.columns([5, 0.04, 3.5])

# ============================================================
# LEFT — Chat panel
# ============================================================

with left_col:
    st.markdown('<div class="section-label">Chat</div>', unsafe_allow_html=True)

    # Message history render
    chat_html = '<div class="chat-messages" id="chat-scroll">'
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            chat_html += f"""
            <div class="bubble-row user">
                <div>
                    <div class="bubble-label user">You</div>
                    <div class="bubble user">{content}</div>
                </div>
            </div>"""
        else:
            source_chips = ""
            for s in sources:
                source_chips += f'<span class="source-chip">{s}</span>'
            chat_html += f"""
            <div class="bubble-row bot">
                <div>
                    <div class="bubble-label">Assistant</div>
                    <div class="bubble bot">{content}{('<br>' + source_chips) if source_chips else ''}</div>
                </div>
            </div>"""

    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Auto-scroll script
    st.markdown("""
    <script>
        const el = document.getElementById('chat-scroll');
        if (el) el.scrollTop = el.scrollHeight;
    </script>
    """, unsafe_allow_html=True)

    # Input row — wrapped in a form so Enter key submits
    with st.form(key="chat_form", clear_on_submit=True):
        input_col, mic_col, send_col = st.columns([7, 1.2, 1.2])

        with input_col:
            user_input = st.text_input(
                label="message",
                label_visibility="collapsed",
                placeholder="Ask anything about the videos...",
                key="chat_input",
            )

        with mic_col:
            audio = mic_recorder(
                start_prompt="Mic",
                stop_prompt="Stop",
                just_once=True,
                use_container_width=True,
                key="mic_input",
            )

        with send_col:
            send_clicked = st.form_submit_button("Send", use_container_width=True)

    # --- Handle mic input ---
    if audio and audio.get("bytes"):
        with st.spinner("Transcribing your question..."):
            transcribed = transcribe(audio["bytes"])
        if transcribed:
            st.session_state.messages.append({"role": "user", "content": transcribed, "sources": []})
            with st.spinner("Thinking..."):
                result = ask_agent(transcribed)
            st.session_state.messages.append({
                "role": "bot",
                "content": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()

    # --- Handle text input ---
    if send_clicked and user_input.strip():
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question, "sources": []})
        with st.spinner("Thinking..."):
            result = ask_agent(question)
        st.session_state.messages.append({
            "role": "bot",
            "content": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear chat", key="clear_btn"):
            st.session_state.messages = []
            st.session_state.chat_history = ChatMessageHistory()
            st.session_state.agent = build_agent(st.session_state.vectorstore)
            st.rerun()

# ============================================================
# DIVIDER
# ============================================================

with div_col:
    st.markdown('<div class="panel-divider"></div>', unsafe_allow_html=True)

# ============================================================
# RIGHT — Video library panel
# ============================================================

with right_col:

    # -- URL ingestion --
    st.markdown('<div class="section-label">Add Video</div>', unsafe_allow_html=True)

    with st.container():
        url_input = st.text_input(
            label="youtube_url",
            label_visibility="collapsed",
            placeholder="https://youtube.com/watch?v=...",
            key="url_input",
        )
        title_input = st.text_input(
            label="video_title",
            label_visibility="collapsed",
            placeholder="video_title_slug",
            key="title_input",
        )
        cat_select = st.selectbox(
            label="category",
            label_visibility="collapsed",
            options=["education", "tech_ai", "entertainment"],
            key="cat_select",
        )
        use_whisper = st.toggle(
            "Use Whisper transcription (slower, more accurate)",
            value=False,
            key="whisper_toggle",
        )
        ingest_clicked = st.button("Ingest", use_container_width=True, key="ingest_btn")

    if ingest_clicked and url_input.strip() and title_input.strip():
        spinner_msg = "Downloading audio and transcribing with Whisper..." if use_whisper else "Fetching captions and embedding..."
        with st.spinner(spinner_msg):
            success = ingest_video(
                url=url_input.strip(),
                title=title_input.strip(),
                category=cat_select,
                vectorstore=st.session_state.vectorstore,
                embedding_model=emb_model,
                use_whisper=use_whisper,
            )
        if success:
            # Rebuild agent with updated vectorstore (no chat context lost)
            st.session_state.agent = build_agent(st.session_state.vectorstore)
        st.rerun()

    if st.session_state.ingestion_status:
        status = st.session_state.ingestion_status
        css_class = "status-ok" if status.startswith("ok") else "status-err"
        st.markdown(
            f'<div class="status-bar {css_class}">{status}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Collapsible video library --
    with st.expander(f"Video Library ({len(st.session_state.video_library)} videos)", expanded=True):

        # Category filter buttons
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        filter_map = {
            "all":           filter_col1,
            "education":     filter_col2,
            "tech_ai":       filter_col3,
            "entertainment": filter_col4,
        }
        filter_labels = {
            "all": "All",
            "education": "Edu",
            "tech_ai": "AI",
            "entertainment": "Ent",
        }

        for cat, col in filter_map.items():
            with col:
                if st.button(filter_labels[cat], key=f"filter_{cat}", use_container_width=True):
                    st.session_state.category_filter = cat
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Video cards
        active_filter = st.session_state.category_filter
        visible_videos = [
            v for v in st.session_state.video_library
            if active_filter == "all" or v["category"] == active_filter
        ]

        for video in visible_videos:
            vid_id  = video["video_id"]
            title   = video["title"].replace("_", " ")
            cat     = video["category"]
            thumb   = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"
            yt_url  = f"https://www.youtube.com/watch?v={vid_id}"

            st.markdown(f"""
            <div class="video-card">
                <a href="{yt_url}" target="_blank">
                    <img class="video-thumb" src="{thumb}" alt="{title}" onerror="this.style.display='none'">
                </a>
                <div class="video-meta">
                    <div class="video-title">{title}</div>
                    <div class="video-cat {cat}">{cat}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Doc count
        try:
            count = st.session_state.vectorstore._collection.count()
            st.markdown(
                f'<div class="status-bar" style="margin-top:0.5rem;">{count} chunks indexed</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass
