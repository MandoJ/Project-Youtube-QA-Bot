# Tubot — YouTube QA Bot

A multimodal RAG-powered chatbot that lets you have a conversation with any YouTube video. Paste a URL, ask questions, get cited answers. Built in 5 days as an Ironhack final project.

---

## What it does

- Ingests any YouTube video in real time — no retraining, no restart
- Answers natural language questions about video content, citing the source
- Handles voice input via local Whisper (GPU-accelerated)
- Maintains conversation memory across an entire session
- Supports multiple videos simultaneously — cross-video queries work out of the box
- Falls back to general knowledge when a question is outside the dataset

---

## Stack

| Component | Role |
|---|---|
| LangChain + LangGraph | ReAct agent and tool orchestration |
| ChromaDB | Local vector store — fully persistent, zero hosting cost |
| GPT-4o-mini | Reasoning and answer generation |
| text-embedding-3-small | Transcript embeddings |
| Whisper (medium, local) | Speech-to-text via GPU |
| Streamlit | Web interface |
| LangSmith | Tracing and evaluation |

**Hardware:** RTX 5080 · CUDA 12.8 · Python 3.12.3 · WSL Ubuntu

---

## Project structure

```
Project-Youtube-QA-Bot/
├── app.py                    # Streamlit app — entry point
├── src/
│   ├── agent.py              # LangGraph ReAct agent + tools
│   ├── ingestion.py          # Transcript fetching, chunking, embedding
│   └── whisper_utils.py      # Local Whisper transcription
├── notebooks/
│   ├── 1_data_collection.ipynb
│   ├── 2_pipeline.ipynb
│   ├── 3_agent.ipynb
│   └── 4_whisper.ipynb
├── data/
│   ├── transcripts/          # Raw transcripts by category
│   └── vectorstore/          # ChromaDB persisted to disk
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting started

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.8 (for local Whisper)
- OpenAI API key
- LangSmith API key

### Installation

```bash
git clone https://github.com/MandoJ/Project-Youtube-QA-Bot.git
cd Project-Youtube-QA-Bot

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=youtube-qa-bot
```

### Run the app

```bash
.venv/bin/python -m streamlit run app.py
```

---

## How it works

1. **Ingest** — A YouTube URL is passed to `ingestion.py`, which fetches the transcript via `youtube-transcript-api`, splits it into 500-character chunks with 50-character overlap, embeds them using `text-embedding-3-small`, and stores them in ChromaDB.

2. **Retrieve** — On each query, a semantic search pulls the 6 most relevant chunks from the vector store.

3. **Reason** — A LangGraph ReAct agent decides whether to use the transcript retriever or fall back to general knowledge.

4. **Respond** — GPT-4o-mini generates a cited answer referencing the source video and category. Conversation history is maintained via `RunnableWithMessageHistory`.

5. **Voice** — Whisper runs locally on GPU and converts spoken questions to text before passing them to the agent.

---

## Dataset

9 videos across 3 categories — 514 total chunks embedded at a cost of ~$0.001.

| Category | Videos |
|---|---|
| Education | Neural Networks, Gradient Descent, Transformers |
| Tech / AI | GPT-4 Announcement, LangChain Crash Course, RAG Explained |
| Entertainment | Bad Bunny on Hot Ones, Erwin AOT Speech, Fantano IGOR Review |

---

## Results

Tracked via LangSmith over 50 traces during development:

- **Error rate:** 0%
- **Median latency (P50):** 4.91s
- **Total API cost:** $0.02

---

## Known limitations

- No timestamp-level retrieval — chunks don't carry segment timestamps yet
- Memory is session-scoped — closing the tab resets the conversation
- Phrasing sensitivity — semantically similar but differently worded queries can return different chunks

---

## What's next

- Clickable timestamps linking answers back to the source moment in the video
- Persistent cross-session memory via Redis or SQLite
- Public deployment on Streamlit Community Cloud
- User authentication for private video libraries

---

## Cost breakdown

| Item | Cost |
|---|---|
| Embeddings (514 chunks, text-embedding-3-small) | ~$0.001 |
| GPT-4o-mini usage during development | ~$0.02 |
| Whisper (local GPU) | $0.00 |
| ChromaDB (local) | $0.00 |
| **Total** | **< $0.03** |
