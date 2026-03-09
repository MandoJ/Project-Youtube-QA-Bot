# Presentation Notes — YouTube QA Bot

---

## Slide 1 — Title
Project: YouTube QA Bot
A Multimodal RAG-Powered Conversational Assistant
Built with LangChain, ChromaDB, OpenAI, and Whisper

---

## Slide 2 — Problem Statement
- Video content is rich but hard to query
- Businesses and individuals can't efficiently extract information from hours of video
- No easy way to ask natural language questions about video content
- Auto-generated captions degrade on multilingual/bilingual content

**Supporting Facts:**
- YouTube has over 800 million videos as of 2024
- Over 500 hours of video are uploaded to YouTube every minute
- Only ~20% of video content is ever fully watched by viewers
- Businesses lose significant time manually reviewing video content for insights
- YouTube's auto-caption accuracy drops significantly on non-native English speakers
  and code-switching (mixing two languages), as demonstrated by our Bad Bunny test

---

## Slide 3 — Solution Overview
- Ingest YouTube transcripts and store them in a vector database
- Use a RAG pipeline to retrieve relevant context for any question
- Wrap in an LLM agent with tools and memory for conversational interaction
- Support voice input via Whisper for true multimodal experience
- Accept any YouTube URL and ingest it in real time — no retraining required
- Deploy as a web app accessible to anyone

**Supporting Facts:**
- RAG (Retrieval Augmented Generation) reduces LLM hallucinations by grounding
  answers in real source documents rather than relying solely on model memory
- Unlike fine-tuning, RAG allows the knowledge base to be updated without
  retraining the model — just add new transcripts and re-embed
- Whisper achieves near human-level transcription accuracy across 99 languages
- The entire pipeline from transcript ingestion to answer generation
  takes only a few seconds per query

**Business Use Case — Cross-Video Analysis:**
- Users can load multiple videos simultaneously without losing conversation context
- The chat history stays intact when switching or adding videos
- Example: load 3 competitor product videos and ask "compare the pricing models
  across all three" — the bot answers using context from all 3 simultaneously
- This cross-video querying capability is what separates this tool from
  standard video summarizers and makes it genuinely useful for business intelligence

---

## Slide 4 — Tech Stack
- LangChain + LangGraph for agent and tool orchestration
- ChromaDB for vector storage and similarity search
- OpenAI GPT-4o-mini as the reasoning LLM
- OpenAI text-embedding-3-small for embeddings
- Whisper (local, GPU-accelerated) for speech recognition
- Streamlit for the web interface
- LangSmith for tracing and evaluation
- Hardware: RTX 5080 + CUDA 12.8 for local inference

**Supporting Facts:**
- GPT-4o-mini scores 82% on MMLU benchmark — comparable to GPT-4 on most
  practical tasks at roughly 10x lower cost
- GPT-4o-mini costs $0.15 per 1M input tokens vs $5.00 for GPT-4o
- ChromaDB is fully local and open source — zero database hosting cost
- LangGraph replaced the older LangChain agents in v0.1+ for more reliable
  tool-calling and state management
- Running Whisper locally on RTX 5080 with CUDA 12.8 eliminates
  Whisper API costs entirely (~$0.006/minute otherwise)

---

## Slide 5 — Data Pipeline
- 9 YouTube transcripts across 3 categories: Education, Tech/AI, Entertainment
- Transcripts fetched via youtube-transcript-api
- Chunked using RecursiveCharacterTextSplitter (500 chars, 50 overlap)
- 514 total chunks embedded and stored in ChromaDB

**Embedding Cost Breakdown:**
- 514 chunks × 500 chars (avg) = ~257,000 characters
- ~257,000 chars ÷ 4 = ~64,000 tokens
- text-embedding-3-small = $0.02 per 1,000,000 tokens
- 64,000 ÷ 1,000,000 × $0.02 = ~$0.001 total
- One of the most cost-efficient embedding models available
- Entire dataset embedded once and persisted to disk — no recurring cost
- Chunk overlap of 50 chars preserves context across chunk boundaries,
  improving retrieval quality on conversational/spoken text

---

## Slide 6 — Agent Architecture
- Two tools: transcript retriever + general knowledge fallback
- LangGraph ReAct agent decides which tool to use per query
- Conversation memory via RunnableWithMessageHistory
- System prompt enforces citation of video source and category

**Supporting Facts:**
- ReAct (Reasoning + Acting) agents reason step by step before deciding
  which tool to call — more reliable than single-pass chains
- Having a general knowledge fallback means the bot never hits a dead end —
  it gracefully handles questions outside the video dataset
- k=4 retrieval means 4 chunks × ~500 chars = ~2,000 chars of context
  per query, well within GPT-4o-mini's 128k token context window
- Metadata filtering (by category) allows scoped queries —
  e.g. only search tech/AI videos for a technical question

---

## Slide 7 — Results & Demo
*(to be filled in as I progress)*
- Retrieval accuracy across categories
- Sample Q&A outputs
- Bilingual content handling observation
- Voice input demo

**Facts to add once complete:**
- LangSmith evaluation scores
- Average response latency
- Number of test queries run
- Accuracy rate on known-answer questions

---

## Slide 8 — Limitations & Observations
- Auto-generated YouTube captions degrade on bilingual content (Bad Bunny example)
- Whisper integration addresses this for audio input
- Memory across sessions to be improved in app layer

**Supporting Facts:**
- YouTube's ASR (Automatic Speech Recognition) is optimized for English —
  accuracy drops to as low as 40-60% on code-switched speech
- Whisper large-v3 achieves <5% Word Error Rate on English and handles
  97 languages natively, including Spanish
- Session memory is currently scoped to a single notebook run —
  persistent cross-session memory would require a database like Redis or SQLite

---

## Slide 9 — Live Demo
*(demo of the deployed Streamlit app)*

**Demo script suggestions:**
- Start with a tech/AI question to show RAG retrieval working cleanly
- Ask about Erwin's speech to show entertainment category working
- Ask a follow-up question to demonstrate memory
- Ask a voice question via Whisper to show multimodal capability
- Ask something outside the dataset to show graceful general knowledge fallback

---

## Slide 10 — Conclusion & Next Steps
*(to be filled in at the end)*

**Facts to add once complete:**
- Total development time
- Total API cost for the entire project
- Potential business applications
- What would be added with more time (more video categories,
  persistent memory, user authentication, custom video ingestion UI)