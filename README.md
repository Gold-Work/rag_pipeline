# RAG Pipeline

A production-ready Retrieval-Augmented Generation system with a hybrid search engine, a LangGraph-powered customer service agent, and a React frontend. Built for French-language document corpora.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Ingestion pipeline                         │
│  PDF/HTML/TXT → Loader → Parser → Chunker → Embedder → ChromaDB    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                           Vector index
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                           Query pipeline                            │
│                                                                     │
│  Question                                                           │
│     │                                                               │
│     ├─ Query rewriting (3 variants, GPT-4o-mini)                    │
│     │                                                               │
│     ├─ Vector search  (text-embedding-3-small, ChromaDB)            │
│     ├─ BM25 keyword search (rank-bm25 / Okapi TF-IDF)              │
│     │       └── Reciprocal Rank Fusion                              │
│     │                                                               │
│     ├─ Cross-encoder reranking (mmarco-mMiniLMv2, multilingual)     │
│     ├─ Prompt augmentation                                          │
│     └─ Generation (GPT-4o-mini) ──► Answer + sources + latencies   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                        SAV agent (LangGraph)                        │
│                                                                     │
│  Customer message                                                   │
│     │                                                               │
│     ▼                                                               │
│  [decide]  ── tool ──► [execute] ──► [respond] ──► Final reply      │
│     └──── none ───────────────────────────┘                        │
│                                                                     │
│  Tools: query_rag · check_order · send_email · create_ticket        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Hybrid retrieval** — vector search (OpenAI embeddings) + BM25 combined via Reciprocal Rank Fusion
- **Query rewriting** — expands each question into 3 variants to improve recall
- **Cross-encoder reranking** — multilingual model re-scores top-20 chunks before generation
- **Content-addressed cache** — TTL cache keyed on `question + top_k + index version + chunk IDs`; auto-invalidates on re-ingestion
- **FastAPI REST API** — JWT auth, rate limiting (30 req/min), file upload, deep health check
- **LangGraph SAV agent** — 3-node graph (decide / execute / respond) with structured LLM routing and fuzzy order-ID resolution
- **React frontend** — search UI with latency bars and cache HIT/MISS badge
- **RAGAS evaluation** — automated quality gate (faithfulness, answer relevancy, context precision ≥ 0.75) against a 10-question golden dataset
- **Observability** — structured JSON logs, Langfuse traces for every span and LLM generation
- **CI/CD** — GitHub Actions: lint, type-check, security scan, tests, RAGAS eval, Docker build

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM & embeddings | OpenAI GPT-4o-mini, text-embedding-3-small |
| Vector store | ChromaDB (persistent) |
| Keyword search | rank-bm25 (Okapi TF-IDF) |
| Reranker | sentence-transformers cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 |
| Agent framework | LangGraph 1.x |
| API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |
| Evaluation | RAGAS 0.2.x |
| Observability | Langfuse |
| Document parsing | LangChain loaders (PDF, HTML, TXT) |
| Containerisation | Docker + docker-compose |

---

## Project structure

```
├── api/                  FastAPI application
├── frontend/             React + Vite UI
├── scripts/
│   ├── run_ingestion.py  Index documents into ChromaDB
│   ├── run_query.py      One-shot query from the terminal
│   ├── run_agent.py      SAV agent from the terminal
│   ├── eval_pipeline.py  RAGAS quality evaluation
│   └── verify_ingestion.py  Inspect indexed chunks
├── src/
│   ├── agent/            LangGraph SAV agent (state, tools, graph)
│   ├── ingestion/        Loader, parser, chunker, embedder
│   ├── query/            Retriever, reranker, augmenter, generator, cache
│   └── utils/            Config, logger, auth, observability
├── tests/                pytest test suite
├── config.yaml           Central configuration
└── docker-compose.yml
```

---

## Getting started

### Prerequisites

- Python 3.12
- An OpenAI API key
- (Optional) Langfuse account for observability

### 1. Environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, API_USERNAME, API_PASSWORD
# and optionally LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Index documents

Place PDF, HTML, or TXT files in `data/raw/`, then run:

```bash
python scripts/run_ingestion.py
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# UI available at http://localhost:3000
```

### 6. Query from the terminal

```bash
python scripts/run_query.py "What is the return policy?"
```

### 7. Run the SAV agent

```bash
python scripts/run_agent.py "Where is my order CMD-2025-00142?"
```

### 8. Run with Docker

```bash
docker-compose up --build
```

---

## Configuration

All parameters are centralised in `config.yaml`:

| Key | Default | Description |
|---|---|---|
| `retrieval.top_k_retrieval` | 20 | Candidates retrieved before reranking |
| `retrieval.top_k_rerank` | 5 | Chunks kept after reranking |
| `retrieval.chunk_size` | 1200 | Target chunk length in characters |
| `retrieval.chunk_overlap` | 200 | Overlap between consecutive chunks |
| `embedding.model` | text-embedding-3-small | OpenAI embedding model |
| `rewriter.n_variants` | 3 | Query variants generated per question |
| `llm.model` | gpt-4o-mini | Generation model |
| `cache.ttl_seconds` | 3600 | Cache TTL in seconds |

---

## Evaluation

```bash
# Requires indexed documents and OPENAI_API_KEY
python scripts/eval_pipeline.py
```

Runs RAGAS metrics (faithfulness, answer relevancy, context precision) against a 10-question golden dataset. Exits with code 1 if any metric falls below 0.75.

---

## Running tests

```bash
pytest tests/ -v --cov=src --cov=api
```
