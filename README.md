# Agentic Multimodal RAG

A production-oriented Retrieval-Augmented Generation system that ingests PDFs (text, tables, and images) via Sarvam OCR parsing, indexes them in FAISS, and answers questions through a LangGraph agentic retrieval pipeline. Backend is FastAPI; frontend is a React chat UI.

## Architecture

```
PDF Upload → Split → Sarvam OCR → Parse (text/images/tables) → Chunk → Embed → FAISS
                                                                              ↓
User Query → Agent → Retrieve → Rerank → Route → Generate Answer (GPT-4o-mini)
```

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Agent orchestration | LangGraph |
| Document OCR | Sarvam AI Document Intelligence |
| Vector store | FAISS (per-embedder index) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`, 384-dim) or OpenAI (`text-embedding-3-small`, 1536-dim) |
| Reranking | Cohere `rerank-v3.5` or local CrossEncoder |
| Generation | OpenAI `gpt-4o-mini` |
| Frontend | React + plain CSS |

## Directory Structure

```
rag-backend/
├── app/
│   ├── main.py                  # FastAPI app, routes, CORS
│   ├── config.py                # paths, model names, defaults
│   ├── api/
│   │   └── routes_ingest.py     # /upload, /ingest, /documents
│   ├── core/
│   │   ├── embedders.py         # HF / OpenAI embedder wrappers
│   │   ├── rerankers.py         # Cohere / local reranker
│   │   └── faiss_store.py       # FAISS index wrapper
│   ├── agent/
│   │   ├── state.py             # LangGraph state schema
│   │   ├── nodes.py             # agent/retrieve/rerank/route/generate
│   │   ├── graph.py             # graph wiring
│   │   └── query.py             # query entrypoint
│   └── ingestion/
│       ├── pdf_split.py         # splits large PDFs into ≤N-page chunks
│       ├── sarvam_parser.py     # Sarvam OCR calls + markdown parsing
│       └── pipeline.py          # orchestrates split → OCR → parse → index
├── data/
│   ├── docs/                    # uploaded PDFs
│   ├── artifacts/
│   │   ├── pdf_chunks/          # split PDF parts
│   │   ├── sarvam_outputs/      # OCR result zips
│   │   ├── extracted_content/   # unzipped markdown
│   │   ├── extracted_images/    # decoded images
│   │   └── faiss_index/{hf,openai}/
│   └── chat_history/
└── requirements.txt

rag-frontend/
└── src/
    ├── App.jsx
    ├── ChatApp.jsx               # chat UI, settings modal, upload
    └── App.css
```

## Ingestion Pipeline

1. **Split** — PDFs over `max_pages` are split into chunks via `pypdf` (`pdf_split.py`).
2. **OCR** — Each chunk is sent to Sarvam AI Document Intelligence, returning markdown with embedded base64 images (`sarvam_parser.py`).
3. **Clean** — `clean_raw_markdown` removes hallucinated repeated words, phrases, and punctuation runs from the raw OCR output. This is the only cleaning step; it runs once, before any extraction.
4. **Parse** — From the cleaned markdown:
   - **Text**: everything except base64 image blobs.
   - **Tables**: markdown pipe-tables (including those wrapped in stray `*`) and HTML `<table>` blocks. Captions are located by searching preceding text for a `Table N:` pattern first, falling back to following text.
   - **Images**: base64-decoded to PNG. Description/caption are extracted from an 8000-character window following the image; if no explicit caption is found, the first 5000 characters of that window are used as the embedding text.
5. **Chunk** — Text blocks are split via `RecursiveCharacterTextSplitter` (user-configurable `chunk_size` / `chunk_overlap`, default 1000/200).
6. **Embed & Index** — Text, image, and table content is embedded and appended to the FAISS index for the selected embedder. Each embedder has its own index directory (`faiss_index/hf/`, `faiss_index/openai/`) since dimensions differ (384 vs 1536).

Ingestion is **incremental**: uploading a new PDF only re-runs OCR and embedding for that file, appending new vectors to the existing index rather than rebuilding from scratch. A full rebuild (`mode=index`) re-parses all stored OCR outputs and regenerates the index from zero — used when switching embedders or after a parsing-logic change.

Deleting a document removes both the PDF file and its corresponding vectors from every embedder's FAISS index (by rebuilding the index excluding matching sources).

## Agentic Retrieval

Implemented as a LangGraph state machine with five nodes:

| Node | Responsibility |
|---|---|
| `agent` | Decides the next step based on current state (retrieve → rerank → route → answer), enforces `max_loops` |
| `retrieve` | Embeds the query and searches FAISS (`top_k_retrieve`) |
| `rerank` | Reorders retrieved docs via Cohere or local CrossEncoder (`top_n_rerank`) |
| `route` | Splits ranked docs by content type (text / image / table) |
| `generate_answer` | Builds a multimodal prompt (text, table content, image captions + actual images) and calls GPT-4o-mini |

Each request is stateless — no LangGraph checkpointer. Chat history is passed in and injected into the query text directly, avoiding stale cached state across turns.


## Setup

### Backend

```bash
cd rag-backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Required environment variables (`.env`):

```
OPENAI_API_KEY=
COHERE_API_KEY=
SARVAM_API_KEY=
```

### Frontend

```bash
cd rag-frontend
npm install
npm run dev
```

Built by- Sarthak 😊
