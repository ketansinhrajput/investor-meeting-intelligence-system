# Transcript Intelligence

> AI-powered earnings call transcript analyzer — extracts Q&A units, speaker registries, strategic statements, and topics from PDF transcripts with full LLM decision traceability.

---

## What It Does

Upload an earnings call transcript PDF and get back:

- **Structured Q&A** — every question and answer separated, follow-up chains linked, questioners identified
- **Speaker Registry** — all participants with role (management / analyst / moderator), title, and company
- **Strategic Statements** — forward-looking statements and key positions from opening/closing remarks
- **Enrichment** — topics, intent, and conversational posture per Q&A unit
- **LLM Traces** — every AI decision is logged with evidence spans for full auditability
- **Chat Agent** — ask natural language questions about any analyzed transcript

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (React 18)                        │
│   Login → Upload PDF → View Analysis → Chat with Transcript      │
└─────────────────────────┬────────────────────────────────────────┘
                          │ HTTP / SSE
┌─────────────────────────▼────────────────────────────────────────┐
│                   FastAPI Backend  :8100                          │
│   Auth (JWT + SQLite)  │  Run Storage (JSON files)               │
│   Chat Agent (2-phase) │  Pipeline Runner                        │
└─────────────────────────┬────────────────────────────────────────┘
                          │ Python call
┌─────────────────────────▼────────────────────────────────────────┐
│                    Pipeline V2 (Hybrid Intelligence)              │
│                                                                   │
│  PDF → Clean → Metadata → Boundaries → Speakers → Q&A → Enrich  │
│         regex    LLM        hybrid       LLM       blocks  LLM   │
└─────────────────────────┬────────────────────────────────────────┘
                          │ Ollama API
              ┌───────────▼───────────┐
              │  Ollama  :11434        │
              │  gemma3:latest         │
              └────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Ollama (`gemma3:latest` for pipeline, `gpt-oss:20b` for chat) |
| **Pipeline** | Python 3.12, LangChain, LangGraph, pdfplumber, Pydantic v2 |
| **Backend** | FastAPI, SQLAlchemy, SQLite, PyJWT, bcrypt, ChromaDB |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Framer Motion |

---

## Prerequisites

- **Python 3.12+**
- **Node.js 18+**
- **Ollama** running at `http://localhost:11434`

```bash
# Pull required models
ollama pull gemma3
ollama pull nomic-embed-text   # for chat vector search
```

---

## Setup

All virtual environments are pre-created. Just activate and run.

### Backend

```bash
cd backend
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
uvicorn main:app --reload --port 8100
```

- API: http://localhost:8100
- Docs: http://localhost:8100/docs
- SQLite DB auto-created at `backend/data/call_transcript.db`

### Frontend

```bash
cd frontend
npm run dev
```

- UI: http://localhost:3000

### Pipeline (CLI only)

```bash
# From repo root
.venv\Scripts\activate
python scripts/run_pipeline_v2.py data/input/transcript.pdf
python scripts/run_pipeline_v2.py data/input/transcript.pdf --skip-enrichment
```

---

## First-Time Setup

### Create an admin user

```bash
cd backend
venv\Scripts\activate
python scripts/manage_users.py create admin <password> --role admin
```

### Environment files

**`backend/.env`**
```env
JWT_SECRET_KEY=your-secret-key
LLM_MODEL_NAME=gemma3:latest
```

**`.env`** (root — pipeline only)
```env
LLM_OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL_NAME=gemma3:latest
```

---

## Pipeline V2 — How It Works

Philosophy: **Regex proposes (high recall) → LLM labels and structures → Code enforces invariants**

```
Stage 0  PDF Extraction      pdfplumber → raw text per page
Stage 0B Text Cleaning       Regex removes page headers, page markers
Stage 1  Metadata            LLM extracts company, ticker, quarter, year
Stage 2  Boundary Detection  Regex proposes sections, LLM confirms
Stage 3  Speaker Registry    LLM sees ALL candidates at once → canonical names + roles
Stage 4  Q&A Extraction      Regex detects investor blocks → LLM structures each block
Stage 5  Strategic           LLM extracts statements from opening/closing sections
Stage 6  Enrichment          LLM adds topics, intent, posture (skippable)
```

Every stage emits a **trace** with LLM inputs, outputs, and evidence spans — visible in the UI Traces tab.

**Key rule:** LLM may extend or merge content, never silently drop it.

---

## Web UI

Log in at http://localhost:3000, upload a PDF transcript, and explore the results across six tabs:

| Tab | Contents |
|---|---|
| **Overview** | Run stats, stage status, errors/warnings, timing |
| **Speakers** | Speaker table with role, title, company, alias merges, LLM reasoning |
| **Q&A Explorer** | Q&A pairs, follow-up chains, questioner, page references |
| **Analyst Summary** | Executive summary of key themes and positions |
| **Traces** | Per-stage LLM decisions with evidence spans |
| **Raw Text** | Per-page extracted text |
| **Raw JSON** | Full pipeline state, copy to clipboard |

The **chat drawer** (available on any run detail view) lets you ask natural-language questions about the transcript. The agent uses structured retrieval first, with ChromaDB vector search as fallback, and cites specific Q&A units and page references in every answer.

---

## API Reference

All endpoints are under `/api/`:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Get JWT token |
| `POST` | `/api/upload` | Upload PDF |
| `POST` | `/api/analyze` | Start analysis run |
| `GET` | `/api/runs` | List all runs |
| `GET` | `/api/runs/{id}/summary` | Run stats and stage status |
| `GET` | `/api/runs/{id}/speakers` | Speaker registry |
| `GET` | `/api/runs/{id}/qa` | Q&A units |
| `GET` | `/api/runs/{id}/traces` | LLM decision traces |
| `GET` | `/api/runs/{id}/raw` | Raw extracted text |
| `GET` | `/api/runs/{id}/json` | Full pipeline JSON |
| `POST` | `/api/chat/{id}` | Chat with transcript (streaming SSE) |
| `DELETE` | `/api/runs/{id}` | Delete a run |
| `GET` | `/api/health` | Health check |

---

## Development

```bash
# Linting
cd .venv/Scripts && activate
ruff check src/
ruff format src/

# Type checking
mypy src/

# Tests
pytest                          # all tests
pytest tests/unit/              # unit tests only
pytest tests/unit/test_models.py  # single file

# Frontend type check
cd frontend && npm run lint
```

---

## Data Storage

```
backend/data/
├── call_transcript.db          # SQLite — users and auth
├── uploads/                    # Uploaded PDFs
└── runs/{run_id}/
    ├── metadata.json           # Status, timestamps, stage progress
    ├── stage_extraction_result.json
    ├── stage_speakers_result.json
    ├── stage_speakers_trace.json
    ├── stage_qa_result.json
    ├── stage_qa_trace.json
    └── pipeline_output.json    # Full PipelineV2State
```

To clear all runs:
```bash
rm -rf backend/data/runs/*
curl -X DELETE http://localhost:8100/api/runs/{run_id}
```

---

## Known Limitations

- Pipeline runs **synchronously** — no live stage progress until completion
- Requires Ollama running locally with `gemma3` pulled
- Windows console: avoid Unicode characters in log output
