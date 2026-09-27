# Medical Knowledge Base API

A FastAPI service that answers clinical questions from guideline PDFs. The PDFs
are embedded with `all-MiniLM-L6-v2` and stored in a local FAISS index, which is
the only data store: there is no database to provision or migrate. Routes,
controllers, services and repositories are plain functions; Pydantic classes
validate API data.

## Run it

Install the project and development test tools:

```powershell
uv sync --group dev
```

Start the development server:

```powershell
uv run fastapi dev main.py
```

Open <http://127.0.0.1:8000/docs> to try the generated OpenAPI documentation.
On startup the application loads the embedding model and indexes every PDF in
`src/data` into the FAISS vector store so the assistant has something to search.
To add knowledge, drop a PDF into `src/data` and restart. Files are tracked by a
hash of their content, so unchanged PDFs are not re-embedded, edited ones are
re-indexed, and deleted ones are removed from the index. Scanned PDFs without a
text layer are skipped with a warning.

> **Run one worker only.** Chat sessions are held in process memory. Under more
> than one uvicorn worker, consecutive requests for the same session land on
> different processes at random and conversation memory appears to vanish
> intermittently. Use `--workers 1` until session memory is moved into
> a shared store. `fastapi dev` is already single-worker. The FAISS index is
> also held in process memory, so a second worker would not see uploads made
> through the first.

## The question-answering assistant

`POST /api/v1/agent/ask` answers a clinical question using only the indexed
knowledge base. It is built for medical staff: answers keep clinical terminology
and report dosing and contraindications as the entries state them.

Add `ANTHROPIC_API_KEY` to `src/.env` to enable it. Without a key the endpoint
returns `503 agent_unavailable` rather than failing at startup.

**Accuracy over fluency.** A graph node audits every draft answer against the
passages it was built from and rejects unsupported claims, so the assistant
abstains when the knowledge base does not cover a question instead of answering
from the model's own medical knowledge. An abstention is reported as
`"confidence": "abstained"`, and every clinical claim in a real answer carries a
`[n]` marker resolved in the `citations` array.

Session memory is keyed on `session_id`: send back the value from a previous
answer to ask a follow-up in the same conversation. Memory lasts for the process
lifetime and is not persisted.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | _(empty)_ | Enables the assistant |
| `AGENT_MODEL` | `claude-sonnet-5` | The answering model |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `VECTOR_STORE_DIR` | `var/vector_store` | Where the FAISS index is persisted |
| `DOCUMENTS_DIR` | `src/data` | PDFs indexed into the vector store on startup |
| `MAX_UPLOAD_BYTES` | `20971520` | Largest accepted PDF upload (20 MB) |
| `RETRIEVAL_TOP_K` | `8` | Candidates fetched per search |
| `RETRIEVAL_MIN_SCORE` | `0.35` | Cosine floor applied before grading |
| `SESSION_MAX_TURNS` | `20` | History window kept per session |
| `SESSION_TTL_SECONDS` | `3600` | Idle lifetime before a session is evicted |

## Tests

Run the full test suite with:

```powershell
uv run pytest
```

Each test that touches the vector store gets its own temporary FAISS directory,
so the suite never modifies `var/vector_store` or `src/data`. The tests cover
settings, DTO validation, PDF extraction and syncing, uploads, the RAG routes,
and global exception responses.

No test reaches the Claude API. The model adapter is stubbed throughout, and
embeddings are replaced with a small deterministic vector space so retrieval
behaviour can be asserted exactly. The agent tests cover the graph's control
flow rather than its prose: when it answers, when it retries a search, when it
rewrites an ungrounded answer, and when it refuses.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Application health check |
| `POST` | `/api/v1/agent/ask` | Answer a clinical question from the index |
| `GET` | `/api/v1/rag/documents` | List every document in the vector store |
| `POST` | `/api/v1/rag/documents` | Index plain text, searchable immediately |
| `POST` | `/api/v1/rag/documents/upload` | Upload a PDF (multipart field `file`); saved to `src/data` and indexed. A name already in use gets a `_2`, `_3`... suffix; nothing is replaced |
| `DELETE` | `/api/v1/rag/documents/{document_id}` | Remove a document from the index (and its PDF file, for PDFs) |
| `POST` | `/api/v1/rag/reindex` | Rebuild the index from the PDFs in `src/data` |
| `GET` | `/api/v1/rag/stats` | Report how much knowledge is searchable |

The `/api/v1/rag` write endpoints change what the assistant treats as ground
truth and are currently unauthenticated. Put them behind an owner-only guard
before running this anywhere but a development machine.

Example upload:

```powershell
curl.exe -F "file=@Asthma_Guideline.pdf" http://127.0.0.1:8000/api/v1/rag/documents/upload
```

## Structure

```text
src/
├── agent/         # LangGraph orchestration over Claude and the FAISS index
│   ├── agent.py     # answer_question(): the only entrypoint other layers use
│   ├── graph.py     # Supervisor graph: routes a request to one capability
│   ├── state.py     # Typed graph state and the history reducer
│   ├── llm/         # Claude adapter and the local embedding model
│   ├── memory/      # Session identity and the conversation checkpointer
│   ├── nodes/       # Node implementations for both graphs
│   ├── prompts/     # Byte-stable system prompts (kept stable for caching)
│   └── tools/       # Capability subgraphs and the routing registry
├── config/        # Settings read from the environment and .env
├── controllers/   # Functions that coordinate HTTP results
├── dto/           # Pydantic request and response schemas
├── exceptions/    # Domain errors and global exception handlers
├── middlewares/   # Request timing middleware
├── repositories/  # The FAISS vector store and its metadata sidecar
├── routes/        # FastAPI endpoint functions
├── data/          # PDF guidelines indexed into the vector store
├── seeders/       # Startup loading of src/data into the vector store
├── services/      # Business use-case functions
├── utils/         # Small general helpers
└── validations/   # Reusable domain validation rules
```

Each source folder has an `__init__.py`, making it an explicit Python package.
Python can also use namespace packages without these files, but explicit package
markers make imports, tooling, and project navigation more predictable for this
application. Do not add them to generated folders such as `__pycache__`.

## Validation and errors

Pydantic, included with FastAPI, is the right validation package for API request
and response schemas. `IndexDocumentRequest` handles field shapes and OpenAPI
documentation; the `validations/` functions handle reusable
domain rules such as trimming text and deduplicating tags.

Expected domain errors (such as an invalid or duplicate PDF), malformed requests, and
unexpected failures all pass through the global handlers in
`src/exceptions/handlers.py` and return a consistent JSON error body.
