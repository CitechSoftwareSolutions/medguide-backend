# Medical Knowledge Base API

A small FastAPI example using function-based routes, controllers, services, and
repositories. Pydantic classes are used only where structured data and automatic
validation are useful.

## Run it

```powershell
uv run fastapi dev main.py
```

Open <http://127.0.0.1:8000/docs> to try the generated OpenAPI documentation.
The application seeds two in-memory entries on startup; data resets when the
server stops.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Application health check |
| `GET` | `/api/v1/knowledge` | List entries; optionally use `?knowledge_type=condition` |
| `GET` | `/api/v1/knowledge/{entry_id}` | Retrieve one entry |
| `POST` | `/api/v1/knowledge` | Create an entry |

Example request:

```json
{
  "title": "Migraine",
  "summary": "A recurrent headache disorder that can cause throbbing pain.",
  "knowledge_type": "condition",
  "tags": ["neurology", "headache"]
}
```

## Structure

```text
src/
├── config/        # Runtime settings
├── controllers/   # Functions that coordinate HTTP results
├── dto/           # Pydantic request and response schemas
├── enums/         # Controlled values, such as knowledge types
├── exceptions/    # Domain errors and global exception handlers
├── middlewares/   # Request timing middleware
├── models/        # Pydantic domain models (an ORM model would live here later)
├── repositories/  # In-memory storage boundary
├── routes/        # FastAPI endpoint functions
├── seeders/       # Development sample data
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
and response schemas. `CreateKnowledgeEntryRequest` handles field shapes, enum
values, and OpenAPI documentation; the `validations/` functions handle reusable
domain rules such as trimming text and deduplicating tags.

Expected domain errors (not found and duplicate title), malformed requests, and
unexpected failures all pass through the global handlers in
`src/exceptions/handlers.py` and return a consistent JSON error body.
