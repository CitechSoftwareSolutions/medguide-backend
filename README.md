# Medical Knowledge Base API

A small FastAPI example using function-based routes, controllers, services, and
repositories. PostgreSQL persistence uses SQLAlchemy 2.x and the Psycopg 3
driver; Pydantic classes validate API data.

## Run it

Install the project and development test tools:

```powershell
uv sync --group dev
```

Apply the current database schema to Neon:

```powershell
uv run alembic upgrade head
```

Start the development server:

```powershell
uv run fastapi dev main.py
```

Open <http://127.0.0.1:8000/docs> to try the generated OpenAPI documentation.
The application seeds two entries on startup if `knowledge_entries` is empty.

## Database and migrations

`src/.env` contains the ignored `DATABASE_URL` used by the application. A Neon
URL beginning with `postgresql://` is converted internally to the SQLAlchemy
Psycopg URL format while preserving its SSL settings.

The initial migration is
`migrations/versions/20260904_0001_create_knowledge_entries.py`. It creates:

- the PostgreSQL `knowledge_type` enum;
- the `knowledge_entries` table; and
- a unique title constraint.

Create later migrations after changing ORM models with:

```powershell
uv run alembic revision --autogenerate -m "describe the change"
uv run alembic upgrade head
```

## Tests

Run the full test suite with:

```powershell
uv run pytest
```

The tests use fakes and mocks for repository transactions, so they do not write
to the configured Neon database. They cover settings, DTO validation, service
rules, repository transaction handling, controller/route behavior, and global
exception responses. Database schema changes are validated separately through
Alembic migrations.

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
├── config/        # Settings and SQLAlchemy engine/session factory
├── controllers/   # Functions that coordinate HTTP results
├── dto/           # Pydantic request and response schemas
├── enums/         # Controlled values, such as knowledge types
├── exceptions/    # Domain errors and global exception handlers
├── middlewares/   # Request timing middleware
├── models/        # SQLAlchemy ORM models
├── repositories/  # SQLAlchemy persistence operations and transactions
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
