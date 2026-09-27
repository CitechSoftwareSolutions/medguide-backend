"""Load the PDF guidelines shipped in ``src/data`` into the vector store.

To add knowledge, drop a PDF into that directory and restart the API (or call
``POST /api/v1/rag/reindex``). Unchanged files are not re-embedded.
"""

from src.services.document_service import sync_documents_directory


def seed_vector_store_from_documents() -> dict[str, int]:
    """Index new or changed PDFs and drop the chunks of removed ones."""
    return sync_documents_directory()
