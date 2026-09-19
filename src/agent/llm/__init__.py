"""Model adapters: the Claude API client and the local embedding model."""

from .claude_client import complete_structured, complete_text, get_client
from .embeddings import (
    encode_documents,
    encode_query,
    get_embedding_model,
    warm_embedding_model,
)

__all__ = [
    "complete_structured",
    "complete_text",
    "encode_documents",
    "encode_query",
    "get_client",
    "get_embedding_model",
    "warm_embedding_model",
]
