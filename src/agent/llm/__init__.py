"""Model adapters: the Ollama chat client and the local embedding model."""

from .embeddings import (
    encode_documents,
    encode_query,
    get_embedding_model,
    warm_embedding_model,
)
from .ollama_client import complete_structured, complete_text, warm_chat_model

__all__ = [
    "complete_structured",
    "complete_text",
    "encode_documents",
    "encode_query",
    "get_embedding_model",
    "warm_chat_model",
    "warm_embedding_model",
]
