"""Local sentence-embedding model used for indexing and retrieval."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from src.shared.config import get_settings
from src.shared.errors import AgentUnavailableError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Return the one process-wide embedding model.

    Imported lazily so the heavy Torch import is paid at first use rather than
    at application import time.

    A model that cannot load is reported as ``AgentUnavailableError`` so the
    API answers 503 instead of a bare 500. That covers a missing download as
    well as a native library the OS refuses to load (Windows Smart App Control
    blocks some unsigned scikit-learn and Torch DLLs). ``lru_cache`` does not
    cache exceptions, so the load is retried on the next request.
    """
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", settings.embedding_model_name)
        return SentenceTransformer(settings.embedding_model_name)
    except Exception as exc:
        logger.exception("Could not load embedding model %s", settings.embedding_model_name)
        raise AgentUnavailableError(
            "The assistant's search model could not be loaded on the server."
        ) from exc


def warm_embedding_model() -> None:
    """Load and exercise the model so the first request pays no load cost."""
    encode_query("warmup")


def _encode(texts: list[str]) -> np.ndarray:
    """Encode text into L2-normalized float32 vectors.

    Normalizing here is what makes FAISS inner-product search equal to exact
    cosine similarity, so every caller must go through this function.
    """
    vectors = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_documents(texts: list[str]) -> np.ndarray:
    """Encode knowledge chunks for indexing."""
    if not texts:
        return np.empty((0, get_settings().embedding_dimensions), dtype=np.float32)
    return _encode(texts)


def encode_query(text: str) -> np.ndarray:
    """Encode one search query into a single vector."""
    return _encode([text])[0]
