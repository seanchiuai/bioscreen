"""ESM-2 sequence embeddings with mean pooling and cosine similarity.

Uses the ``facebook/esm2_t33_650M_UR50D`` model (or any ESM-2 variant) via
HuggingFace ``transformers`` to produce fixed-size protein representations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, EsmModel

from app.config import get_settings

# Type alias for embedding arrays
EmbeddingArray = np.ndarray  # shape (embedding_dim,)


# ── Model wrapper ─────────────────────────────────────────────────────────────


class EmbeddingModel:
    """Wraps an ESM-2 model for batch protein embedding.

    Args:
        model_name: HuggingFace model ID (default: ``facebook/esm2_t33_650M_UR50D``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        max_length: Maximum tokeniser length; sequences are truncated silently.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cpu",
        max_length: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model: EsmModel | None = None
        self._tokenizer = None

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download and load the model and tokeniser from HuggingFace Hub."""
        logger.info("Loading tokeniser for {}", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info("Loading ESM-2 model {}", self.model_name)
        self._model = EsmModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        logger.info(
            "ESM-2 ready – hidden_size={}, device={}",
            self._model.config.hidden_size,
            self.device,
        )

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        if self._model is None:
            raise RuntimeError("Model not loaded – call load() first.")
        return self._model.config.hidden_size

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Core embedding methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, sequence: str) -> EmbeddingArray:
        """Compute a single mean-pooled embedding for one sequence.

        Args:
            sequence: Single-letter amino acid sequence.

        Returns:
            1-D numpy array of shape ``(embedding_dim,)``.
        """
        return self.embed_batch([sequence])[0]

    @torch.no_grad()
    def embed_batch(
        self,
        sequences: list[str],
        batch_size: int = 8,
    ) -> list[EmbeddingArray]:
        """Compute mean-pooled embeddings for a list of sequences.

        Long sequences are truncated to ``self.max_length`` tokens.

        Args:
            sequences: List of amino acid sequences.
            batch_size: Number of sequences to tokenise simultaneously.

        Returns:
            List of 1-D numpy arrays, one per input sequence.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded – call load() first.")

        all_embeddings: list[EmbeddingArray] = []

        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            # outputs.last_hidden_state: (batch, seq_len, hidden)
            hidden = outputs.last_hidden_state  # (B, L, H)
            attention_mask = inputs["attention_mask"]  # (B, L)

            # Mean-pool over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            sum_hidden = (hidden * mask_expanded).sum(dim=1)  # (B, H)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)   # (B, 1)
            mean_pool = (sum_hidden / count).cpu().numpy()     # (B, H)

            all_embeddings.extend(mean_pool)

        return all_embeddings

    # ------------------------------------------------------------------
    # Utility: save / load pre-computed embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: Union[str, Path]) -> None:
        """Save an (N, D) embedding matrix to a ``.npy`` file."""
        np.save(str(path), embeddings)

    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> np.ndarray:
        """Load an (N, D) embedding matrix from a ``.npy`` file."""
        return np.load(str(path))


# ── Similarity helpers ────────────────────────────────────────────────────────


def cosine_similarity(a: EmbeddingArray, b: EmbeddingArray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: Embedding vector.
        b: Embedding vector.

    Returns:
        Scalar in ``[-1, 1]``.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(
    queries: np.ndarray,
    keys: np.ndarray,
) -> np.ndarray:
    """Compute an (N, M) cosine similarity matrix.

    Args:
        queries: (N, D) query embedding matrix.
        keys: (M, D) key embedding matrix.

    Returns:
        (N, M) float32 array of pairwise cosine similarities.
    """
    # L2-normalise rows
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
    k_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9)
    return q_norm @ k_norm.T


# ── Module-level singleton ────────────────────────────────────────────────────

_model_instance: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    """Return the module-level :class:`EmbeddingModel` singleton.

    The model is loaded lazily on first call using settings from
    :func:`app.config.get_settings`.
    """
    global _model_instance
    if _model_instance is None:
        settings = get_settings()
        _model_instance = EmbeddingModel(
            model_name=settings.esm2_model_name,
            device=settings.device,
        )
        _model_instance.load()
    return _model_instance
