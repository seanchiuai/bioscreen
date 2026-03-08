"""FAISS-based toxin protein database with metadata storage.

Provides fast cosine similarity search over ESM-2 embeddings of known toxin proteins
with associated metadata from UniProt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from loguru import logger


class ToxinDatabase:
    """FAISS-indexed database of toxin protein embeddings with metadata.

    Stores ESM-2 embeddings for fast cosine similarity search and maintains
    parallel metadata for protein annotations (UniProt ID, name, organism, etc.).
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        meta_path: Union[str, Path],
        embedding_dim: int = 1280,  # ESM-2 650M embedding dimension
    ) -> None:
        """Initialize the toxin database.

        Args:
            index_path: Path to save/load the FAISS index file.
            meta_path: Path to save/load the metadata JSON file.
            embedding_dim: Dimension of protein embeddings (default for ESM-2 650M).
        """
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.embedding_dim = embedding_dim

        # FAISS index for similarity search
        self._index: Optional[faiss.Index] = None

        # Metadata storage
        self._metadata: List[Dict[str, Any]] = []

        # Track if database is loaded
        self._loaded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of proteins in the database."""
        return len(self._metadata) if self._loaded else 0

    @property
    def is_loaded(self) -> bool:
        """Whether the database has been loaded."""
        return self._loaded

    @property
    def embedding_dimension(self) -> int:
        """Dimension of the stored embeddings."""
        return self.embedding_dim

    # ─────────────────────────────────────────────────────────────────────────
    # Database management
    # ─────────────────────────────────────────────────────────────────────────

    def create_empty(self) -> None:
        """Create an empty FAISS index ready for adding embeddings."""
        # Use inner product index for cosine similarity
        # (embeddings will be L2-normalized before adding)
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._metadata = []
        self._loaded = True
        logger.info("Created empty toxin database with dimension {}", self.embedding_dim)

    def add_proteins(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> None:
        """Add protein embeddings and metadata to the database.

        Args:
            embeddings: Array of shape (n_proteins, embedding_dim).
            metadata: List of metadata dictionaries, one per protein.
            normalize: Whether to L2-normalize embeddings for cosine similarity.
        """
        if not self._loaded:
            self.create_empty()

        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) "
                f"doesn't match metadata count ({len(metadata)})"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) "
                f"doesn't match expected dimension ({self.embedding_dim})"
            )

        # Ensure embeddings are float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # L2-normalize for cosine similarity
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            embeddings = embeddings / norms

        # Add to FAISS index
        self._index.add(embeddings)

        # Add metadata
        self._metadata.extend(metadata)

        logger.info(
            "Added {} proteins to toxin database (total: {})",
            len(metadata),
            self.size,
        )

    def save(self) -> None:
        """Save the FAISS index and metadata to disk."""
        if not self._loaded:
            raise RuntimeError("Cannot save unloaded database")

        # Ensure parent directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))
        logger.info("Saved FAISS index to {}", self.index_path)

        # Save metadata as JSON
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        logger.info("Saved metadata to {}", self.meta_path)

    def load(self) -> None:
        """Load the FAISS index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")

        # Load FAISS index
        self._index = faiss.read_index(str(self.index_path))
        logger.info("Loaded FAISS index from {}", self.index_path)

        # Load metadata
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        logger.info("Loaded {} metadata entries from {}", len(self._metadata), self.meta_path)

        # Verify consistency
        index_size = self._index.ntotal
        meta_size = len(self._metadata)
        if index_size != meta_size:
            raise ValueError(
                f"Index size ({index_size}) doesn't match metadata size ({meta_size})"
            )

        self._loaded = True
        logger.info(
            "Toxin database loaded: {} proteins, embedding_dim={}",
            self.size,
            self.embedding_dim,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Search operations
    # ─────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        normalize_query: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for the k most similar proteins.

        Args:
            query_embedding: 1D array of shape (embedding_dim,).
            k: Number of nearest neighbors to return.
            normalize_query: Whether to L2-normalize the query for cosine similarity.

        Returns:
            Tuple of (distances, indices):
                - distances: Cosine similarities (or inner products).
                - indices: Database indices of the matches.
        """
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load() first.")

        if query_embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Query embedding shape {query_embedding.shape} "
                f"doesn't match expected ({self.embedding_dim},)"
            )

        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)

        if normalize_query:
            norm = np.linalg.norm(query)
            if norm > 1e-12:
                query = query / norm

        # Search
        k = min(k, self.size)  # Can't return more than database size
        distances, indices = self._index.search(query, k)

        return distances[0], indices[0]

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Get metadata for a protein by its database index.

        Args:
            index: Database index (from search results).

        Returns:
            Metadata dictionary for the protein.
        """
        if not self._loaded:
            raise RuntimeError("Database not loaded. Call load() first.")

        if not (0 <= index < self.size):
            raise IndexError(f"Index {index} out of range [0, {self.size})")

        return self._metadata[index]

    def get_metadata_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get metadata for multiple proteins.

        Args:
            indices: List of database indices.

        Returns:
            List of metadata dictionaries.
        """
        return [self.get_metadata(idx) for idx in indices]

    # ─────────────────────────────────────────────────────────────────────────
    # Database statistics and inspection
    # ─────────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics and summary information."""
        if not self._loaded:
            return {"loaded": False}

        stats = {
            "loaded": True,
            "total_proteins": self.size,
            "embedding_dimension": self.embedding_dim,
            "index_type": type(self._index).__name__,
            "storage_paths": {
                "index": str(self.index_path),
                "metadata": str(self.meta_path),
            },
        }

        # Analyze metadata if available
        if self._metadata:
            organisms = {}
            toxin_types = {}
            seq_lengths = []

            for meta in self._metadata:
                # Count organisms
                organism = meta.get("organism", "Unknown")
                organisms[organism] = organisms.get(organism, 0) + 1

                # Count toxin types
                toxin_type = meta.get("toxin_type", "Unknown")
                toxin_types[toxin_type] = toxin_types.get(toxin_type, 0) + 1

                # Collect sequence lengths
                seq_len = meta.get("sequence_length", 0)
                if seq_len > 0:
                    seq_lengths.append(seq_len)

            stats["metadata_summary"] = {
                "unique_organisms": len(organisms),
                "top_organisms": sorted(organisms.items(), key=lambda x: x[1], reverse=True)[:5],
                "unique_toxin_types": len(toxin_types),
                "top_toxin_types": sorted(toxin_types.items(), key=lambda x: x[1], reverse=True)[:5],
            }

            if seq_lengths:
                stats["sequence_length_stats"] = {
                    "mean": float(np.mean(seq_lengths)),
                    "median": float(np.median(seq_lengths)),
                    "min": int(np.min(seq_lengths)),
                    "max": int(np.max(seq_lengths)),
                    "std": float(np.std(seq_lengths)),
                }

        return stats

    def search_by_uniprot_id(self, uniprot_id: str) -> Optional[int]:
        """Find the database index for a protein by UniProt ID.

        Args:
            uniprot_id: UniProt accession to search for.

        Returns:
            Database index if found, None otherwise.
        """
        if not self._loaded:
            return None

        for i, meta in enumerate(self._metadata):
            if meta.get("uniprot_id") == uniprot_id:
                return i

        return None

    def get_random_sample(self, n: int = 5, seed: Optional[int] = None) -> List[Tuple[int, Dict[str, Any]]]:
        """Get a random sample of proteins for inspection.

        Args:
            n: Number of proteins to sample.
            seed: Random seed for reproducibility.

        Returns:
            List of (index, metadata) tuples.
        """
        if not self._loaded or self.size == 0:
            return []

        if seed is not None:
            np.random.seed(seed)

        n = min(n, self.size)
        indices = np.random.choice(self.size, size=n, replace=False)

        return [(int(idx), self._metadata[idx]) for idx in indices]

    # ─────────────────────────────────────────────────────────────────────────
    # Database maintenance
    # ─────────────────────────────────────────────────────────────────────────

    def validate_consistency(self) -> Dict[str, Any]:
        """Validate database consistency and identify potential issues.

        Returns:
            Dictionary with validation results and any issues found.
        """
        issues = []
        warnings = []

        if not self._loaded:
            issues.append("Database not loaded")
            return {"valid": False, "issues": issues, "warnings": warnings}

        # Check size consistency
        if self._index.ntotal != len(self._metadata):
            issues.append(
                f"Size mismatch: index has {self._index.ntotal} entries, "
                f"metadata has {len(self._metadata)} entries"
            )

        # Check metadata completeness
        required_fields = ["uniprot_id", "name", "sequence_length"]
        for i, meta in enumerate(self._metadata):
            missing = [field for field in required_fields if field not in meta]
            if missing:
                warnings.append(f"Entry {i} missing fields: {missing}")

        # Check for duplicate UniProt IDs
        uniprot_ids = [meta.get("uniprot_id") for meta in self._metadata if meta.get("uniprot_id")]
        if len(uniprot_ids) != len(set(uniprot_ids)):
            warnings.append("Duplicate UniProt IDs found in metadata")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_entries": self.size,
        }