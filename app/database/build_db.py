"""Database building module for protein embeddings from UniProt.

This module queries the UniProt REST API, downloads protein sequences,
computes ESM-2 embeddings, and builds a FAISS index for similarity search.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
import numpy as np
from Bio import SeqIO
from loguru import logger
from tqdm import tqdm

from app.config import get_settings
from app.database.toxin_db import ToxinDatabase
from app.pipeline.embedding import EmbeddingModel
from app.pipeline.sequence import validate_protein_sequence


class UniProtClient:
    """Client for UniProt REST API interactions."""

    BASE_URL = "https://rest.uniprot.org"
    SEARCH_ENDPOINT = "/uniprotkb/search"

    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_proteins(
        self,
        query: str = "(keyword:KW-0800) OR (keyword:KW-0872) OR (keyword:KW-0903)",
        max_records: int = 5000,
        include_isoforms: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search UniProt for protein entries.

        Args:
            query: UniProt search query (default searches for toxins)
            max_records: Maximum number of records to fetch
            include_isoforms: Whether to include protein isoforms

        Returns:
            List of protein metadata dictionaries
        """
        if not self.session:
            raise RuntimeError("UniProtClient must be used as async context manager")

        # Parameters for UniProt API
        params = {
            "query": query,
            "format": "json",
            "size": min(self.batch_size, max_records),
            "fields": "accession,protein_name,organism_name,length,sequence,go,ec,keyword,reviewed"
        }

        if not include_isoforms:
            params["query"] += " AND (reviewed:true)"

        all_proteins: List[Dict[str, Any]] = []
        offset = 0

        logger.info(f"Starting UniProt search with query: {query}")

        while len(all_proteins) < max_records:
            current_params = params.copy()
            current_params["offset"] = offset
            current_params["size"] = min(self.batch_size, max_records - len(all_proteins))

            url = f"{self.BASE_URL}{self.SEARCH_ENDPOINT}?" + urlencode(current_params)

            logger.debug(f"Fetching batch from offset {offset}")

            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"UniProt API error: {response.status}")
                        break

                    data = await response.json()
                    results = data.get("results", [])

                    if not results:
                        logger.info("No more results from UniProt")
                        break

                    # Process and validate results
                    for entry in results:
                        try:
                            protein_data = self._process_uniprot_entry(entry)
                            if protein_data:
                                all_proteins.append(protein_data)
                        except Exception as e:
                            logger.warning(f"Error processing entry {entry.get('primaryAccession', 'unknown')}: {e}")
                            continue

                    offset += len(results)
                    logger.info(f"Fetched {len(all_proteins)} proteins so far")

                    # Small delay to be respectful to UniProt servers
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                break

        logger.info(f"Collected {len(all_proteins)} proteins from UniProt")
        return all_proteins[:max_records]

    def _process_uniprot_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single UniProt entry into our metadata format.

        Args:
            entry: Raw UniProt JSON entry

        Returns:
            Processed metadata dictionary or None if invalid
        """
        try:
            # Extract basic fields
            uniprot_id = entry.get("primaryAccession", "")
            if not uniprot_id:
                return None

            # Get protein name
            protein_name = ""
            if "proteinDescription" in entry:
                name_data = entry["proteinDescription"]
                if "recommendedName" in name_data:
                    protein_name = name_data["recommendedName"].get("fullName", {}).get("value", "")
                elif "submissionNames" in name_data and name_data["submissionNames"]:
                    protein_name = name_data["submissionNames"][0].get("fullName", {}).get("value", "")

            # Get organism
            organism = ""
            if "organism" in entry:
                organism = entry["organism"].get("scientificName", "")

            # Get sequence
            sequence = entry.get("sequence", {}).get("value", "")
            if not sequence:
                logger.debug(f"No sequence for {uniprot_id}")
                return None

            # Validate sequence
            validation = validate_protein_sequence(sequence)
            if not validation.valid:
                logger.debug(f"Invalid sequence for {uniprot_id}: {validation.message}")
                return None

            # Extract GO terms
            go_terms = []
            for annotation in entry.get("dbReferences", []):
                if annotation.get("type") == "GO":
                    go_id = annotation.get("id", "")
                    properties = annotation.get("properties", [])
                    term_name = ""
                    for prop in properties:
                        if prop.get("key") == "term":
                            term_name = prop.get("value", "")
                            break
                    if go_id and term_name:
                        go_terms.append(f"{go_id}:{term_name}")

            # Extract EC numbers
            ec_numbers = []
            for annotation in entry.get("dbReferences", []):
                if annotation.get("type") == "EC":
                    ec_numbers.append(annotation.get("id", ""))

            # Determine toxin type from keywords
            toxin_type = "unknown"
            for keyword in entry.get("keywords", []):
                keyword_id = keyword.get("id", "")
                if keyword_id == "KW-0800":  # Toxin
                    toxin_type = "toxin"
                elif keyword_id == "KW-0872":  # Ion channel impairing toxin
                    toxin_type = "ion_channel_toxin"
                elif keyword_id == "KW-0903":  # Neurotoxin
                    toxin_type = "neurotoxin"
                # Add more specific toxin types as needed

            return {
                "uniprot_id": uniprot_id,
                "name": protein_name or f"Protein {uniprot_id}",
                "organism": organism,
                "sequence": validation.cleaned,
                "sequence_length": len(validation.cleaned),
                "toxin_type": toxin_type,
                "go_terms": go_terms,
                "ec_numbers": ec_numbers,
                "reviewed": entry.get("entryAudit", {}).get("entryStatus") == "reviewed",
            }

        except Exception as e:
            logger.warning(f"Error processing UniProt entry: {e}")
            return None


async def download_uniprot_proteins(
    query: str = "(keyword:KW-0800) OR (keyword:KW-0872) OR (keyword:KW-0903)",
    max_proteins: int = 5000,
) -> List[Dict[str, Any]]:
    """Download protein data from UniProt.

    Args:
        query: UniProt search query
        max_proteins: Maximum number of proteins to download

    Returns:
        List of protein metadata dictionaries
    """
    settings = get_settings()

    async with UniProtClient(batch_size=settings.uniprot_batch_size) as client:
        proteins = await client.search_proteins(
            query=query,
            max_records=max_proteins,
        )

    return proteins


def compute_embeddings_batch(
    sequences: List[str],
    embedding_model: EmbeddingModel,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute ESM-2 embeddings for a list of sequences.

    Args:
        sequences: List of protein sequences
        embedding_model: Loaded ESM-2 embedding model
        batch_size: Batch size for embedding computation

    Returns:
        Array of shape (n_sequences, embedding_dim)
    """
    logger.info(f"Computing embeddings for {len(sequences)} sequences")

    # Compute embeddings in batches with progress bar
    all_embeddings = []

    with tqdm(total=len(sequences), desc="Computing embeddings") as pbar:
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_embeddings = embedding_model.embed_batch(batch, batch_size=len(batch))
            all_embeddings.extend(batch_embeddings)
            pbar.update(len(batch))

    return np.array(all_embeddings)


async def build_database(
    output_dir: Path = Path("data"),
    max_proteins: int = 5000,
    query: str = "(keyword:KW-0800) OR (keyword:KW-0872) OR (keyword:KW-0903)",
    embedding_batch_size: int = 8,
) -> ToxinDatabase:
    """Build protein database with embeddings and FAISS index.

    Args:
        output_dir: Directory to save database files
        max_proteins: Maximum number of proteins to include
        query: UniProt search query
        embedding_batch_size: Batch size for embedding computation

    Returns:
        Built ToxinDatabase instance
    """
    settings = get_settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths — must match config defaults so the app finds them at startup
    index_path = output_dir / "toxin_db.faiss"
    metadata_path = output_dir / "toxin_meta.json"

    logger.info(f"Building protein database in {output_dir}")
    logger.info(f"Target: {max_proteins} proteins")
    logger.info(f"Query: {query}")

    # Step 1: Download proteins from UniProt
    logger.info("Step 1: Downloading proteins from UniProt...")
    proteins = await download_uniprot_proteins(
        query=query,
        max_proteins=max_proteins,
    )

    if not proteins:
        raise ValueError("No proteins downloaded from UniProt")

    logger.info(f"Downloaded {len(proteins)} proteins")

    # Step 2: Load ESM-2 model and compute embeddings
    logger.info("Step 2: Loading ESM-2 model...")
    embedding_model = EmbeddingModel(
        model_name=settings.esm2_model_name,
        device=settings.device,
    )
    embedding_model.load()

    # Extract sequences
    sequences = [protein["sequence"] for protein in proteins]

    # Compute embeddings
    logger.info("Step 3: Computing ESM-2 embeddings...")
    embeddings = compute_embeddings_batch(
        sequences=sequences,
        embedding_model=embedding_model,
        batch_size=embedding_batch_size,
    )

    logger.info(f"Computed embeddings shape: {embeddings.shape}")

    # Step 3: Build FAISS index
    logger.info("Step 4: Building FAISS index...")
    db = ToxinDatabase(
        index_path=index_path,
        meta_path=metadata_path,
        embedding_dim=embedding_model.embedding_dim,
    )

    db.create_empty()
    db.add_proteins(embeddings=embeddings, metadata=proteins)

    # Step 4: Save database
    logger.info("Step 5: Saving database...")
    db.save()

    # Validate database
    validation = db.validate_consistency()
    if not validation["valid"]:
        logger.error(f"Database validation failed: {validation['issues']}")
        raise RuntimeError("Database validation failed")

    logger.info(f"Database built successfully:")
    logger.info(f"  - {db.size} proteins")
    logger.info(f"  - Index: {index_path}")
    logger.info(f"  - Metadata: {metadata_path}")

    # Print some statistics
    stats = db.get_statistics()
    if "metadata_summary" in stats:
        logger.info(f"  - Organisms: {stats['metadata_summary']['unique_organisms']}")
        logger.info(f"  - Toxin types: {stats['metadata_summary']['unique_toxin_types']}")

    return db


def save_fasta(proteins: List[Dict[str, Any]], fasta_path: Path) -> None:
    """Save protein sequences to FASTA file.

    Args:
        proteins: List of protein metadata dictionaries
        fasta_path: Output FASTA file path
    """
    logger.info(f"Saving {len(proteins)} sequences to {fasta_path}")

    with open(fasta_path, "w") as f:
        for protein in proteins:
            uniprot_id = protein["uniprot_id"]
            name = protein["name"]
            organism = protein["organism"]
            sequence = protein["sequence"]

            header = f">{uniprot_id}|{name}|{organism}"
            f.write(f"{header}\n")
            f.write(f"{sequence}\n")

    logger.info(f"FASTA file saved: {fasta_path}")


if __name__ == "__main__":
    import asyncio

    # Simple test
    async def main():
        db = await build_database(max_proteins=10)
        print(f"Test database built with {db.size} proteins")

    asyncio.run(main())