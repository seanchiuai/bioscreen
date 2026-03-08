#!/usr/bin/env python3
"""Synchronous database builder that avoids asyncio/torch conflicts on Python 3.14."""

import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import requests
from tqdm import tqdm
from urllib.parse import urlencode

from app.pipeline.embedding import EmbeddingModel
from app.pipeline.sequence import validate_protein_sequence
from app.database.toxin_db import ToxinDatabase


def fetch_proteins(max_proteins: int = 500) -> list[dict]:
    """Fetch toxin proteins from UniProt REST API synchronously with pagination."""
    query = "(keyword:KW-0800) OR (keyword:KW-0872) OR (keyword:KW-0903) AND (reviewed:true)"
    page_size = 500
    offset = 0
    all_results = []

    print(f"Fetching up to {max_proteins} proteins from UniProt...")
    while len(all_results) < max_proteins:
        params = {
            "query": query,
            "format": "json",
            "size": min(page_size, max_proteins - len(all_results)),
            "offset": offset,
            "fields": "accession,protein_name,organism_name,length,sequence,go,ec,keyword",
        }
        url = f"https://rest.uniprot.org/uniprotkb/search?{urlencode(params)}"
        resp = requests.get(url)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        all_results.extend(results)
        offset += len(results)
        print(f"  Fetched {len(all_results)} entries so far...")
        import time; time.sleep(0.2)

    proteins = []
    for entry in all_results:
        uid = entry.get("primaryAccession", "")
        seq = entry.get("sequence", {}).get("value", "")
        if not uid or not seq:
            continue
        validation = validate_protein_sequence(seq)
        if not validation.valid:
            continue

        # Name
        name = ""
        pd = entry.get("proteinDescription", {})
        if "recommendedName" in pd:
            name = pd["recommendedName"].get("fullName", {}).get("value", "")
        elif "submissionNames" in pd and pd["submissionNames"]:
            name = pd["submissionNames"][0].get("fullName", {}).get("value", "")

        # Organism
        organism = entry.get("organism", {}).get("scientificName", "")

        # Toxin type from keywords
        toxin_type = "toxin"
        for kw in entry.get("keywords", []):
            kid = kw.get("id", "")
            if kid == "KW-0903":
                toxin_type = "neurotoxin"
            elif kid == "KW-0872":
                toxin_type = "ion_channel_toxin"

        proteins.append({
            "uniprot_id": uid,
            "name": name or f"Protein {uid}",
            "organism": organism,
            "sequence": validation.cleaned,
            "sequence_length": len(validation.cleaned),
            "toxin_type": toxin_type,
            "go_terms": [],
            "ec_numbers": [],
            "reviewed": True,
        })
        if len(proteins) >= max_proteins:
            break

    print(f"Got {len(proteins)} valid proteins")
    return proteins


def main():
    max_proteins = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Download
    proteins = fetch_proteins(max_proteins)
    if not proteins:
        print("No proteins fetched!")
        sys.exit(1)

    # Step 2: Load model & compute embeddings
    print("Loading ESM-2 model...")
    from app.config import get_settings
    settings = get_settings()
    model = EmbeddingModel(model_name=settings.esm2_model_name, device=settings.device)
    model.load()

    sequences = [p["sequence"] for p in proteins]
    print(f"Computing embeddings for {len(sequences)} sequences...")
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
        batch = sequences[i:i + batch_size]
        embs = model.embed_batch(batch, batch_size=len(batch))
        all_embeddings.extend(embs)
    embeddings = np.array(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 3: Build FAISS index
    print("Building FAISS index...")
    db = ToxinDatabase(
        index_path=output_dir / "toxin_db.faiss",
        meta_path=output_dir / "toxin_meta.json",
        embedding_dim=model.embedding_dim,
    )
    db.create_empty()
    db.add_proteins(embeddings=embeddings, metadata=proteins)
    db.save()

    # Validate
    val = db.validate_consistency()
    if val["valid"]:
        print(f"Database built successfully: {db.size} proteins")
    else:
        print(f"Validation failed: {val['issues']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
