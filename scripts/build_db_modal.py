#!/usr/bin/env python3
"""Build toxin DB using Modal for GPU-accelerated ESM-2 embeddings.

Usage:
    modal run scripts/build_db_modal.py
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import modal

# ---------------------------------------------------------------------------
# Modal image with ESM-2 dependencies
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "safetensors")
)

app = modal.App("bioscreen-embedding")


# ---------------------------------------------------------------------------
# Remote GPU function: compute ESM-2 embeddings
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",  # Max speed
    timeout=300,
)
def compute_embeddings_gpu(
    sequences: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 64,
) -> list[list[float]]:
    """Compute ESM-2 embeddings on GPU. Returns list of lists (JSON-serializable)."""
    import torch
    import numpy as np
    from transformers import AutoTokenizer, EsmModel

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().cuda()
    print(f"Model loaded on GPU. Processing {len(sequences)} sequences...")

    all_embeddings = []
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embeddings.extend(mean_pool.cpu().float().numpy().tolist())

            done = min(i + batch_size, len(sequences))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{len(sequences)} ({rate:.1f} seq/s)")

    print(f"Done in {time.time() - t0:.1f}s")
    return all_embeddings


# ---------------------------------------------------------------------------
# Local: fetch proteins, call GPU, build FAISS index
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    import numpy as np
    import requests
    from urllib.parse import urlencode
    from app.pipeline.sequence import validate_protein_sequence
    from app.database.toxin_db import ToxinDatabase

    MAX_PROTEINS = 2000
    OUTPUT_DIR = Path("data")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Fetch proteins from UniProt ---
    print(f"Fetching up to {MAX_PROTEINS} proteins from UniProt...")
    query = "(keyword:KW-0800) OR (keyword:KW-0872) OR (keyword:KW-0903) AND (reviewed:true)"
    page_size = 500
    all_results = []
    offset = 0

    while len(all_results) < MAX_PROTEINS:
        params = {
            "query": query,
            "format": "json",
            "size": min(page_size, MAX_PROTEINS - len(all_results)),
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
        print(f"  {len(all_results)} entries fetched...")
        time.sleep(0.1)

    # Process entries
    proteins = []
    for entry in all_results:
        uid = entry.get("primaryAccession", "")
        seq = entry.get("sequence", {}).get("value", "")
        if not uid or not seq:
            continue
        validation = validate_protein_sequence(seq)
        if not validation.valid:
            continue

        name = ""
        pd = entry.get("proteinDescription", {})
        if "recommendedName" in pd:
            name = pd["recommendedName"].get("fullName", {}).get("value", "")
        elif "submissionNames" in pd and pd["submissionNames"]:
            name = pd["submissionNames"][0].get("fullName", {}).get("value", "")

        organism = entry.get("organism", {}).get("scientificName", "")
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
        if len(proteins) >= MAX_PROTEINS:
            break

    print(f"Got {len(proteins)} valid proteins")
    sequences = [p["sequence"] for p in proteins]

    # --- Step 2: Compute embeddings on GPU via Modal ---
    print("Sending to Modal GPU for embedding computation...")
    t0 = time.time()
    embeddings_list = compute_embeddings_gpu.remote(sequences)
    gpu_time = time.time() - t0
    print(f"GPU embeddings returned in {gpu_time:.1f}s")

    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # --- Step 3: Build FAISS index locally ---
    print("Building FAISS index...")
    db = ToxinDatabase(
        index_path=OUTPUT_DIR / "toxin_db.faiss",
        meta_path=OUTPUT_DIR / "toxin_meta.json",
        embedding_dim=embeddings.shape[1],
    )
    db.create_empty()
    db.add_proteins(embeddings=embeddings, metadata=proteins)
    db.save()

    val = db.validate_consistency()
    if val["valid"]:
        print(f"Database built successfully: {db.size} proteins, {embeddings.shape[1]}-dim embeddings")
    else:
        print(f"Validation failed: {val['issues']}")
        sys.exit(1)
