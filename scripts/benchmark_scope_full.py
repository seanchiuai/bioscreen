#!/usr/bin/env python3
"""Benchmark BioScreen full path vs BLAST on SCOPe subset.

Runs embedding + ESMFold + Foldseek on Modal, then scores locally.
Uses a small subset (200 sequences) since each needs an ESMFold API call.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import modal

app = modal.App("bioscreen-benchmark-full")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "safetensors", "requests")
)


@app.function(image=image, gpu="A100", timeout=600)
def embed_and_predict_batch(
    sequences: list[str],
    api_key: str,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
) -> dict:
    """Embed on GPU + predict structures via NIM API."""
    import torch
    import requests as req
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().cuda()

    embeddings = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(sequences), 32):
            batch = sequences[i:i+32]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=1024)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.extend(pooled.cpu().float().numpy().tolist())

    # Structure predictions
    pdb_strings = []
    url = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for i, seq in enumerate(sequences):
        try:
            resp = req.post(url, headers=headers, json={"sequence": seq[:800]}, timeout=60)
            if resp.status_code == 200:
                pdb_strings.append(resp.json().get("pdbs", [None])[0])
            else:
                pdb_strings.append(None)
        except Exception:
            pdb_strings.append(None)
        if i % 20 == 0:
            print(f"  Predicted {i+1}/{len(sequences)} structures")
        import time as t
        t.sleep(0.2)

    return {"embeddings": embeddings, "pdb_strings": pdb_strings}


@app.local_entrypoint()
def main():
    import asyncio
    from app.database.toxin_db import ToxinDatabase
    from app.pipeline.scoring import compute_score
    from app.pipeline.similarity import FoldseekSearcher
    from app.pipeline.active_site import detect_pockets, compute_active_site_score
    from app.config import get_settings

    settings = get_settings()

    # Parse SCOPe — take 200 diverse sequences
    seqs, labels = [], []
    with open("/tmp/scope_40.fa") as f:
        seq, label = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: seqs.append(seq.upper()); labels.append(label)
                label = line.split()[1] if len(line.split()) > 1 else "?"
                seq = ""
            else: seq += line
        if seq: seqs.append(seq.upper()); labels.append(label)

    # Sample 200 from diverse families
    family_map = {}
    for s, l in zip(seqs, labels):
        fam = ".".join(l.split(".")[:3])
        if fam not in family_map: family_map[fam] = []
        family_map[fam].append((s, l))

    subset_seqs, subset_labels = [], []
    for fam, entries in sorted(family_map.items()):
        s, l = entries[0]
        subset_seqs.append(s)
        subset_labels.append(l)
        if len(subset_seqs) >= 200: break

    print(f"Subset: {len(subset_seqs)} sequences from {len(set(subset_labels))} families")

    # Run on Modal — embeddings + structure predictions
    print("Running on Modal A100 (embeddings + ESMFold)...")
    t0 = time.time()
    results = embed_and_predict_batch.remote(subset_seqs, settings.nvidia_api_key)
    modal_time = time.time() - t0
    print(f"Modal time: {modal_time:.1f}s")

    embeddings = np.array(results["embeddings"], dtype=np.float32)
    pdb_strings = results["pdb_strings"]
    has_pdb = sum(1 for p in pdb_strings if p)
    print(f"Structures predicted: {has_pdb}/{len(subset_seqs)}")

    # Load DB
    db = ToxinDatabase(index_path="data/toxin_db.faiss", meta_path="data/toxin_meta.json")
    db.load()
    fs = FoldseekSearcher()

    # Score each — full path
    print("Scoring with full path (embedding + Foldseek + active site)...")
    scores = []
    for i in range(len(subset_seqs)):
        emb = embeddings[i]
        try:
            dists, _ = db.search(emb, k=1)
            emb_sim = float(dists[0])
        except Exception:
            emb_sim = 0.0

        tm = None
        lddt = 0
        active = None
        pdb = pdb_strings[i]

        if pdb:
            hits = asyncio.get_event_loop().run_until_complete(fs.search(pdb, top_k=1))
            if hits:
                tm = hits[0].tm_score
                lddt = hits[0].lddt
                active = lddt  # Use lDDT as active site proxy

                # Pocket comparison
                pockets = detect_pockets(pdb)
                target_pdbs = {}
                for h in hits[:3]:
                    p = f"data/toxin_structures/{h.target_id}.pdb"
                    if os.path.exists(p):
                        target_pdbs[h.target_id] = open(p).read()
                if target_pdbs and pockets:
                    matches = compute_active_site_score(pdb, target_pdbs, top_k=1)
                    if matches:
                        active = 0.6 * lddt + 0.4 * matches[0].overlap_score

        score, _ = compute_score(
            embedding_sim=emb_sim, structural_sim=tm,
            function_overlap=0.0, active_site_overlap=active,
            sequence_length=len(subset_seqs[i]))
        scores.append(score)

        if i % 50 == 0:
            print(f"  Scored {i+1}/{len(subset_seqs)}")

    # BLAST
    print("Running BLAST...")
    with open("/tmp/scope_sub200.fasta", "w") as f:
        for i, seq in enumerate(subset_seqs):
            f.write(f">seq_{i}\n{seq}\n")

    subprocess.run(["makeblastdb", "-in", "/tmp/toxin_blast.fasta",
                    "-dbtype", "prot", "-out", "/tmp/toxin_blastdb"], capture_output=True)
    r = subprocess.run(
        ["blastp", "-query", "/tmp/scope_sub200.fasta", "-db", "/tmp/toxin_blastdb",
         "-outfmt", "6 qseqid sseqid pident evalue", "-max_target_seqs", "1", "-evalue", "10"],
        capture_output=True, text=True)

    blast_hits = {}
    for line in r.stdout.strip().split("\n"):
        if not line: continue
        parts = line.split("\t")
        idx = int(parts[0].split("_")[1])
        e = float(parts[3])
        if idx not in blast_hits or e < blast_hits[idx]:
            blast_hits[idx] = e

    blast_flagged = sum(1 for i in range(len(subset_seqs)) if blast_hits.get(i, 999) < 0.01)
    bio_flagged = sum(1 for s in scores if s >= 0.50)
    scores_arr = np.array(scores)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: SCOPe 2.08 — FULL PATH")
    print(f"{'='*60}")
    print(f"  Sequences:        {len(subset_seqs)}")
    print(f"  Structures:       {has_pdb}")
    print(f"  BLAST flagged:    {blast_flagged}/{len(subset_seqs)} ({blast_flagged/len(subset_seqs)*100:.1f}%)")
    print(f"  BioScreen flagged: {bio_flagged}/{len(subset_seqs)} ({bio_flagged/len(subset_seqs)*100:.1f}%)")
    print(f"  Modal time:       {modal_time:.1f}s")
    print(f"  Score distribution:")
    print(f"    LOW  (<0.50):  {(scores_arr < 0.50).sum()}")
    print(f"    MED  (0.50-0.74): {((scores_arr >= 0.50) & (scores_arr < 0.75)).sum()}")
    print(f"    HIGH (>=0.75): {(scores_arr >= 0.75).sum()}")


if __name__ == "__main__":
    main()
