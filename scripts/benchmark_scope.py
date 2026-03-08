#!/usr/bin/env python3
"""Benchmark BioScreen vs BLAST on SCOPe 2.08 (40% identity) dataset.

Runs embedding-based screening on Modal (parallelized) and compares
detection rates against BLAST for remote homology detection.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import modal

app = modal.App("bioscreen-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "numpy", "safetensors")
)


@app.function(image=image, gpu="A100", timeout=600)
def embed_batch_gpu(
    sequences: list[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 64,
) -> list[list[float]]:
    """Compute ESM-2 embeddings on GPU."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).eval().cuda()

    all_embeddings = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embeddings.extend(pooled.cpu().float().numpy().tolist())
    return all_embeddings


@app.local_entrypoint()
def main():
    from app.database.toxin_db import ToxinDatabase

    SCOPE_FASTA = "/tmp/scope_40.fa"
    if not os.path.exists(SCOPE_FASTA):
        print("Download SCOPe first")
        sys.exit(1)

    # Parse SCOPe FASTA
    print("Parsing SCOPe dataset...")
    sequences = []
    labels = []  # SCOP family IDs
    current_seq = ""
    current_label = ""

    with open(SCOPE_FASTA) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq and current_label:
                    sequences.append(current_seq.upper())
                    labels.append(current_label)
                # Parse SCOP family from header: e.g. "a.1.1.1"
                parts = line.split()
                current_label = parts[1] if len(parts) > 1 else "unknown"
                current_seq = ""
            else:
                current_seq += line
        if current_seq and current_label:
            sequences.append(current_seq.upper())
            labels.append(current_label)

    print(f"Loaded {len(sequences)} sequences, {len(set(labels))} unique families")

    # Subsample for speed: take 2000 sequences from diverse families
    family_to_seqs = {}
    for seq, label in zip(sequences, labels):
        family = ".".join(label.split(".")[:3])  # superfamily level
        if family not in family_to_seqs:
            family_to_seqs[family] = []
        family_to_seqs[family].append((seq, label))

    # Pick up to 3 per superfamily, max 2000 total
    subset_seqs = []
    subset_labels = []
    for family, entries in sorted(family_to_seqs.items()):
        for seq, label in entries[:3]:
            subset_seqs.append(seq)
            subset_labels.append(label)
            if len(subset_seqs) >= 2000:
                break
        if len(subset_seqs) >= 2000:
            break

    print(f"Subset: {len(subset_seqs)} sequences from {len(set(subset_labels))} families")

    # Step 1: Embed all sequences on Modal GPU
    print("\nStep 1: Computing embeddings on Modal A100...")
    t0 = time.time()
    embeddings_list = embed_batch_gpu.remote(subset_seqs)
    embed_time = time.time() - t0
    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"Embeddings: {embeddings.shape} in {embed_time:.1f}s")

    # Step 2: Load toxin DB and search
    print("\nStep 2: Searching against toxin DB...")
    db = ToxinDatabase(index_path="data/toxin_db.faiss", meta_path="data/toxin_meta.json")
    db.load()

    # For each sequence, get max similarity to toxin DB
    from app.pipeline.scoring import compute_score

    bioscreen_scores = []
    for i, emb in enumerate(embeddings):
        try:
            distances, indices = db.search(emb, k=1)
            sim = float(distances[0])
            score, _ = compute_score(
                embedding_sim=sim,
                structural_sim=None,
                function_overlap=0.0,
                sequence_length=len(subset_seqs[i]),
            )
            bioscreen_scores.append(score)
        except Exception:
            bioscreen_scores.append(0.0)

    # Step 3: Run BLAST
    print("\nStep 3: Running BLAST on subset...")
    # Build BLAST DB from toxins
    with open("/tmp/toxin_blast.fasta", "w") as f:
        with open("data/toxin_meta.json") as mf:
            for m in json.load(mf):
                f.write(f">{m['uniprot_id']}\n{m['sequence']}\n")
    subprocess.run(
        ["makeblastdb", "-in", "/tmp/toxin_blast.fasta", "-dbtype", "prot",
         "-out", "/tmp/toxin_blastdb"],
        capture_output=True,
    )

    # Write subset as FASTA
    with open("/tmp/scope_subset.fasta", "w") as f:
        for i, seq in enumerate(subset_seqs):
            f.write(f">seq_{i}\n{seq}\n")

    # Run BLAST
    t0 = time.time()
    result = subprocess.run(
        ["blastp", "-query", "/tmp/scope_subset.fasta", "-db", "/tmp/toxin_blastdb",
         "-outfmt", "6 qseqid sseqid pident evalue", "-max_target_seqs", "1",
         "-evalue", "10", "-num_threads", "4"],
        capture_output=True, text=True,
    )
    blast_time = time.time() - t0

    # Parse BLAST results
    blast_hits = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        qid = parts[0]
        evalue = float(parts[3])
        idx = int(qid.split("_")[1])
        if idx not in blast_hits or evalue < blast_hits[idx]:
            blast_hits[idx] = evalue

    blast_flagged = sum(1 for i in range(len(subset_seqs)) if blast_hits.get(i, 999) < 0.01)
    bioscreen_flagged = sum(1 for s in bioscreen_scores if s >= 0.45)

    # Analyze by family — which families does BioScreen catch that BLAST misses?
    families_blast_misses = set()
    families_bioscreen_catches = set()
    bioscreen_only = 0

    for i in range(len(subset_seqs)):
        blast_hit = blast_hits.get(i, 999) < 0.01
        bioscreen_hit = bioscreen_scores[i] >= 0.45

        family = ".".join(subset_labels[i].split(".")[:3])

        if not blast_hit:
            families_blast_misses.add(family)
        if bioscreen_hit:
            families_bioscreen_catches.add(family)
        if bioscreen_hit and not blast_hit:
            bioscreen_only += 1

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS: SCOPe 2.08 (40% identity)")
    print(f"{'='*60}")
    print(f"  Sequences tested:     {len(subset_seqs)}")
    print(f"  Unique families:      {len(set(subset_labels))}")
    print(f"")
    print(f"  BLAST flagged:        {blast_flagged}/{len(subset_seqs)} ({blast_flagged/len(subset_seqs)*100:.1f}%)")
    print(f"  BioScreen flagged:    {bioscreen_flagged}/{len(subset_seqs)} ({bioscreen_flagged/len(subset_seqs)*100:.1f}%)")
    print(f"  BioScreen-only:       {bioscreen_only} (caught by BioScreen but not BLAST)")
    print(f"")
    print(f"  Embedding time:       {embed_time:.1f}s (Modal A100)")
    print(f"  BLAST time:           {blast_time:.1f}s")
    print(f"")

    # Score distribution
    scores_arr = np.array(bioscreen_scores)
    print(f"  BioScreen score distribution:")
    print(f"    LOW  (<0.45):  {(scores_arr < 0.45).sum()}")
    print(f"    MED  (0.45-0.74): {((scores_arr >= 0.45) & (scores_arr < 0.75)).sum()}")
    print(f"    HIGH (>=0.75): {(scores_arr >= 0.75).sum()}")

    # Save results
    results = {
        "benchmark": "SCOPe 2.08 (40% identity)",
        "total_sequences": len(subset_seqs),
        "unique_families": len(set(subset_labels)),
        "blast_flagged": blast_flagged,
        "bioscreen_flagged": bioscreen_flagged,
        "bioscreen_only": bioscreen_only,
        "embed_time_s": embed_time,
        "blast_time_s": blast_time,
    }
    with open("data/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to data/benchmark_results.json")


if __name__ == "__main__":
    main()
