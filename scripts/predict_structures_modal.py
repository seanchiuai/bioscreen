#!/usr/bin/env python3
"""Predict structures for toxin reference set using ESMFold NIM API in parallel on Modal.

Parallelizes NIM API calls across multiple Modal containers for speed.
Then builds a toxin-specific Foldseek database locally.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import modal

app = modal.App("bioscreen-structures")

image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")


@app.function(image=image, max_containers=20, timeout=120)
def predict_one(sequence: str, uniprot_id: str, api_key: str) -> tuple[str, str | None]:
    """Predict structure for one sequence via NIM API. Returns (uniprot_id, pdb_string)."""
    import requests

    url = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    seq = sequence[:800]  # ESMFold limit
    try:
        resp = requests.post(url, headers=headers, json={"sequence": seq}, timeout=60)
        if resp.status_code == 200:
            pdb = resp.json().get("pdbs", [None])[0]
            return (uniprot_id, pdb)
        elif resp.status_code == 429:
            # Rate limited — wait and retry once
            time.sleep(5)
            resp = requests.post(url, headers=headers, json={"sequence": seq}, timeout=60)
            if resp.status_code == 200:
                return (uniprot_id, resp.json().get("pdbs", [None])[0])
        return (uniprot_id, None)
    except Exception as e:
        print(f"Error for {uniprot_id}: {e}")
        return (uniprot_id, None)


@app.local_entrypoint()
def main():
    import subprocess

    structures_dir = Path("data/toxin_structures")
    structures_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open("data/toxin_meta.json") as f:
        metadata = json.load(f)

    # Get API key
    from app.config import get_settings
    settings = get_settings()
    api_key = settings.nvidia_api_key
    if not api_key:
        print("No NVIDIA_API_KEY set!")
        sys.exit(1)

    # Find which IDs still need structures
    existing = {p.stem for p in structures_dir.glob("*.pdb") if p.stat().st_size > 100}
    to_predict = [(m["sequence"], m["uniprot_id"]) for m in metadata if m["uniprot_id"] not in existing]
    print(f"Already have: {len(existing)} structures")
    print(f"Need to predict: {len(to_predict)}")

    if not to_predict:
        print("All structures already exist!")
    else:
        # Predict in parallel on Modal
        t0 = time.time()
        results = list(predict_one.map(
            [seq for seq, _ in to_predict],
            [uid for _, uid in to_predict],
            [api_key] * len(to_predict),
        ))
        elapsed = time.time() - t0

        # Save results
        success = 0
        failed = 0
        for uid, pdb_string in results:
            if pdb_string:
                (structures_dir / f"{uid}.pdb").write_text(pdb_string)
                success += 1
            else:
                failed += 1

        print(f"Predicted {success} structures in {elapsed:.1f}s ({failed} failed)")

    # Build toxin-specific Foldseek database
    total_pdbs = len(list(structures_dir.glob("*.pdb")))
    print(f"\nTotal PDB files: {total_pdbs}")
    print("Building Foldseek toxin database...")

    db_dir = Path("data/foldseek_toxin_db")
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "toxins"

    result = subprocess.run(
        ["foldseek", "createdb", str(structures_dir), str(db_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Foldseek toxin DB built: {db_path}")
    else:
        print(f"Foldseek createdb failed: {result.stderr}")
