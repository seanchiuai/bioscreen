#!/usr/bin/env python3
"""Download AlphaFold structures for toxin reference proteins.

For each UniProt ID in our toxin DB, tries AlphaFold DB first (free bulk download),
then falls back to ESMFold NIM API for any missing structures.
Finally builds a toxin-specific Foldseek database from the collected PDB files.
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb"
STRUCTURES_DIR = Path("data/toxin_structures")
META_PATH = Path("data/toxin_meta.json")


def download_alphafold(uniprot_id: str, output_dir: Path) -> bool:
    """Download AlphaFold structure for a UniProt ID. Returns True if successful."""
    pdb_path = output_dir / f"{uniprot_id}.pdb"
    if pdb_path.exists() and pdb_path.stat().st_size > 100:
        return True  # already downloaded

    url = ALPHAFOLD_URL.format(uniprot_id)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and len(resp.content) > 100:
            pdb_path.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_batch_alphafold(uniprot_ids: list[str], output_dir: Path, workers: int = 10) -> tuple[list[str], list[str]]:
    """Download AlphaFold structures in parallel. Returns (success_ids, failed_ids)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    success = []
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_alphafold, uid, output_dir): uid
            for uid in uniprot_ids
        }
        with tqdm(total=len(uniprot_ids), desc="Downloading AlphaFold structures") as pbar:
            for future in as_completed(futures):
                uid = futures[future]
                if future.result():
                    success.append(uid)
                else:
                    failed.append(uid)
                pbar.update(1)

    return success, failed


def predict_esmfold_nim(sequence: str, api_key: str) -> str | None:
    """Predict structure using ESMFold NIM API. Returns PDB string or None."""
    url = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Truncate very long sequences for ESMFold
    seq = sequence[:800]
    try:
        resp = requests.post(url, headers=headers, json={"sequence": seq}, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("pdbs", [None])[0]
    except requests.RequestException as e:
        print(f"  ESMFold API error: {e}")
    return None


def predict_missing_with_esmfold(failed_ids: list[str], metadata: list[dict], output_dir: Path, api_key: str) -> list[str]:
    """Predict structures for IDs missing from AlphaFold using ESMFold NIM."""
    if not api_key:
        print("No NVIDIA API key, skipping ESMFold predictions")
        return failed_ids

    # Build ID → sequence lookup
    seq_map = {m["uniprot_id"]: m["sequence"] for m in metadata}
    still_failed = []

    print(f"Predicting {len(failed_ids)} structures with ESMFold NIM API...")
    for uid in tqdm(failed_ids, desc="ESMFold predictions"):
        seq = seq_map.get(uid, "")
        if not seq:
            still_failed.append(uid)
            continue

        pdb_string = predict_esmfold_nim(seq, api_key)
        if pdb_string:
            pdb_path = output_dir / f"{uid}.pdb"
            pdb_path.write_text(pdb_string)
        else:
            still_failed.append(uid)

        time.sleep(0.5)  # Rate limiting

    return still_failed


def build_foldseek_db(structures_dir: Path, db_path: Path):
    """Build a Foldseek database from a directory of PDB files."""
    import subprocess

    pdb_files = list(structures_dir.glob("*.pdb"))
    if not pdb_files:
        print("No PDB files found!")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building Foldseek database from {len(pdb_files)} structures...")
    result = subprocess.run(
        ["foldseek", "createdb", str(structures_dir), str(db_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"Foldseek toxin database built: {db_path}")
    else:
        print(f"Foldseek createdb failed: {result.stderr}")


def main():
    # Load metadata
    with open(META_PATH) as f:
        metadata = json.load(f)

    uniprot_ids = [m["uniprot_id"] for m in metadata]
    print(f"Total toxins: {len(uniprot_ids)}")

    # Step 1: Download from AlphaFold DB
    success, failed = download_batch_alphafold(uniprot_ids, STRUCTURES_DIR)
    print(f"AlphaFold: {len(success)} downloaded, {len(failed)} missing")

    # Step 2: Predict missing with ESMFold NIM
    if failed:
        from app.config import get_settings
        settings = get_settings()
        still_failed = predict_missing_with_esmfold(
            failed, metadata, STRUCTURES_DIR, settings.nvidia_api_key
        )
        print(f"After ESMFold: {len(still_failed)} still missing")
    else:
        still_failed = []

    # Step 3: Build toxin-specific Foldseek database
    toxin_foldseek_db = Path("data/foldseek_toxin_db/toxins")
    build_foldseek_db(STRUCTURES_DIR, toxin_foldseek_db)

    # Summary
    total_structures = len(list(STRUCTURES_DIR.glob("*.pdb")))
    print(f"\nDone! {total_structures}/{len(uniprot_ids)} structures collected")
    if still_failed:
        print(f"Missing: {still_failed[:20]}{'...' if len(still_failed) > 20 else ''}")


if __name__ == "__main__":
    main()
