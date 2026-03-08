#!/usr/bin/env python3
"""Enrich toxin metadata with GO terms and EC numbers from UniProt.

Fetches annotations for all UniProt IDs in our toxin DB and patches
the metadata JSON in place. Does not touch the FAISS index.
"""

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from tqdm import tqdm

META_PATH = Path("data/toxin_meta.json")
BATCH_SIZE = 50  # Keep URL short enough for UniProt API


def fetch_annotations_batch(uniprot_ids: list[str]) -> dict[str, dict]:
    """Fetch GO terms and EC numbers for a batch of UniProt IDs."""
    query = " OR ".join(f"accession:{uid}" for uid in uniprot_ids)
    params = {
        "query": query,
        "format": "json",
        "fields": "accession,go,ec",
        "size": len(uniprot_ids),
    }
    url = f"https://rest.uniprot.org/uniprotkb/search?{urlencode(params)}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Error fetching batch: {e}")
        return {}

    results = {}
    for entry in data.get("results", []):
        uid = entry.get("primaryAccession", "")
        if not uid:
            continue

        # Extract GO terms
        go_terms = []
        for ref in entry.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "GO":
                go_id = ref.get("id", "")
                go_name = ""
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GoTerm":
                        go_name = prop.get("value", "")
                if go_id:
                    go_terms.append(f"{go_id}:{go_name}" if go_name else go_id)

        # Extract EC numbers
        ec_numbers = []
        for ref in entry.get("uniProtKBCrossReferences", []):
            if ref.get("database") == "EC":
                ec_numbers.append(ref.get("id", ""))
        # Also check protein description for EC
        for ec in entry.get("proteinDescription", {}).get("ecNumbers", []):
            ec_val = ec.get("value", "")
            if ec_val and ec_val not in ec_numbers:
                ec_numbers.append(ec_val)

        results[uid] = {"go_terms": go_terms, "ec_numbers": ec_numbers}

    return results


def main():
    with open(META_PATH) as f:
        metadata = json.load(f)

    print(f"Loaded {len(metadata)} entries from {META_PATH}")

    # Check current state
    has_go = sum(1 for m in metadata if m.get("go_terms"))
    print(f"Currently have GO terms: {has_go}/{len(metadata)}")

    # Collect all UniProt IDs
    all_ids = [m["uniprot_id"] for m in metadata]

    # Fetch in batches
    annotations = {}
    for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="Fetching GO terms"):
        batch = all_ids[i:i + BATCH_SIZE]
        batch_results = fetch_annotations_batch(batch)
        annotations.update(batch_results)
        time.sleep(0.3)  # Rate limiting

    print(f"Fetched annotations for {len(annotations)} proteins")

    # Patch metadata
    updated = 0
    for entry in metadata:
        uid = entry["uniprot_id"]
        if uid in annotations:
            ann = annotations[uid]
            if ann["go_terms"]:
                entry["go_terms"] = ann["go_terms"]
                updated += 1
            if ann["ec_numbers"]:
                entry["ec_numbers"] = ann["ec_numbers"]

    print(f"Updated {updated} entries with GO terms")

    # Save
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=None)

    # Verify
    has_go_after = sum(1 for m in metadata if m.get("go_terms"))
    has_ec = sum(1 for m in metadata if m.get("ec_numbers"))
    print(f"GO terms: {has_go_after}/{len(metadata)}")
    print(f"EC numbers: {has_ec}/{len(metadata)}")

    # Sample
    for m in metadata[:5]:
        if m.get("go_terms"):
            print(f"  {m['uniprot_id']}: {m['go_terms'][:3]}")
            break


if __name__ == "__main__":
    main()
