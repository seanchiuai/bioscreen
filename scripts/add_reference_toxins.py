#!/usr/bin/env python3
"""Add reference toxins to the existing FAISS database.

Adds canonical toxins (ricin, diphtheria toxin, etc.) so that
structurally-similar demo proteins with low sequence identity
can be detected — demonstrating advantage over BLAST.

Runs in two phases to avoid OOM:
  Phase 1: Fetch sequences + compute embeddings (save to .npy)
  Phase 2: Load FAISS DB + append embeddings + save
"""

import asyncio
import gc
import json
import sys
from pathlib import Path

import httpx
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


# Reference toxins to add to the DB (canonical versions that demos will match against)
REFERENCE_TOXINS = {
    # RIP family — ricin-like fold
    "P02879": {"name": "Ricin A chain", "organism": "Ricinus communis", "toxin_type": "ribosome_inactivating_protein"},
    "P11140": {"name": "Abrin-a A chain", "organism": "Abrus precatorius", "toxin_type": "ribosome_inactivating_protein"},
    # ADP-ribosyltransferases
    "P00588": {"name": "Diphtheria toxin", "organism": "Corynebacterium diphtheriae", "toxin_type": "ADP_ribosyltransferase"},
    "P11439": {"name": "Exotoxin A", "organism": "Pseudomonas aeruginosa", "toxin_type": "ADP_ribosyltransferase"},
    # Pore-forming toxins
    "P77335": {"name": "Cytolysin A (ClyA)", "organism": "Escherichia coli", "toxin_type": "pore_forming_toxin"},
    "P09616": {"name": "Alpha-hemolysin (HlyA)", "organism": "Staphylococcus aureus", "toxin_type": "pore_forming_toxin"},
}

# Demo sequences — these are the QUERY proteins that users will screen
DEMO_TOXINS = {
    "P09989": {"name": "Trichosanthin", "organism": "Trichosanthes kirilowii", "toxin_type": "ribosome_inactivating_protein"},
    "P20656": {"name": "Saporin", "organism": "Saponaria officinalis", "toxin_type": "ribosome_inactivating_protein"},
    "Q6T4F7": {"name": "Bouganin", "organism": "Bougainvillea spectabilis", "toxin_type": "ribosome_inactivating_protein"},
    "P24817": {"name": "MAP30", "organism": "Momordica charantia", "toxin_type": "ribosome_inactivating_protein"},
    "Q8GN24": {"name": "Cholix toxin", "organism": "Vibrio cholerae", "toxin_type": "ADP_ribosyltransferase"},
}


async def fetch_uniprot_sequence(client: httpx.AsyncClient, uniprot_id: str) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        return "".join(line.strip() for line in lines if not line.startswith(">"))
    except Exception as e:
        logger.error(f"Failed to fetch {uniprot_id}: {e}")
        return None


async def fetch_all_sequences(protein_ids: dict) -> dict[str, str]:
    sequences = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for uid in protein_ids:
            seq = await fetch_uniprot_sequence(client, uid)
            if seq:
                sequences[uid] = seq
                logger.info(f"  {uid} ({protein_ids[uid]['name']}): {len(seq)}aa")
            else:
                logger.warning(f"  {uid}: FAILED")
    return sequences


async def main():
    data_dir = project_root / "data"
    index_path = data_dir / "toxin_db.faiss"
    meta_path = data_dir / "toxin_meta.json"
    tmp_emb_path = data_dir / "_tmp_new_embeddings.npy"
    tmp_meta_path = data_dir / "_tmp_new_metadata.json"

    # --- Check which toxins need adding (just read metadata JSON, no FAISS) ---
    with open(meta_path) as f:
        existing_meta = json.load(f)
    existing_ids = {m.get("uniprot_id") for m in existing_meta}
    logger.info(f"Existing DB has {len(existing_meta)} entries")

    new_refs = {uid: info for uid, info in REFERENCE_TOXINS.items() if uid not in existing_ids}
    if not new_refs:
        logger.info("All reference toxins already in DB!")
    else:
        logger.info(f"Need to add {len(new_refs)} reference toxins: {list(new_refs.keys())}")

        # --- Phase 1: Fetch sequences & compute embeddings ---
        logger.info("Fetching sequences from UniProt...")
        sequences = await fetch_all_sequences(new_refs)
        if not sequences:
            logger.error("No sequences fetched!")
            return

        logger.info("Loading ESM-2 model...")
        from app.pipeline.embedding import EmbeddingModel
        model = EmbeddingModel(model_name="facebook/esm2_t33_650M_UR50D", device="cpu")
        model.load()

        uid_list = list(sequences.keys())
        embeddings = []
        for uid in uid_list:
            seq = sequences[uid][:512]  # truncate long seqs
            logger.info(f"  Embedding {uid} ({len(seq)}aa)...")
            emb = model.embed(seq)
            embeddings.append(emb)

        emb_matrix = np.array(embeddings, dtype=np.float32)

        # Build metadata for new entries
        new_metadata = []
        for uid in uid_list:
            info = new_refs[uid]
            new_metadata.append({
                "uniprot_id": uid,
                "name": info["name"],
                "organism": info["organism"],
                "sequence": sequences[uid],
                "sequence_length": len(sequences[uid]),
                "toxin_type": info["toxin_type"],
                "go_terms": [],
                "ec_numbers": [],
                "reviewed": True,
            })

        # Save to temp files
        np.save(str(tmp_emb_path), emb_matrix)
        with open(tmp_meta_path, "w") as f:
            json.dump(new_metadata, f)
        logger.info(f"Saved {len(uid_list)} embeddings to temp files")

        # Free the model
        del model, embeddings, emb_matrix
        gc.collect()

        # --- Phase 2: Load FAISS DB and append ---
        logger.info("Loading FAISS index and appending new entries...")
        from app.database.toxin_db import ToxinDatabase
        db = ToxinDatabase(index_path=str(index_path), meta_path=str(meta_path))
        db.load()

        new_emb = np.load(str(tmp_emb_path))
        with open(tmp_meta_path) as f:
            new_meta = json.load(f)

        db.add_proteins(new_emb, new_meta)
        db.save()
        logger.info(f"DB now has {db.size} entries")

        # Cleanup temp files
        tmp_emb_path.unlink(missing_ok=True)
        tmp_meta_path.unlink(missing_ok=True)

        del db
        gc.collect()

    # --- Fetch demo sequences for frontend ---
    logger.info("\nFetching demo sequences for frontend...")
    demo_seqs = await fetch_all_sequences(DEMO_TOXINS)

    logger.info("\n=== DEMO SEQUENCES ===\n")
    for uid, seq in demo_seqs.items():
        info = DEMO_TOXINS[uid]
        logger.info(f'{info["name"]} ({uid}): {len(seq)}aa')

    # Save demo sequences JSON
    demo_output = {}
    for uid, seq in demo_seqs.items():
        info = DEMO_TOXINS[uid]
        demo_output[uid] = {
            "name": info["name"],
            "organism": info["organism"],
            "sequence": seq,
            "length": len(seq),
        }
    demo_path = data_dir / "demo_sequences.json"
    with open(demo_path, "w") as f:
        json.dump(demo_output, f, indent=2)
    logger.info(f"Demo sequences saved to {demo_path}")


if __name__ == "__main__":
    asyncio.run(main())
