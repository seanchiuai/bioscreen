#!/usr/bin/env python3
"""Demonstration script for the protein toxin screening pipeline.

This script shows how to use the pipeline components to:
1. Validate a protein sequence
2. Generate ESM-2 embeddings
3. Run similarity search against the toxin database
4. Predict molecular function
5. Compute risk score

Uses insulin as a demo sequence (should be low toxin risk).
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from app.config import get_settings
from app.database.toxin_db import ToxinDatabase
from app.pipeline.embedding import get_embedding_model
from app.pipeline.function import FunctionPredictor
from app.pipeline.scoring import compute_score
from app.pipeline.sequence import validate_protein_sequence
from app.pipeline.similarity import CombinedSimilaritySearcher


# Demo sequence: Human insulin
INSULIN_SEQUENCE = (
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGG"
    "PGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
)

DEMO_SEQUENCES = {
    "insulin": {
        "name": "Human Insulin",
        "sequence": INSULIN_SEQUENCE,
        "expected_risk": "LOW",
        "description": "Hormone that regulates glucose metabolism"
    },
    "lysozyme": {
        "name": "Human Lysozyme",
        "sequence": (
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINS"
            "RWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDV"
            "QAWIRGCRL"
        ),
        "expected_risk": "LOW",
        "description": "Antimicrobial enzyme"
    },
    "short_test": {
        "name": "Short Test Peptide",
        "sequence": "MKAIFVLKGWWRT",
        "expected_risk": "UNKNOWN",
        "description": "Short synthetic peptide for testing"
    }
}


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )


async def load_database() -> ToxinDatabase:
    """Load the toxin database."""
    settings = get_settings()

    if not settings.toxin_db_path.exists() or not settings.toxin_meta_path.exists():
        logger.error("Toxin database not found. Please run 'python scripts/build_db.py' first.")
        logger.error(f"Expected files:")
        logger.error(f"  - {settings.toxin_db_path}")
        logger.error(f"  - {settings.toxin_meta_path}")
        raise FileNotFoundError("Toxin database not found")

    logger.info("Loading toxin database...")
    db = ToxinDatabase(
        index_path=settings.toxin_db_path,
        meta_path=settings.toxin_meta_path,
        embedding_dim=1280,  # ESM-2 650M
    )

    db.load()
    logger.info(f"Loaded database with {db.size} proteins")

    return db


async def demo_sequence_analysis(
    sequence_name: str,
    sequence_data: dict,
    toxin_db: ToxinDatabase,
    embedding_model,
    function_predictor,
) -> dict:
    """Run complete analysis pipeline on a sequence."""
    sequence = sequence_data["sequence"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {sequence_data['name']}")
    logger.info(f"Expected Risk: {sequence_data['expected_risk']}")
    logger.info(f"Description: {sequence_data['description']}")
    logger.info(f"Sequence Length: {len(sequence)} amino acids")
    logger.info(f"{'='*60}")

    results = {
        "sequence_id": sequence_name,
        "sequence_name": sequence_data["name"],
        "sequence": sequence,
        "sequence_length": len(sequence),
        "expected_risk": sequence_data["expected_risk"],
        "description": sequence_data["description"]
    }

    # Step 1: Sequence validation
    logger.info("Step 1: Validating sequence...")
    validation = validate_protein_sequence(sequence)

    if not validation.valid:
        logger.error(f"❌ Invalid sequence: {validation.message}")
        results["error"] = f"Invalid sequence: {validation.message}"
        return results

    logger.info(f"✅ Sequence valid: {validation.sequence_type.value}")
    results["validation"] = {
        "valid": validation.valid,
        "sequence_type": validation.sequence_type.value,
        "cleaned_length": len(validation.cleaned),
        "message": validation.message
    }

    # Step 2: Generate embedding
    logger.info("Step 2: Generating ESM-2 embedding...")
    try:
        query_embedding = embedding_model.embed(validation.cleaned)
        logger.info(f"✅ Generated embedding: shape {query_embedding.shape}")
        results["embedding"] = {
            "shape": query_embedding.shape,
            "norm": float(np.linalg.norm(query_embedding))
        }
    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        results["error"] = f"Embedding failed: {e}"
        return results

    # Step 3: Similarity search
    logger.info("Step 3: Running similarity search...")
    try:
        similarity_searcher = CombinedSimilaritySearcher(toxin_db)
        similarity_result = await similarity_searcher.search(
            query_embedding=query_embedding,
            top_k=5,
            run_structure=False  # Skip structure for demo
        )

        max_sim = similarity_result.max_embedding_sim
        top_matches = []

        for i, hit in enumerate(similarity_result.embedding_hits[:3], 1):
            match_info = {
                "rank": i,
                "uniprot_id": hit.metadata.get("uniprot_id", "Unknown"),
                "name": hit.metadata.get("name", "Unknown")[:50],
                "organism": hit.metadata.get("organism", "Unknown"),
                "toxin_type": hit.metadata.get("toxin_type", "unknown"),
                "similarity": float(hit.cosine_similarity),
                "sequence_length": hit.metadata.get("sequence_length", 0)
            }
            top_matches.append(match_info)
            logger.info(f"  Match {i}: {match_info['name']} ({match_info['uniprot_id']}) - {match_info['similarity']:.3f}")

        logger.info(f"✅ Similarity search complete. Max similarity: {max_sim:.3f}")
        results["similarity"] = {
            "max_embedding_similarity": float(max_sim),
            "top_matches": top_matches,
            "total_matches": len(similarity_result.embedding_hits)
        }

    except Exception as e:
        logger.error(f"❌ Similarity search failed: {e}")
        results["error"] = f"Similarity search failed: {e}"
        return results

    # Step 4: Function prediction
    logger.info("Step 4: Predicting molecular function...")
    try:
        function_prediction = function_predictor.predict(validation.cleaned)

        function_info = {
            "summary": function_prediction.summary,
            "go_terms": function_prediction.go_terms[:3],  # Top 3
            "ec_numbers": function_prediction.ec_numbers[:3],  # Top 3
        }

        logger.info(f"✅ Function prediction: {function_prediction.summary}")
        if function_prediction.go_terms:
            logger.info(f"  Top GO terms: {[term['term'] for term in function_prediction.go_terms[:3]]}")

        results["function_prediction"] = function_info

    except Exception as e:
        logger.error(f"❌ Function prediction failed: {e}")
        results["function_prediction"] = {"error": str(e)}

    # Step 5: Risk scoring
    logger.info("Step 5: Computing risk score...")
    try:
        # Calculate function overlap (simplified)
        function_overlap = 0.0
        if "function_prediction" in results and results["function_prediction"].get("go_terms"):
            query_go_terms = {term["term"] for term in results["function_prediction"]["go_terms"]}
            for match in top_matches[:3]:
                # This would normally use actual GO terms from database
                # For demo, we'll use a placeholder calculation
                if match["toxin_type"] != "unknown":
                    function_overlap = max(function_overlap, 0.1)  # Placeholder

        risk_score, score_explanation = compute_score(
            embedding_sim=max_sim,
            structural_sim=None,  # Not computed in demo
            function_overlap=function_overlap,
        )

        # Determine risk level using default thresholds
        if risk_score >= 0.75:
            risk_level = "HIGH"
        elif risk_score >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        logger.info(f"✅ Risk assessment complete")
        logger.info(f"  Risk Score: {risk_score:.3f}")
        logger.info(f"  Risk Level: {risk_level}")
        logger.info(f"  Expected: {sequence_data['expected_risk']}")

        # Check if result matches expectation
        matches_expected = risk_level == sequence_data['expected_risk']
        if matches_expected:
            logger.info(f"✅ Result matches expected risk level")
        else:
            logger.warning(f"⚠️  Result ({risk_level}) differs from expected ({sequence_data['expected_risk']})")

        results["risk_assessment"] = {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "expected_risk": sequence_data["expected_risk"],
            "matches_expected": matches_expected,
            "score_explanation": score_explanation,
            "max_embedding_similarity": float(max_sim),
            "function_overlap": float(function_overlap)
        }

    except Exception as e:
        logger.error(f"❌ Risk scoring failed: {e}")
        results["risk_assessment"] = {"error": str(e)}

    return results


async def main():
    """Main demo function."""
    setup_logging()

    logger.info("🧬 BioScreen Pipeline Demo")
    logger.info("This demo runs the complete toxin screening pipeline on example sequences")
    logger.info("")

    try:
        # Load components
        logger.info("Loading pipeline components...")

        # Load database
        toxin_db = await load_database()

        # Load embedding model
        logger.info("Loading ESM-2 embedding model...")
        embedding_model = get_embedding_model()
        logger.info(f"✅ Loaded ESM-2 model: {embedding_model.model_name}")

        # Load function predictor
        logger.info("Loading function predictor...")
        function_predictor = FunctionPredictor()
        logger.info("✅ Function predictor ready")

        # Run demos
        all_results = []

        for seq_id, seq_data in DEMO_SEQUENCES.items():
            result = await demo_sequence_analysis(
                seq_id,
                seq_data,
                toxin_db,
                embedding_model,
                function_predictor
            )
            all_results.append(result)

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("DEMO SUMMARY")
        logger.info(f"{'='*60}")

        for result in all_results:
            sequence_name = result.get("sequence_name", "Unknown")
            risk_score = result.get("risk_assessment", {}).get("risk_score", "Error")
            risk_level = result.get("risk_assessment", {}).get("risk_level", "Error")
            expected = result.get("expected_risk", "Unknown")
            matches = result.get("risk_assessment", {}).get("matches_expected", False)

            status = "✅" if matches else "⚠️"
            logger.info(f"{status} {sequence_name}: {risk_score:.3f} ({risk_level}) [Expected: {expected}]")

        # Save results
        output_file = Path("demo_results.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\n📁 Results saved to: {output_file}")
        logger.info("🎉 Demo completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    import numpy as np

    asyncio.run(main())