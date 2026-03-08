#!/usr/bin/env python3
"""Command-line interface for building the protein toxin database.

Usage:
    python scripts/build_db.py --output-dir data --max-proteins 5000
    python scripts/build_db.py --help
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from app.config import get_settings
from app.database.build_db import build_database


def setup_logging(verbose: bool = False):
    """Configure logging for the script."""
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, format=log_format, level=level, colorize=True)


async def main():
    """Main entry point for the database building script."""
    parser = argparse.ArgumentParser(
        description="Build protein toxin database with ESM-2 embeddings and FAISS index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save database files (proteins.index and proteins.json)"
    )

    parser.add_argument(
        "--max-proteins",
        type=int,
        default=5000,
        help="Maximum number of proteins to include in the database"
    )

    parser.add_argument(
        "--query",
        type=str,
        default="(keyword:KW-0800) OR (keyword:KW-0872) OR (keyword:KW-0903)",
        help="UniProt search query for toxin proteins"
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size for ESM-2 embedding computation"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for ESM-2 model (overrides config setting)"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="ESM-2 model name (overrides config setting)"
    )

    parser.add_argument(
        "--save-fasta",
        action="store_true",
        help="Save downloaded sequences as FASTA file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download proteins but don't compute embeddings or build index"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Show configuration
    logger.info("🧬 Building protein toxin database")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max proteins: {args.max_proteins}")
    logger.info(f"UniProt query: {args.query}")
    logger.info(f"Embedding batch size: {args.embedding_batch_size}")

    if args.dry_run:
        logger.info("DRY RUN - will not compute embeddings or build index")

    # Override settings if specified
    settings = get_settings()
    if args.device:
        logger.info(f"Overriding device: {settings.device} -> {args.device}")
        settings.device = args.device
    if args.model_name:
        logger.info(f"Overriding model: {settings.esm2_model_name} -> {args.model_name}")
        settings.esm2_model_name = args.model_name

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            # Just download and validate proteins
            logger.info("Starting dry run...")
            from app.database.build_db import download_uniprot_proteins, save_fasta

            proteins = await download_uniprot_proteins(
                query=args.query,
                max_proteins=args.max_proteins
            )

            logger.info(f"Downloaded {len(proteins)} proteins")

            # Show some statistics
            organisms = {}
            toxin_types = {}
            seq_lengths = []

            for protein in proteins:
                organism = protein.get("organism", "Unknown")
                organisms[organism] = organisms.get(organism, 0) + 1

                toxin_type = protein.get("toxin_type", "unknown")
                toxin_types[toxin_type] = toxin_types.get(toxin_type, 0) + 1

                seq_lengths.append(protein.get("sequence_length", 0))

            logger.info("Statistics:")
            logger.info(f"  - Unique organisms: {len(organisms)}")
            logger.info(f"  - Top organisms: {sorted(organisms.items(), key=lambda x: x[1], reverse=True)[:5]}")
            logger.info(f"  - Toxin types: {dict(sorted(toxin_types.items(), key=lambda x: x[1], reverse=True))}")

            if seq_lengths:
                import numpy as np
                logger.info(f"  - Sequence length: mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.1f}, min={min(seq_lengths)}, max={max(seq_lengths)}")

            # Save FASTA if requested
            if args.save_fasta:
                fasta_path = args.output_dir / "proteins.fasta"
                save_fasta(proteins, fasta_path)

            logger.info("Dry run completed successfully!")

        else:
            # Full database build
            db = await build_database(
                output_dir=args.output_dir,
                max_proteins=args.max_proteins,
                query=args.query,
                embedding_batch_size=args.embedding_batch_size
            )

            logger.info("✅ Database build completed successfully!")

            # Show final statistics
            stats = db.get_statistics()
            logger.info("Final database statistics:")
            logger.info(f"  - Total proteins: {stats['total_proteins']}")
            logger.info(f"  - Embedding dimension: {stats['embedding_dimension']}")

            if "metadata_summary" in stats:
                summary = stats["metadata_summary"]
                logger.info(f"  - Unique organisms: {summary['unique_organisms']}")
                logger.info(f"  - Unique toxin types: {summary['unique_toxin_types']}")

            if "sequence_length_stats" in stats:
                seq_stats = stats["sequence_length_stats"]
                logger.info(f"  - Avg sequence length: {seq_stats['mean']:.1f} ± {seq_stats['std']:.1f}")

            # Save FASTA if requested
            if args.save_fasta:
                logger.info("Saving FASTA file...")
                # Reload proteins for FASTA export
                from app.database.build_db import save_fasta
                proteins = []
                for i in range(db.size):
                    meta = db.get_metadata(i)
                    proteins.append(meta)

                fasta_path = args.output_dir / "proteins.fasta"
                save_fasta(proteins, fasta_path)

            # Validate the database
            validation = db.validate_consistency()
            if validation["valid"]:
                logger.info("✅ Database validation passed")
            else:
                logger.error(f"❌ Database validation failed: {validation['issues']}")
                if validation["warnings"]:
                    logger.warning(f"Warnings: {validation['warnings']}")

    except KeyboardInterrupt:
        logger.info("❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Build failed: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())