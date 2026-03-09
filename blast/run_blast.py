#!/usr/bin/env python3
"""Standalone BLAST+ CLI tool for pairwise sequence alignment.

Completely independent from the BioScreen app — no shared imports.

Usage:
    python blast/run_blast.py --query "MKTLLILAVL..." --subject "MKFLILLAVL..."
    python blast/run_blast.py --query query.fasta --subject subject.fasta
    python blast/run_blast.py --query unsafe.fasta --subject safe.fasta
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys
import tempfile


def is_fasta_file(value: str) -> bool:
    """Check if value looks like a path to an existing file."""
    return os.path.isfile(value)


def detect_seq_type(sequence: str) -> str:
    """Detect if sequence is protein or nucleotide."""
    clean = re.sub(r"[^A-Za-z]", "", sequence)
    nuc_chars = set("ATCGUNatcgun")
    nuc_count = sum(1 for c in clean if c in nuc_chars)
    if len(clean) == 0:
        sys.exit("Error: empty sequence provided.")
    if nuc_count / len(clean) > 0.9:
        return "nucl"
    return "prot"


def raw_to_fasta(raw: str, label: str = "seq") -> str:
    """Wrap a raw sequence string in FASTA format."""
    clean = re.sub(r"\s+", "", raw)
    lines = [f">{label}"]
    for i in range(0, len(clean), 60):
        lines.append(clean[i : i + 60])
    return "\n".join(lines) + "\n"


def load_input(value: str, label: str) -> tuple[str, str]:
    """Load input as FASTA content + detect type.

    Returns (fasta_content, seq_type).
    """
    if is_fasta_file(value):
        with open(value) as f:
            content = f.read().strip()
        # Extract just the sequence characters for type detection
        seq_lines = [l for l in content.splitlines() if not l.startswith(">")]
        seq_type = detect_seq_type("".join(seq_lines))
        return content + "\n", seq_type
    else:
        seq_type = detect_seq_type(value)
        return raw_to_fasta(value, label), seq_type


def run_blast(query_input: str, subject_input: str) -> None:
    """Run BLAST+ alignment between query and subject."""
    query_fasta, q_type = load_input(query_input, "query")
    subject_fasta, s_type = load_input(subject_input, "subject")

    # Pick blast program
    if q_type == "prot" and s_type == "prot":
        program = "blastp"
    elif q_type == "nucl" and s_type == "nucl":
        program = "blastn"
    else:
        sys.exit(
            f"Error: mismatched sequence types (query={q_type}, subject={s_type}). "
            "Both must be protein or both nucleotide."
        )

    # Write temp files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as qf:
        qf.write(query_fasta)
        query_path = qf.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as sf:
        sf.write(subject_fasta)
        subject_path = sf.name

    try:
        # Use tabular output for parsing:
        # qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
        cmd = [
            program,
            "-query", query_path,
            "-subject", subject_path,
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-evalue", "10",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"BLAST error:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        print_summary(result.stdout)

    finally:
        os.unlink(query_path)
        os.unlink(subject_path)


def print_summary(blast_output: str) -> None:
    """Parse tabular BLAST output and print a clean, readable summary."""
    if not blast_output.strip():
        print("\n  No alignments found — sequences share no detectable similarity.\n")
        return

    # Parse rows
    rows = []
    reader = csv.reader(io.StringIO(blast_output), delimiter="\t")
    for row in reader:
        if len(row) < 12:
            continue
        rows.append({
            "query": row[0],
            "subject": row[1],
            "identity": float(row[2]),
            "length": int(row[3]),
            "evalue": float(row[10]),
            "bitscore": float(row[11]),
        })

    if not rows:
        print("\n  No alignments found — sequences share no detectable similarity.\n")
        return

    # Group by query
    queries = {}
    for r in rows:
        queries.setdefault(r["query"], []).append(r)

    # Print summary
    print()
    print("=" * 78)
    print("  BLAST RESULTS SUMMARY")
    print("=" * 78)

    for query, hits in queries.items():
        # Short name: take first part before |
        q_short = query.split("|")[0] if "|" in query else query
        q_label = query.split("|")[1] if "|" in query else query
        print(f"\n  Query: {q_label} ({q_short})")
        print(f"  {'Hit':<35} {'Identity':>8} {'Length':>6} {'E-value':>10} {'Score':>6}  {'Risk'}")
        print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*10} {'-'*6}  {'-'*10}")

        for hit in sorted(hits, key=lambda x: x["evalue"]):
            s_label = hit["subject"].split("|")[1] if "|" in hit["subject"] else hit["subject"]
            ident = f"{hit['identity']:.1f}%"
            ev = f"{hit['evalue']:.1e}" if hit["evalue"] < 0.01 else f"{hit['evalue']:.2f}"
            score = f"{hit['bitscore']:.0f}"

            # Risk assessment based on e-value and identity
            if hit["evalue"] < 1e-5 and hit["identity"] > 50:
                risk = "HIGH"
                marker = ">>>"
            elif hit["evalue"] < 0.01 and hit["identity"] > 30:
                risk = "MEDIUM"
                marker = " >>"
            elif hit["evalue"] < 1.0:
                risk = "LOW"
                marker = "  >"
            else:
                risk = "NONE"
                marker = "   "

            print(f"  {s_label:<35} {ident:>8} {hit['length']:>6} {ev:>10} {score:>6}  {marker} {risk}")

    # Overall verdict
    best_evalue = min(r["evalue"] for r in rows)
    best_identity = max(r["identity"] for r in rows)

    print()
    print("-" * 78)
    if best_evalue < 1e-5 and best_identity > 50:
        print("  VERDICT: Significant sequence similarity detected.")
    elif best_evalue < 0.01:
        print("  VERDICT: Moderate similarity — warrants further investigation.")
    else:
        print("  VERDICT: No significant similarity. BLAST cannot distinguish these.")
        print("           (Structure-based screening recommended for AI-designed proteins)")
    print("-" * 78)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run BLAST+ pairwise alignment between sequences.",
        epilog="Requires BLAST+ installed (brew install blast).",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query sequence: raw sequence string or path to FASTA file.",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject sequence: raw sequence string or path to FASTA file.",
    )
    args = parser.parse_args()

    run_blast(args.query, args.subject)


if __name__ == "__main__":
    main()
