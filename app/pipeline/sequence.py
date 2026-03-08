"""Sequence validation and preprocessing utilities.

Handles:
- Amino acid sequence validation
- DNA / RNA → protein translation
- FASTA parsing
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


# ── Constants ─────────────────────────────────────────────────────────────────

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
IUPAC_AA = STANDARD_AA | set("BJOUXZ*-")  # extended / ambiguous / gap

STANDARD_NUC = set("ACGT")
IUPAC_NUC = STANDARD_NUC | set("URYSWKMBDHVN")

# Standard genetic code (codon → amino acid, stops as "*")
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


class SequenceType(Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    valid: bool
    sequence_type: SequenceType
    cleaned: str
    message: str = ""


# ── Detection ─────────────────────────────────────────────────────────────────


def detect_sequence_type(seq: str) -> SequenceType:
    """Heuristically detect whether *seq* is protein, DNA, or RNA.

    Prefers the more specific type when ambiguous (e.g. a string of only
    ``ACGT`` characters is reported as DNA rather than protein even though
    those letters are also valid amino acids).
    """
    upper = seq.upper()
    chars = set(upper)

    if chars <= STANDARD_NUC:
        # Purely ACGT — could be DNA *or* protein; prefer DNA
        return SequenceType.DNA
    if "U" in chars and chars <= IUPAC_NUC:
        return SequenceType.RNA
    if chars <= IUPAC_AA:
        return SequenceType.PROTEIN
    return SequenceType.UNKNOWN


# ── Validation ────────────────────────────────────────────────────────────────


def validate_sequence(raw: str) -> ValidationResult:
    """Validate and clean an input sequence string.

    Strips FASTA headers, whitespace, and checks character set.

    Args:
        raw: Raw sequence string, optionally with FASTA header.

    Returns:
        :class:`ValidationResult` with the cleaned sequence and detected type.
    """
    # Strip FASTA header lines
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    cleaned = re.sub(r"\s+", "", "".join(lines)).upper()

    if not cleaned:
        return ValidationResult(
            valid=False,
            sequence_type=SequenceType.UNKNOWN,
            cleaned="",
            message="Empty sequence after stripping headers and whitespace.",
        )

    seq_type = detect_sequence_type(cleaned)

    if seq_type == SequenceType.UNKNOWN:
        invalid_chars = set(cleaned) - IUPAC_AA
        return ValidationResult(
            valid=False,
            sequence_type=SequenceType.UNKNOWN,
            cleaned=cleaned,
            message=f"Sequence contains invalid characters: {invalid_chars}",
        )

    return ValidationResult(valid=True, sequence_type=seq_type, cleaned=cleaned)


def validate_protein_sequence(seq: str) -> ValidationResult:
    """Strictly validate a protein sequence (standard 20 AA only).

    Args:
        seq: Amino acid sequence (FASTA single-letter codes).

    Returns:
        :class:`ValidationResult`.
    """
    result = validate_sequence(seq)
    if not result.valid:
        return result

    if result.sequence_type != SequenceType.PROTEIN:
        return ValidationResult(
            valid=False,
            sequence_type=result.sequence_type,
            cleaned=result.cleaned,
            message=(
                f"Expected protein sequence but detected {result.sequence_type.value}. "
                "Use translate_to_protein() first."
            ),
        )

    non_standard = set(result.cleaned) - STANDARD_AA
    if non_standard:
        return ValidationResult(
            valid=True,  # Still usable, just warn
            sequence_type=SequenceType.PROTEIN,
            cleaned=result.cleaned,
            message=(
                f"Sequence contains non-standard/ambiguous amino acids: {non_standard}. "
                "These will be ignored during embedding."
            ),
        )

    return result


# ── Translation ───────────────────────────────────────────────────────────────


def _rna_to_dna(seq: str) -> str:
    """Replace uracil with thymine."""
    return seq.upper().replace("U", "T")


def translate_to_protein(
    nucleotide: str,
    stop_at_stop_codon: bool = True,
    reading_frame: int = 0,
) -> str:
    """Translate a DNA or RNA sequence to protein using the standard genetic code.

    Args:
        nucleotide: DNA or RNA nucleotide string (FASTA / IUPAC alphabet).
        stop_at_stop_codon: If ``True``, translation stops at the first ``*``
            stop codon and it is excluded from the output.
        reading_frame: 0-based offset to the first codon (0, 1, or 2).

    Returns:
        Single-letter amino acid sequence string.

    Raises:
        ValueError: If the sequence contains non-nucleotide characters or the
            reading frame is invalid.
    """
    if reading_frame not in (0, 1, 2):
        raise ValueError(f"reading_frame must be 0, 1, or 2; got {reading_frame}")

    result = validate_sequence(nucleotide)
    if not result.valid:
        raise ValueError(f"Invalid nucleotide sequence: {result.message}")
    if result.sequence_type not in (SequenceType.DNA, SequenceType.RNA):
        raise ValueError(
            f"translate_to_protein() requires a DNA or RNA sequence, "
            f"got {result.sequence_type.value}"
        )

    dna = _rna_to_dna(result.cleaned)[reading_frame:]
    protein_chars: list[str] = []

    for i in range(0, len(dna) - 2, 3):
        codon = dna[i : i + 3]
        if len(codon) < 3:
            break
        aa = CODON_TABLE.get(codon, "X")  # X = unknown
        if aa == "*" and stop_at_stop_codon:
            break
        protein_chars.append(aa)

    return "".join(protein_chars)


def find_orfs(dna: str, min_length: int = 100) -> list[dict]:
    """Find all open reading frames in a DNA sequence (all 6 frames).

    Args:
        dna: DNA sequence string.
        min_length: Minimum ORF length in nucleotides.

    Returns:
        List of dicts with keys ``frame``, ``start``, ``end``, ``protein``.
    """
    result = validate_sequence(dna)
    if not result.valid or result.sequence_type not in (
        SequenceType.DNA,
        SequenceType.RNA,
    ):
        return []

    clean_dna = _rna_to_dna(result.cleaned)
    orfs: list[dict] = []

    def _search_frame(seq: str, frame: int, reverse: bool) -> None:
        i = 0
        while i < len(seq) - 2:
            codon = seq[i : i + 3]
            if codon == "ATG":
                # Start codon found – extend until stop
                j = i + 3
                prot_chars = ["M"]
                while j <= len(seq) - 3:
                    c = seq[j : j + 3]
                    aa = CODON_TABLE.get(c, "X")
                    if aa == "*":
                        break
                    prot_chars.append(aa)
                    j += 3
                orf_len = j - i
                if orf_len >= min_length:
                    orfs.append(
                        {
                            "frame": frame if not reverse else -(frame + 1),
                            "start": i,
                            "end": j,
                            "protein": "".join(prot_chars),
                        }
                    )
                i = j  # skip to end of this ORF
            else:
                i += 3

    rc = clean_dna.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    for frame in range(3):
        _search_frame(clean_dna[frame:], frame, reverse=False)
        _search_frame(rc[frame:], frame, reverse=True)

    return sorted(orfs, key=lambda o: len(o["protein"]), reverse=True)
