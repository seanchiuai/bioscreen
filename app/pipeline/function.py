"""Protein function prediction using sequence analysis and mock annotations.

This module provides functionality to predict molecular function from protein sequences.
Currently uses mock predictions but can be extended with real ML models.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List

from loguru import logger

from app.models.schemas import FunctionPrediction


class FunctionPredictor:
    """Predicts protein function from amino acid sequences.

    Currently provides mock predictions based on sequence properties,
    but designed to be easily replaceable with real ML models.
    """

    def __init__(self) -> None:
        """Initialize the function predictor with mock databases."""
        self._load_mock_databases()

    def _load_mock_databases(self) -> None:
        """Load mock GO terms and EC numbers for prediction."""
        # Common GO terms for proteins that might be toxins or harmful
        self.toxin_go_terms = [
            {"term": "GO:0005576", "name": "extracellular region", "confidence": "0.8"},
            {"term": "GO:0090729", "name": "toxin activity", "confidence": "0.9"},
            {"term": "GO:0042742", "name": "defense response to bacterium", "confidence": "0.7"},
            {"term": "GO:0019835", "name": "cytolysis", "confidence": "0.85"},
            {"term": "GO:0016020", "name": "membrane", "confidence": "0.75"},
            {"term": "GO:0005102", "name": "signaling receptor binding", "confidence": "0.8"},
        ]

        self.benign_go_terms = [
            {"term": "GO:0003824", "name": "catalytic activity", "confidence": "0.9"},
            {"term": "GO:0005515", "name": "protein binding", "confidence": "0.85"},
            {"term": "GO:0008152", "name": "metabolic process", "confidence": "0.8"},
            {"term": "GO:0005737", "name": "cytoplasm", "confidence": "0.75"},
            {"term": "GO:0006810", "name": "transport", "confidence": "0.7"},
            {"term": "GO:0006950", "name": "response to stress", "confidence": "0.65"},
        ]

        # Common EC numbers
        self.toxin_ec_numbers = [
            {"number": "3.4.21.-", "name": "serine endopeptidase", "confidence": "0.8"},
            {"number": "3.2.2.-", "name": "hydrolysing N-glycosyl compounds", "confidence": "0.75"},
            {"number": "3.1.4.-", "name": "phosphoric diester hydrolases", "confidence": "0.7"},
        ]

        self.benign_ec_numbers = [
            {"number": "1.1.1.-", "name": "acting on CH-OH group NAD/NADP", "confidence": "0.85"},
            {"number": "2.7.1.-", "name": "phosphotransferases, alcohol acceptor", "confidence": "0.8"},
            {"number": "4.2.1.-", "name": "hydro-lyases", "confidence": "0.75"},
            {"number": "6.1.1.-", "name": "ligases forming aminoacyl-tRNA", "confidence": "0.7"},
        ]

        # Motifs that might indicate toxin-like function
        self.toxin_motifs = [
            "RGD",  # Cell adhesion motif
            "NGR",  # Vascular targeting
            "KGD",  # Integrin binding
            "LDV",  # Cell adhesion
            "REDV", # Endothelial cell binding
        ]

        self.danger_keywords = [
            "cytolytic", "hemolytic", "neurotoxin", "cardiotoxin",
            "phospholipase", "hyaluronidase", "metalloprotease"
        ]

    def _sequence_hash(self, sequence: str) -> int:
        """Generate a deterministic hash for reproducible mock predictions."""
        return int(hashlib.md5(sequence.encode()).hexdigest(), 16) % (2**32)

    def _analyze_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Analyze sequence for features that might indicate function."""
        features = {}

        # Basic composition analysis
        total_length = len(sequence)
        if total_length == 0:
            return features

        # Amino acid composition
        charged_residues = sum(sequence.count(aa) for aa in "RHKDE")
        hydrophobic_residues = sum(sequence.count(aa) for aa in "AILMFWYV")
        polar_residues = sum(sequence.count(aa) for aa in "STNQ")
        aromatic_residues = sum(sequence.count(aa) for aa in "FWY")

        features["charged_fraction"] = charged_residues / total_length
        features["hydrophobic_fraction"] = hydrophobic_residues / total_length
        features["polar_fraction"] = polar_residues / total_length
        features["aromatic_fraction"] = aromatic_residues / total_length

        # Signal peptide indicators (N-terminal hydrophobic stretch)
        n_term_hydrophobic = 0
        for i, aa in enumerate(sequence[:30]):  # First 30 residues
            if aa in "AILMFWYV":
                n_term_hydrophobic += 1
        features["signal_peptide_score"] = n_term_hydrophobic / min(30, total_length)

        # Cysteine content (disulfide bonds)
        features["cysteine_content"] = sequence.count("C") / total_length

        # Check for toxin-like motifs
        motif_count = 0
        for motif in self.toxin_motifs:
            motif_count += sequence.count(motif)
        features["toxin_motif_density"] = motif_count / (total_length / 100)  # per 100 residues

        return features

    def _predict_toxin_likelihood(self, sequence: str, features: Dict[str, float]) -> float:
        """Predict likelihood that sequence encodes a toxin-like protein."""
        score = 0.0

        # High signal peptide score suggests secreted protein
        if features.get("signal_peptide_score", 0) > 0.6:
            score += 0.3

        # High charged residue content
        if features.get("charged_fraction", 0) > 0.3:
            score += 0.2

        # Presence of toxin motifs
        if features.get("toxin_motif_density", 0) > 1.0:
            score += 0.4

        # High cysteine content (many toxins have disulfide bonds)
        if features.get("cysteine_content", 0) > 0.05:
            score += 0.2

        # Moderate hydrophobic content (membrane interaction)
        hydrophobic = features.get("hydrophobic_fraction", 0)
        if 0.3 < hydrophobic < 0.6:
            score += 0.15

        # Add some sequence-based randomness for variety
        seq_hash = self._sequence_hash(sequence)
        random.seed(seq_hash)
        score += random.uniform(-0.1, 0.1)

        return min(1.0, max(0.0, score))

    def predict(self, sequence: str) -> FunctionPrediction:
        """Predict molecular function for a protein sequence.

        Args:
            sequence: Single-letter amino acid sequence.

        Returns:
            FunctionPrediction with GO terms, EC numbers, and summary.
        """
        logger.debug(f"Predicting function for sequence of length {len(sequence)}")

        # Analyze sequence features
        features = self._analyze_sequence_features(sequence)

        # Predict toxin likelihood
        toxin_likelihood = self._predict_toxin_likelihood(sequence, features)

        # Use sequence hash for reproducible predictions
        seq_hash = self._sequence_hash(sequence)
        random.seed(seq_hash)

        # Select GO terms based on toxin likelihood
        predicted_go_terms = []
        if toxin_likelihood > 0.5:
            # More likely to be toxin-related
            n_toxin_terms = min(3, random.randint(1, 4))
            selected_terms = random.sample(self.toxin_go_terms, n_toxin_terms)
            predicted_go_terms.extend(selected_terms)

            # Maybe add some benign terms too
            if random.random() > 0.3:
                n_benign_terms = random.randint(1, 2)
                selected_benign = random.sample(self.benign_go_terms, n_benign_terms)
                predicted_go_terms.extend(selected_benign)
        else:
            # More likely to be benign
            n_benign_terms = min(4, random.randint(2, 5))
            selected_terms = random.sample(self.benign_go_terms, n_benign_terms)
            predicted_go_terms.extend(selected_terms)

        # Select EC numbers
        predicted_ec_numbers = []
        if random.random() > 0.4:  # 60% chance of having EC number
            if toxin_likelihood > 0.5:
                ec_pool = self.toxin_ec_numbers + self.benign_ec_numbers
            else:
                ec_pool = self.benign_ec_numbers

            n_ec = random.randint(1, 2)
            selected_ec = random.sample(ec_pool, min(n_ec, len(ec_pool)))
            predicted_ec_numbers.extend(selected_ec)

        # Generate summary
        summary_parts = []

        if toxin_likelihood > 0.7:
            summary_parts.append("Potentially harmful protein with toxin-like characteristics")
        elif toxin_likelihood > 0.4:
            summary_parts.append("Protein with some toxin-like features")
        else:
            summary_parts.append("Likely benign protein")

        if features.get("signal_peptide_score", 0) > 0.6:
            summary_parts.append("contains signal peptide")

        if features.get("cysteine_content", 0) > 0.05:
            summary_parts.append("disulfide-rich structure")

        if features.get("charged_fraction", 0) > 0.3:
            summary_parts.append("highly charged surface")

        # Add enzymatic activity if EC numbers predicted
        if predicted_ec_numbers:
            activities = [ec["name"] for ec in predicted_ec_numbers]
            summary_parts.append(f"enzymatic activity: {', '.join(activities)}")

        summary = "; ".join(summary_parts).capitalize() + "."

        return FunctionPrediction(
            go_terms=predicted_go_terms,
            ec_numbers=predicted_ec_numbers,
            summary=summary,
        )

    def batch_predict(self, sequences: List[str]) -> List[FunctionPrediction]:
        """Predict function for multiple sequences.

        Args:
            sequences: List of protein sequences.

        Returns:
            List of FunctionPrediction objects.
        """
        return [self.predict(seq) for seq in sequences]