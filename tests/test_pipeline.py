import pytest
from app.pipeline.sequence import validate_sequence, validate_protein_sequence, translate_to_protein, SequenceType
from app.pipeline.scoring import compute_score

def test_validate_valid_amino_acid_sequence():
    result = validate_protein_sequence('MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPSVMDFAMSPEMLNGGSVTTYRLSSGKTPQTLKYVGGQPASQNLVYNLLSTAM')
    assert result.valid
    assert result.sequence_type == SequenceType.PROTEIN

def test_validate_sequence_with_invalid_chars():
    result = validate_sequence('ACGT12345INVALID!!!')
    assert not result.valid

def test_translate_dna_to_protein():
    dna = 'ATGGCCAAATAA'
    protein = translate_to_protein(dna)
    assert protein.startswith('M')
    assert isinstance(protein, str)

def test_score_returns_value_in_range():
    score, explanation = compute_score(0.5, 0.5, 0.5)
    assert 0.0 <= score <= 1.0
    assert isinstance(explanation, str)
    assert len(explanation) > 0

def test_score_low_inputs_gives_low_score():
    score, _ = compute_score(0.1, 0.1, 0.0)
    assert score < 0.5

def test_score_high_inputs_gives_high_score():
    score, _ = compute_score(0.95, 0.95, 0.9)
    assert score > 0.5

def test_score_without_structural_sim():
    score, explanation = compute_score(0.7, None, 0.3)
    assert 0.0 <= score <= 1.0
    assert 'structural analysis not performed' in explanation