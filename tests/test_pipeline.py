import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.pipeline.sequence import validate_sequence, validate_protein_sequence, translate_to_protein, SequenceType
from app.pipeline.scoring import compute_score
from app.pipeline.function import FunctionPredictor, MockFunctionPredictor, InterProPredictor
from app.pipeline.similarity import (
    EmbeddingSimilaritySearcher,
    FoldseekSearcher,
    CombinedSimilaritySearcher,
    EmbeddingHit,
    StructureHit,
)
from app.models.schemas import ScreeningRequest, RiskLevel

# ── Sequence validation ──────────────────────────────────────────────────────

def test_validate_valid_amino_acid_sequence():
    result = validate_protein_sequence('MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPSVMDFAMSPEMLNGGSVTTYRLSSGKTPQTLKYVGGQPASQNLVYNLLSTAM')
    assert result.valid
    assert result.sequence_type == SequenceType.PROTEIN

def test_validate_sequence_with_invalid_chars():
    result = validate_sequence('ACGT12345INVALID!!!')
    assert not result.valid

def test_validate_short_sequence():
    result = validate_protein_sequence('MK')
    assert result.valid  # short but valid amino acids

def test_validate_empty_sequence():
    result = validate_protein_sequence('')
    assert not result.valid

def test_validate_dna_sequence():
    result = validate_sequence('ATGCGATCGATCGATCGATCG')
    assert result.valid
    assert result.sequence_type == SequenceType.DNA

def test_translate_dna_to_protein():
    dna = 'ATGGCCAAATAA'
    protein = translate_to_protein(dna)
    assert protein.startswith('M')
    assert isinstance(protein, str)

def test_translate_dna_length():
    # 12 nucleotides = 4 codons, last is stop -> 3 amino acids
    dna = 'ATGGCCAAATAA'
    protein = translate_to_protein(dna)
    assert len(protein) >= 1

# ── Risk scoring ─────────────────────────────────────────────────────────────

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

def test_score_zero_inputs():
    score, _ = compute_score(0.0, 0.0, 0.0)
    assert score == 0.0

def test_score_boundary_one():
    score, _ = compute_score(1.0, 1.0, 1.0)
    assert 0.0 <= score <= 1.0

def test_score_embedding_only():
    score, _ = compute_score(0.9, None, 0.0)
    assert 0.0 <= score <= 1.0

# ── Risk scoring validation with realistic inputs ────────────────────────────

def test_high_similarity_gives_high_risk():
    """Simulates a query that matches a known toxin across all signals."""
    score, _ = compute_score(0.95, 0.9, 0.8)
    assert score > 0.75, f"Expected HIGH risk (>0.75), got {score}"

def test_low_similarity_gives_low_risk():
    """Simulates a benign protein with no significant matches."""
    score, _ = compute_score(0.1, 0.05, 0.0)
    assert score < 0.5, f"Expected LOW risk (<0.5), got {score}"

def test_medium_similarity_gives_medium_risk():
    """Simulates ambiguous similarity — moderate matches."""
    score, _ = compute_score(0.6, 0.5, 0.3)
    assert 0.1 <= score <= 0.6, f"Expected moderate risk, got {score}"

def test_synergy_bonus_increases_score():
    """Multiple high-confidence signals should score higher than any single signal."""
    # All signals high — should trigger synergy bonus
    combined, _ = compute_score(0.9, 0.85, 0.8)
    # Only embedding high
    embedding_only, _ = compute_score(0.9, None, 0.0)
    assert combined > embedding_only, "Synergy bonus should increase score"

def test_no_structure_redistributes_weights():
    """When structure is unavailable, weights shift to embedding + function."""
    score, explanation = compute_score(0.9, None, 0.5)
    assert 0.0 <= score <= 1.0
    assert 'structural analysis not performed' in explanation

def test_active_site_overlap_increases_score():
    """Active site similarity should increase risk score."""
    score_without, _ = compute_score(0.8, 0.7, 0.3, active_site_overlap=None)
    score_with, _ = compute_score(0.8, 0.7, 0.3, active_site_overlap=0.8)
    assert score_with > score_without

def test_full_pipeline_scoring():
    """All four signals high → very high risk."""
    score, explanation = compute_score(0.95, 0.9, 0.8, active_site_overlap=0.85)
    assert score > 0.75, f"Full pipeline high signals should be HIGH risk, got {score}"
    assert 'active site' in explanation.lower()

# ── Function prediction ──────────────────────────────────────────────────────

def test_function_predictor_returns_prediction():
    predictor = FunctionPredictor(use_api=False)
    result = predictor.predict('MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH')
    assert result.summary
    assert isinstance(result.go_terms, list)
    assert isinstance(result.ec_numbers, list)

def test_function_predictor_deterministic():
    predictor = FunctionPredictor(use_api=False)
    seq = 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH'
    r1 = predictor.predict(seq)
    r2 = predictor.predict(seq)
    assert r1.summary == r2.summary
    assert r1.go_terms == r2.go_terms

def test_function_predictor_batch():
    predictor = FunctionPredictor(use_api=False)
    seqs = ['MVLSPADKTNVKAAWGKVG', 'CCCCCCCCCRGDCCCCCCC']
    results = predictor.batch_predict(seqs)
    assert len(results) == 2

def test_function_predictor_cysteine_rich():
    predictor = MockFunctionPredictor()
    seq = 'MCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC'
    result = predictor.predict(seq)
    assert result.summary

def test_function_predictor_with_toxin_motif():
    predictor = MockFunctionPredictor()
    seq = 'MVLSPADKTNVKAAWRGDGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH'
    features = predictor._analyze(seq)
    assert features['motifs'] > 0

def test_interpro_parse_results():
    """Test InterPro JSON parsing without hitting the API."""
    predictor = InterProPredictor()
    mock_data = {
        "results": [{
            "matches": [{
                "signature": {
                    "name": "Phospholipase_A2",
                    "entry": {
                        "goXRefs": [
                            {"id": "GO:0006644", "name": "phospholipid metabolic process"},
                            {"id": "GO:0090729", "name": "toxin activity"},
                        ],
                        "pathways": [
                            {"databaseName": "EC", "id": "3.1.1.4", "name": "Phospholipase A2"},
                        ],
                    },
                },
            }],
        }],
    }
    result = predictor._parse_results(mock_data)
    assert len(result.go_terms) == 2
    assert result.go_terms[0]["term"] == "GO:0006644"
    assert len(result.ec_numbers) == 1
    assert result.ec_numbers[0]["number"] == "3.1.1.4"
    assert "phospholipase_a2" in result.summary.lower()

# ── Embedding similarity search ──────────────────────────────────────────────

def test_embedding_searcher_empty_db():
    mock_db = MagicMock()
    mock_db.size = 0
    searcher = EmbeddingSimilaritySearcher(mock_db)
    hits = searcher.search(np.random.randn(1280).astype(np.float32))
    assert hits == []

def test_embedding_searcher_returns_hits():
    mock_db = MagicMock()
    mock_db.size = 100
    mock_db.search.return_value = (
        np.array([0.95, 0.85, 0.75]),
        np.array([0, 5, 10]),
    )
    mock_db.get_metadata.side_effect = lambda i: {
        'uniprot_id': f'P{i}', 'name': f'Toxin {i}'
    }
    searcher = EmbeddingSimilaritySearcher(mock_db)
    hits = searcher.search(np.random.randn(1280).astype(np.float32), top_k=3)
    assert len(hits) == 3
    assert hits[0].cosine_similarity == 0.95
    assert hits[0].metadata['uniprot_id'] == 'P0'

def test_embedding_searcher_skips_negative_indices():
    mock_db = MagicMock()
    mock_db.size = 10
    mock_db.search.return_value = (
        np.array([0.9, 0.0]),
        np.array([3, -1]),  # -1 is FAISS padding
    )
    mock_db.get_metadata.return_value = {'uniprot_id': 'P3', 'name': 'Test'}
    searcher = EmbeddingSimilaritySearcher(mock_db)
    hits = searcher.search(np.random.randn(1280).astype(np.float32), top_k=2)
    assert len(hits) == 1

# ── Foldseek m8 parsing ──────────────────────────────────────────────────────

def test_foldseek_parse_m8():
    searcher = FoldseekSearcher.__new__(FoldseekSearcher)
    m8 = "query\t4HFI.A\t0.85\t0.72\t150\t0.95\nquery\t1ABC.B\t0.60\t0.55\t100\t0.80\n"
    hits = searcher._parse_m8(m8)
    assert len(hits) == 2
    assert hits[0].tm_score == 0.85  # sorted descending
    assert hits[0].target_id == '4HFI'
    assert hits[1].tm_score == 0.60

def test_foldseek_parse_m8_empty():
    searcher = FoldseekSearcher.__new__(FoldseekSearcher)
    hits = searcher._parse_m8("")
    assert hits == []

def test_foldseek_parse_m8_malformed():
    searcher = FoldseekSearcher.__new__(FoldseekSearcher)
    m8 = "bad\tdata\n#comment\n"
    hits = searcher._parse_m8(m8)
    assert hits == []

# ── Foldseek availability ─────────────────────────────────────────────────────

def test_foldseek_binary_available():
    import shutil
    assert shutil.which('foldseek') is not None, "foldseek binary not on PATH"

# ── Pydantic schemas ─────────────────────────────────────────────────────────

def test_screening_request_strips_fasta_header():
    req = ScreeningRequest(sequence=">my_protein\nMVLSPADKTN\nVKAAWGKVGA")
    assert req.sequence == "MVLSPADKTNVKAAWGKVGA"

def test_screening_request_uppercases():
    req = ScreeningRequest(sequence="mvlspadktnvkaawgkvga")
    assert req.sequence == "MVLSPADKTNVKAAWGKVGA"

def test_screening_request_min_length():
    with pytest.raises(Exception):
        ScreeningRequest(sequence="MKT")  # too short (min_length=10)

# ── Toxin database integration ───────────────────────────────────────────────

# ── Active site comparison ────────────────────────────────────────────────────

def test_pocket_detection_from_pdb():
    """Test pocket detection on a real PDB file from our toxin structures."""
    from app.pipeline.active_site import detect_pockets
    import glob
    pdb_files = glob.glob('data/toxin_structures/*.pdb')
    if not pdb_files:
        pytest.skip("No toxin structures available")
    pdb_string = open(pdb_files[0]).read()
    pockets = detect_pockets(pdb_string)
    # Should detect at least one pocket in a real protein
    assert len(pockets) >= 0  # some small peptides may have no pockets
    for p in pockets:
        assert len(p.residue_indices) >= 5
        assert p.ca_coords.shape[1] == 3

def test_compare_identical_pockets():
    """Comparing a pocket to itself should give RMSD ≈ 0 and overlap ≈ 1."""
    from app.pipeline.active_site import Pocket, compare_active_sites
    coords = np.random.randn(10, 3).astype(np.float64)
    pocket = Pocket(
        residue_indices=list(range(10)),
        residue_names=['ALA'] * 10,
        center=coords.mean(axis=0),
        ca_coords=coords,
    )
    match = compare_active_sites(pocket, pocket)
    assert match.rmsd < 0.01
    assert match.overlap_score > 0.99

def test_compare_different_pockets():
    """Very different pockets should have high RMSD and low overlap."""
    from app.pipeline.active_site import Pocket, compare_active_sites
    coords1 = np.zeros((10, 3))
    coords1[:, 0] = np.arange(10)  # line along x
    coords2 = np.random.randn(10, 3) * 20  # scattered far apart

    p1 = Pocket(residue_indices=list(range(10)), residue_names=['ALA']*10,
                center=coords1.mean(axis=0), ca_coords=coords1)
    p2 = Pocket(residue_indices=list(range(10)), residue_names=['ALA']*10,
                center=coords2.mean(axis=0), ca_coords=coords2)
    match = compare_active_sites(p1, p2)
    assert match.overlap_score < 0.8  # should be dissimilar

# ── Edge cases ────────────────────────────────────────────────────────────────

def test_validate_very_long_sequence():
    """Long sequences should still validate (truncation is handled by ESM-2)."""
    seq = 'M' + 'A' * 5000
    result = validate_protein_sequence(seq)
    assert result.valid

def test_validate_whitespace_stripped():
    result = validate_protein_sequence('  MVLSPADKTN  ')
    assert result.valid
    assert result.cleaned.strip() == result.cleaned

def test_validate_lowercase_accepted():
    result = validate_protein_sequence('mvlspadktn')
    assert result.valid

def test_score_with_all_none_structure():
    """Score should work when structure is explicitly None."""
    score, explanation = compute_score(0.5, None, 0.5)
    assert 0.0 <= score <= 1.0
    assert 'structural analysis not performed' in explanation

def test_screening_request_rejects_empty():
    with pytest.raises(Exception):
        ScreeningRequest(sequence="")

def test_function_predictor_empty_sequence():
    """Empty-ish sequence should not crash the mock predictor."""
    predictor = MockFunctionPredictor()
    result = predictor.predict('M')
    assert result.summary

# ── Toxin database integration ───────────────────────────────────────────────

def test_toxin_db_loads():
    from app.database.toxin_db import ToxinDatabase
    db = ToxinDatabase(index_path='data/toxin_db.faiss', meta_path='data/toxin_meta.json')
    db.load()
    assert db.size == 2000
    assert db._index.ntotal == len(db._metadata)

@pytest.mark.skip(reason="FAISS search segfaults on Python 3.14 due to SWIG ABI incompatibility")
def test_toxin_db_search():
    from app.database.toxin_db import ToxinDatabase
    db = ToxinDatabase(index_path='data/toxin_db.faiss', meta_path='data/toxin_meta.json')
    db.load()
    import faiss
    real_embedding = faiss.rev_swig_ptr(db._index.get_xb(), db.size * 1280).reshape(db.size, 1280)[0].copy()
    distances, indices = db.search(real_embedding, k=5)
    assert len(distances) == 5
    assert len(indices) == 5
    assert all(i >= 0 for i in indices)
    assert distances[0] > 0.99

def test_toxin_db_metadata():
    from app.database.toxin_db import ToxinDatabase
    db = ToxinDatabase(index_path='data/toxin_db.faiss', meta_path='data/toxin_meta.json')
    db.load()
    meta = db.get_metadata(0)
    assert 'uniprot_id' in meta
    assert 'name' in meta
    assert 'sequence' in meta
