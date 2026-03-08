"""Active site detection and comparison for protein structures.

Identifies binding pockets in PDB structures and compares active site
geometry between a query protein and known toxin structures.
Uses BioPython for structure parsing and numpy for geometric comparison.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser, Selection
from loguru import logger


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Pocket:
    """A detected binding pocket / active site region."""

    residue_indices: list[int]
    residue_names: list[str]
    center: np.ndarray  # (3,) center of mass
    ca_coords: np.ndarray  # (N, 3) C-alpha coordinates
    volume_estimate: float = 0.0


@dataclass
class ActiveSiteMatch:
    """Result of comparing two active sites."""

    rmsd: float  # RMSD of aligned C-alpha atoms
    aligned_residues: int  # number of residues in alignment
    overlap_score: float  # 0-1 normalized similarity
    query_pocket: Pocket
    target_id: str = ""


# ── Pocket detection ─────────────────────────────────────────────────────────


def detect_pockets(
    pdb_string: str,
    distance_threshold: float = 8.0,
    min_pocket_size: int = 5,
    max_pockets: int = 3,
) -> list[Pocket]:
    """Detect binding pockets from a PDB structure using geometric clustering.

    Strategy: find clusters of residues with high local contact density
    (many nearby residues within threshold), which indicates a cavity/pocket.

    Args:
        pdb_string: PDB format string.
        distance_threshold: Max distance (Å) between C-alpha atoms to be neighbors.
        min_pocket_size: Minimum residues to count as a pocket.
        max_pockets: Maximum number of pockets to return.

    Returns:
        List of Pocket objects sorted by size (largest first).
    """
    structure = _parse_pdb_string(pdb_string)
    if structure is None:
        return []

    # Extract C-alpha atoms
    ca_atoms = []
    residue_info = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue  # skip heteroatoms
                if "CA" in residue:
                    ca_atoms.append(residue["CA"].get_vector().get_array())
                    residue_info.append((residue.id[1], residue.get_resname()))
        break  # first model only

    if len(ca_atoms) < min_pocket_size:
        return []

    # Skip very short proteins — pocket detection is unreliable below ~40 residues
    if len(ca_atoms) < 40:
        return []

    coords = np.array(ca_atoms)
    n = len(coords)

    # Compute pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    # Contact density: number of neighbors within threshold for each residue
    contact_counts = (dist_matrix < distance_threshold).sum(axis=1) - 1  # exclude self

    # Find pocket seeds: residues with high contact density (top 30%)
    density_threshold = np.percentile(contact_counts, 70)
    seed_mask = contact_counts >= density_threshold

    if seed_mask.sum() < min_pocket_size:
        return []

    # Cluster seeds into pockets using connected components
    seed_indices = np.where(seed_mask)[0]
    pockets = _cluster_residues(seed_indices, dist_matrix, distance_threshold)

    # Build Pocket objects
    result = []
    for pocket_indices in pockets:
        if len(pocket_indices) < min_pocket_size:
            continue

        pocket_coords = coords[pocket_indices]
        center = pocket_coords.mean(axis=0)
        names = [residue_info[i][1] for i in pocket_indices]
        indices = [residue_info[i][0] for i in pocket_indices]

        # Estimate volume from convex hull approximation
        spread = pocket_coords - center
        volume = np.prod(np.ptp(spread, axis=0))  # bounding box volume

        result.append(Pocket(
            residue_indices=indices,
            residue_names=names,
            center=center,
            ca_coords=pocket_coords,
            volume_estimate=float(volume),
        ))

    # Sort by size (largest pocket first)
    result.sort(key=lambda p: len(p.residue_indices), reverse=True)
    return result[:max_pockets]


def _cluster_residues(
    indices: np.ndarray,
    dist_matrix: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Simple connected-component clustering of residue indices."""
    index_set = set(indices.tolist())
    visited = set()
    clusters = []

    for idx in indices:
        if idx in visited:
            continue
        cluster = []
        stack = [idx]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            # Find neighbors
            for other in index_set:
                if other not in visited and dist_matrix[current, other] < threshold:
                    stack.append(other)
        if cluster:
            clusters.append(sorted(cluster))

    return sorted(clusters, key=len, reverse=True)


# ── Active site comparison ───────────────────────────────────────────────────


def compare_active_sites(
    query_pocket: Pocket,
    target_pocket: Pocket,
) -> ActiveSiteMatch:
    """Compare two active site pockets by geometric RMSD after optimal alignment.

    Uses the Kabsch algorithm (SVD-based) to find the optimal rotation
    that minimizes RMSD between the two sets of C-alpha coordinates.

    Args:
        query_pocket: Pocket from the query structure.
        target_pocket: Pocket from a known toxin structure.

    Returns:
        ActiveSiteMatch with RMSD and overlap score.
    """
    q_coords = query_pocket.ca_coords
    t_coords = target_pocket.ca_coords

    # Use the smaller pocket size for alignment
    n = min(len(q_coords), len(t_coords))
    if n < 3:
        return ActiveSiteMatch(
            rmsd=float("inf"),
            aligned_residues=n,
            overlap_score=0.0,
            query_pocket=query_pocket,
        )

    # Truncate to same size (use first N from each — they're sorted by position)
    q = q_coords[:n].copy()
    t = t_coords[:n].copy()

    # Center both
    q_center = q.mean(axis=0)
    t_center = t.mean(axis=0)
    q -= q_center
    t -= t_center

    # Kabsch algorithm: find optimal rotation via SVD
    H = q.T @ t
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    q_rotated = (R @ q.T).T
    rmsd = np.sqrt(((q_rotated - t) ** 2).sum(axis=1).mean())

    # Convert RMSD to a 0-1 similarity score
    # RMSD < 1.0 Å is very similar, > 5.0 Å is dissimilar
    overlap_score = max(0.0, 1.0 - rmsd / 5.0)

    return ActiveSiteMatch(
        rmsd=float(rmsd),
        aligned_residues=n,
        overlap_score=float(overlap_score),
        query_pocket=query_pocket,
    )


def compute_active_site_score(
    query_pdb: str,
    target_pdbs: dict[str, str],
    top_k: int = 5,
) -> list[ActiveSiteMatch]:
    """Compare query structure's active sites against multiple target structures.

    Args:
        query_pdb: PDB string of the query protein.
        target_pdbs: Dict of {target_id: pdb_string} for known toxins.
        top_k: Number of best matches to return.

    Returns:
        List of ActiveSiteMatch sorted by overlap_score (best first).
    """
    query_pockets = detect_pockets(query_pdb)
    if not query_pockets:
        logger.debug("No pockets detected in query structure")
        return []

    all_matches = []

    for target_id, target_pdb in target_pdbs.items():
        target_pockets = detect_pockets(target_pdb)
        if not target_pockets:
            continue

        # Compare each query pocket against each target pocket, keep best
        best_match = None
        for qp in query_pockets:
            for tp in target_pockets:
                match = compare_active_sites(qp, tp)
                match.target_id = target_id
                if best_match is None or match.overlap_score > best_match.overlap_score:
                    best_match = match

        if best_match and best_match.overlap_score > 0:
            all_matches.append(best_match)

    all_matches.sort(key=lambda m: m.overlap_score, reverse=True)
    return all_matches[:top_k]


# ── Structural superposition ─────────────────────────────────────────────────


@dataclass
class SuperpositionResult:
    """Result of superimposing two full protein structures."""

    query_pdb: str  # unchanged query PDB string
    aligned_target_pdb: str  # target PDB with coordinates aligned to query frame
    rmsd: float  # C-alpha RMSD after alignment
    aligned_residues: int  # number of C-alpha pairs used


def superimpose_structures(
    query_pdb: str,
    target_pdb: str,
) -> SuperpositionResult | None:
    """Superimpose a target structure onto a query structure using Kabsch alignment.

    Aligns the target's C-alpha atoms to the query's, then applies the same
    rotation + translation to all atoms in the target structure. Returns a new
    PDB string for the target with transformed coordinates.

    Args:
        query_pdb: PDB format string for the query (reference) protein.
        target_pdb: PDB format string for the target (toxin) protein.

    Returns:
        SuperpositionResult, or None if alignment fails.
    """
    query_struct = _parse_pdb_string(query_pdb)
    target_struct = _parse_pdb_string(target_pdb)
    if query_struct is None or target_struct is None:
        logger.warning("Failed to parse one or both PDB strings for superposition")
        return None

    # Extract C-alpha coordinates from first model of each structure
    q_ca = []
    for model in query_struct:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    q_ca.append(residue["CA"].get_vector().get_array())
        break

    t_ca = []
    for model in target_struct:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" in residue:
                    t_ca.append(residue["CA"].get_vector().get_array())
        break

    if len(q_ca) < 3 or len(t_ca) < 3:
        logger.warning("Too few C-alpha atoms for superposition (query={}, target={})", len(q_ca), len(t_ca))
        return None

    # Use the shorter length for alignment
    n = min(len(q_ca), len(t_ca))
    q_coords = np.array(q_ca[:n]).copy()
    t_coords = np.array(t_ca[:n]).copy()

    # Center both
    q_center = q_coords.mean(axis=0)
    t_center = t_coords.mean(axis=0)
    q_centered = q_coords - q_center
    t_centered = t_coords - t_center

    # Kabsch algorithm: find optimal rotation via SVD
    H = t_centered.T @ q_centered
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    R = Vt.T @ sign_matrix @ U.T

    # Compute RMSD
    t_rotated = (R @ t_centered.T).T
    rmsd = float(np.sqrt(((t_rotated - q_centered) ** 2).sum(axis=1).mean()))

    # Apply rotation + translation to ALL atoms in the target structure
    for model in target_struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = atom.get_vector().get_array()
                    # Translate to origin, rotate, translate to query frame
                    new_coord = R @ (coord - t_center) + q_center
                    atom.set_coord(new_coord)

    # Write aligned target structure to PDB string
    aligned_pdb = _structure_to_pdb_string(target_struct)
    if aligned_pdb is None:
        return None

    return SuperpositionResult(
        query_pdb=query_pdb,
        aligned_target_pdb=aligned_pdb,
        rmsd=rmsd,
        aligned_residues=n,
    )


def _structure_to_pdb_string(structure) -> str | None:
    """Write a BioPython Structure object to a PDB format string."""
    try:
        from Bio.PDB import PDBIO
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            pdb_io.save(f.name)
            tmp_path = Path(f.name)
        pdb_string = tmp_path.read_text()
        tmp_path.unlink(missing_ok=True)
        return pdb_string
    except Exception as e:
        logger.warning("Failed to write PDB string: {}", e)
        return None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_pdb_string(pdb_string: str):
    """Parse a PDB format string into a BioPython Structure object."""
    parser = PDBParser(QUIET=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_string)
            f.flush()
            structure = parser.get_structure("query", f.name)
        Path(f.name).unlink(missing_ok=True)
        return structure
    except Exception as e:
        logger.warning("Failed to parse PDB: {}", e)
        return None
