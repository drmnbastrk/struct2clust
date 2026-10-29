"""Descriptive structural characterisation of the clusters.

Everything here describes clusters that were already defined from the same
structures, so no p-values are reported: testing differences between groups
that were chosen to be different would be circular (Gao, Bien & Witten 2024).
Effect sizes (RMSF, contact frequencies, circular torsion statistics and the
separation of torsions between clusters) are reported instead.

All analyses use every heavy atom of the ensemble, independent of the atom
selection used for clustering, except the comparison of the cluster medoids,
which explains the clustering distances and therefore uses the same atoms and
the same superposition.
Noise structures are not part of any cluster.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from struct2clust.metrics import kabsch_transform, superpose
from struct2clust.structures import (
    LINK_ATOMS,
    LINK_MAX_DISTANCE,
    Atoms,
    Ensemble,
    canonical_resname,
)

logger = logging.getLogger(__name__)


def residue_table(atoms: Atoms) -> pd.DataFrame:
    """One row per residue position: position (1-based), chain, resid, resname, kind."""
    first = np.unique(atoms.residue, return_index=True)[1]
    return pd.DataFrame({
        "residue": atoms.residue[first] + 1,
        "chain": atoms.chain[first],
        "resid": atoms.resid[first],
        "resname": atoms.resname[first],
        "kind": atoms.kind[first],
    })


# ---------------------------------------------------------------------------
# Superposition, RMSF, radius of gyration
# ---------------------------------------------------------------------------

def superpose_on_mean(coords: np.ndarray, start: int = 0, iterations: int = 10) -> np.ndarray:
    """Iteratively superpose all members onto their evolving mean structure."""
    ref = coords[start]
    aligned = coords.copy()
    for _ in range(iterations):
        for k, xyz in enumerate(coords):
            r, t = kabsch_transform(xyz, ref)
            aligned[k] = xyz @ r.T + t
        new_ref = aligned.mean(axis=0)
        if np.sqrt(((new_ref - ref) ** 2).sum(axis=1).mean()) < 1e-4:
            break
        ref = new_ref
    return aligned


def rmsf_per_residue(aligned: np.ndarray, residue: np.ndarray) -> np.ndarray:
    """Root-mean-square fluctuation per residue (A), averaged over its heavy atoms."""
    sq = ((aligned - aligned.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0)  # per atom
    n_res = residue.max() + 1
    return np.sqrt(np.bincount(residue, sq, n_res) / np.bincount(residue, minlength=n_res))


def radius_of_gyration(coords: np.ndarray) -> np.ndarray:
    centred = coords - coords.mean(axis=1, keepdims=True)
    return np.sqrt((centred ** 2).sum(axis=2).mean(axis=1))


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

def residue_contacts(xyz: np.ndarray, residue: np.ndarray, cutoff: float, min_sep: int):
    """Set of residue pairs (i < j, j - i >= min_sep) with any heavy-atom pair within cutoff."""
    pairs = cKDTree(xyz).query_pairs(cutoff, output_type="ndarray")
    ri, rj = residue[pairs[:, 0]], residue[pairs[:, 1]]
    lo, hi = np.minimum(ri, rj), np.maximum(ri, rj)
    keep = hi - lo >= min_sep
    return np.unique(lo[keep] * (residue.max() + 1) + hi[keep])


def contact_frequencies(coords, residue, cutoff, min_sep) -> np.ndarray:
    """(n_res, n_res) fraction of structures in which each residue pair is in contact."""
    n_res = residue.max() + 1
    counts = np.zeros(n_res * n_res)
    for xyz in coords:
        counts[residue_contacts(xyz, residue, cutoff, min_sep)] += 1
    freq = (counts / len(coords)).reshape(n_res, n_res)
    return freq + freq.T


# ---------------------------------------------------------------------------
# Torsions
# ---------------------------------------------------------------------------

# (name, [(residue offset, atom name) x 4]); chi depends on the base, see _chi_atoms
NUCLEIC_TORSIONS = [
    ("alpha", [(-1, "O3'"), (0, "P"), (0, "O5'"), (0, "C5'")]),
    ("beta", [(0, "P"), (0, "O5'"), (0, "C5'"), (0, "C4'")]),
    ("gamma", [(0, "O5'"), (0, "C5'"), (0, "C4'"), (0, "C3'")]),
    ("delta", [(0, "C5'"), (0, "C4'"), (0, "C3'"), (0, "O3'")]),
    ("epsilon", [(0, "C4'"), (0, "C3'"), (0, "O3'"), (1, "P")]),
    ("zeta", [(0, "C3'"), (0, "O3'"), (1, "P"), (1, "O5'")]),
]
PROTEIN_TORSIONS = [
    ("phi", [(-1, "C"), (0, "N"), (0, "CA"), (0, "C")]),
    ("psi", [(0, "N"), (0, "CA"), (0, "C"), (1, "N")]),
]
_PURINES = ("A", "G", "I", "DA", "DG", "DI")


def _chi_atoms(resname: str):
    """Chi: O4'-C1'-N9-C4 for purines, O4'-C1'-N1-C2 for pyrimidines (IUPAC-IUB 1983).
    Force-field names (G5, AN, RA, ADE, ...) are unified first."""
    if canonical_resname(resname) in _PURINES:
        return [(0, "O4'"), (0, "C1'"), (0, "N9"), (0, "C4")]
    return [(0, "O4'"), (0, "C1'"), (0, "N1"), (0, "C2")]


def dihedral(p: np.ndarray) -> np.ndarray:
    """Dihedral angles in degrees for (n, 4, 3) point quadruples."""
    b0, b1, b2 = p[:, 0] - p[:, 1], p[:, 2] - p[:, 1], p[:, 3] - p[:, 2]
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1).sum(1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(1, keepdims=True) * b1
    x = (v * w).sum(1)
    y = (np.cross(b1, v) * w).sum(1)
    return np.degrees(np.arctan2(y, x))


def linked_to_next(ens: Ensemble, max_distance: float = LINK_MAX_DISTANCE) -> np.ndarray:
    """Per residue position: covalently linked to the next position (C-N or O3'-P
    shorter than ``max_distance`` in the median member). Positions on either side of residues
    missing from the structures are not linked."""
    atoms = ens.atoms
    index = {(int(r), str(n)): i for i, (r, n) in enumerate(zip(atoms.residue, atoms.name))}
    res = residue_table(atoms)
    out = np.zeros(len(res), dtype=bool)
    for r in range(len(res) - 1):
        kind = res.loc[r, "kind"]
        if kind != res.loc[r + 1, "kind"] or res.loc[r, "chain"] != res.loc[r + 1, "chain"]:
            continue
        first, second = LINK_ATOMS[kind]
        i, j = index.get((r, first)), index.get((r + 1, second))
        if i is not None and j is not None:
            d = np.linalg.norm(ens.coords[:, i] - ens.coords[:, j], axis=1)
            out[r] = np.median(d) < max_distance
    return out


def torsion_angles(ens: Ensemble, bond_max_distance: float = LINK_MAX_DISTANCE
                   ) -> pd.DataFrame:
    """Long table: member, residue (1-based), torsion, angle (degrees).

    A torsion that involves a neighbouring residue is computed only if the two
    residues are covalently linked, so no angle spans a gap of missing residues."""
    atoms = ens.atoms
    index = {(int(r), str(n)): i for i, (r, n) in enumerate(zip(atoms.residue, atoms.name))}
    res = residue_table(atoms)
    linked = linked_to_next(ens, bond_max_distance)
    frames = []
    for _, row in res.iterrows():
        r = int(row["residue"]) - 1
        defs = NUCLEIC_TORSIONS + [("chi", _chi_atoms(row["resname"]))] \
            if row["kind"] == "nucleic" else PROTEIN_TORSIONS
        for name, spec in defs:
            idx = []
            for offset, atom in spec:
                rr = r + offset
                ok = (0 <= rr < len(res) and (offset == 0 or linked[min(r, rr)]))
                i = index.get((rr, atom)) if ok else None
                if i is None:
                    break
                idx.append(i)
            if len(idx) < 4:
                continue
            angles = dihedral(ens.coords[:, idx])
            frames.append(pd.DataFrame({
                "member": np.arange(len(ens)), "residue": r + 1, "torsion": name, "angle": angles,
            }))
    if not frames:
        return pd.DataFrame(columns=["member", "residue", "torsion", "angle"])
    return pd.concat(frames, ignore_index=True)


def _resultant(angles_deg: np.ndarray) -> float:
    a = np.radians(angles_deg)
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def circular_separation(groups: list[np.ndarray]) -> float:
    """How strongly a torsion separates clusters, 0 (not at all) to 1.

    Size-weighted mean resultant length within clusters minus the resultant
    length of all angles pooled: the circular analogue of the between-group
    share of variance (circular variance is 1 - R). It is high only when every
    cluster is internally consistent *and* the clusters differ.
    """
    sizes = np.array([len(g) for g in groups], float)
    within = float(np.dot(sizes, [_resultant(g) for g in groups]) / sizes.sum())
    return max(0.0, within - _resultant(np.concatenate(groups)))


def _max_mean_difference(groups: list[np.ndarray]) -> float:
    means = [circular_summary(g)[0] for g in groups]
    return max(angle_difference(a, b) for i, a in enumerate(means) for b in means[i + 1:])


def circular_summary(angles_deg: np.ndarray) -> tuple[float, float]:
    """Circular mean (degrees) and circular standard deviation (degrees)."""
    a = np.radians(angles_deg)
    c, s = np.cos(a).mean(), np.sin(a).mean()
    r = np.clip(np.hypot(c, s), 1e-12, 1.0)
    return float(np.degrees(np.arctan2(s, c))), float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees (0-180)."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


# ---------------------------------------------------------------------------
# Where the clusters differ
# ---------------------------------------------------------------------------

def medoid_differences(ens: Ensemble, labels: np.ndarray, medoids: dict[int, int],
                       fit) -> pd.DataFrame:
    """Per-residue share of the squared deviation between every pair of cluster medoids.

    Each pair is superposed as for the clustering distance (``fit``: atoms ``mask``,
    least squares for RMSD or TM-score superposition for TM). The squared RMSD is
    the mean of the squared atomic deviations, so it splits exactly into the
    contributions of the residues: ``share`` is a
    residue's part of it (the shares of one pair add up to 1) and ``rmsd`` the RMSD
    of the residue's own atoms. Residue positions are 1-based.
    """
    mask = fit.mask
    residue = ens.atoms.residue[mask]
    n_res = ens.n_residues
    count = np.bincount(residue, minlength=n_res)
    present = np.flatnonzero(count > 0)
    rows = []
    k = int(labels.max()) + 1
    for a in range(k):
        for b in range(a + 1, k):
            xa, xb = ens.coords[medoids[a]][mask], ens.coords[medoids[b]][mask]
            r, t = superpose(xa, xb, fit.metric, fit.d0)
            sq = ((xa @ r.T + t - xb) ** 2).sum(axis=1)
            per_res = np.bincount(residue, sq, n_res)
            rows.append(pd.DataFrame({
                "cluster_a": a + 1, "cluster_b": b + 1, "residue": present + 1,
                "share": per_res[present] / max(sq.sum(), 1e-12),
                "rmsd": np.sqrt(per_res[present] / count[present]),
            }))
    if not rows:
        return pd.DataFrame(columns=["cluster_a", "cluster_b", "residue", "share", "rmsd"])
    return pd.concat(rows, ignore_index=True)


def main_residues(shares: pd.Series, fraction: float = 0.5) -> tuple[list[int], float]:
    """The fewest residues whose shares add up to at least ``fraction``.

    ``shares`` is indexed by residue position. Returns (positions in sequence
    order, their combined share).
    """
    s = shares.sort_values(ascending=False)
    n = int(np.searchsorted(s.cumsum().to_numpy(), fraction - 1e-12)) + 1
    top = s.iloc[:n]
    return sorted(int(r) for r in top.index), float(top.sum())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@dataclass
class Biology:
    residues: pd.DataFrame      # residue table
    rmsf: pd.DataFrame          # cluster, residue, rmsf
    rg: pd.DataFrame            # member, cluster, rg
    contacts: dict              # cluster (1-based) -> (n_res, n_res) frequency matrix
    torsions: pd.DataFrame      # cluster, residue, torsion, circular_mean, circular_sd, n
    torsion_spread: pd.DataFrame  # residue, torsion, separation, max_mean_difference


def characterise(
    ens: Ensemble,
    labels: np.ndarray,
    medoids: dict[int, int],
    contact_cutoff: float = 4.5,
    contact_min_separation: int = 2,
    bond_max_distance: float = LINK_MAX_DISTANCE,
) -> Biology:
    atoms = ens.atoms
    residues = residue_table(atoms)
    k = labels.max() + 1

    rmsf_rows, contacts = [], {}
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        start = int(np.flatnonzero(idx == medoids[c])[0])
        aligned = superpose_on_mean(ens.coords[idx], start=start)
        rmsf_rows.append(pd.DataFrame({
            "cluster": c + 1, "residue": residues["residue"],
            "rmsf": rmsf_per_residue(aligned, atoms.residue),
        }))
        contacts[c + 1] = contact_frequencies(
            ens.coords[idx], atoms.residue, contact_cutoff, contact_min_separation
        )
    rmsf = pd.concat(rmsf_rows, ignore_index=True)

    rg = pd.DataFrame({
        "member": np.arange(len(ens)), "cluster": np.where(labels >= 0, labels + 1, 0),
        "rg": radius_of_gyration(ens.coords),
    })

    tors = torsion_angles(ens, bond_max_distance)
    tors["cluster"] = labels[tors["member"].to_numpy()] + 1
    tors = tors[tors["cluster"] > 0]
    rows = []
    for (c, r, t), grp in tors.groupby(["cluster", "residue", "torsion"], sort=True):
        mean, sd = circular_summary(grp["angle"].to_numpy())
        rows.append({"cluster": c, "residue": r, "torsion": t,
                     "circular_mean": mean, "circular_sd": sd, "n": len(grp)})
    torsions = pd.DataFrame(rows)

    spread_rows = []
    if k >= 2:
        for (r, t), grp in tors.groupby(["residue", "torsion"], sort=True):
            per_cluster = [g["angle"].to_numpy() for _, g in grp.groupby("cluster")]
            if len(per_cluster) < 2:
                continue
            spread_rows.append({
                "residue": r, "torsion": t,
                "separation": circular_separation(per_cluster),
                "max_mean_difference": _max_mean_difference(per_cluster),
            })
    spread = pd.DataFrame(spread_rows)
    if not spread.empty:
        spread = spread.sort_values("separation", ascending=False, ignore_index=True)

    logger.info(
        "Biology: RMSF, contacts (cutoff %.1f A) and %d torsion series for %d clusters",
        contact_cutoff, tors.groupby(["residue", "torsion"]).ngroups, k,
    )
    return Biology(residues, rmsf, rg, contacts, torsions, spread)
