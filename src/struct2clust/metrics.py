"""Pairwise distance matrices.

rmsd     Kabsch RMSD after optimal superposition (Angstrom); optionally minimised
         over the naming of chemically equivalent atoms.
tm       1 - TM-score from US-align (Zhang et al. 2022), residue correspondence
         fixed by residue index (``-TMscore 1``; ``-TMscore 2`` for multi-chain).
"""
from __future__ import annotations

import itertools
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from struct2clust.structures import Atoms, Ensemble, canonical_resname, write_pdb

logger = logging.getLogger(__name__)


def _n_workers(n_jobs: Optional[int]) -> int:
    """Threads for the distance matrix: all usable cores for None, -1 or 0."""
    from struct2clust.resources import cpu_count

    cpus = cpu_count()
    return cpus if n_jobs in (None, -1, 0) else max(1, min(n_jobs, cpus))


# ---------------------------------------------------------------------------
# Kabsch
# ---------------------------------------------------------------------------

def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t such that ``mobile @ R.T + t`` best fits ``target``."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, tc - mc @ r.T


def tm_d0(n: int, molecule_type: str) -> float:
    """d0 of the TM-score for n residues: proteins (and complexes) Zhang & Skolnick
    (2004), nucleic acids Gong et al. (2019)."""
    if molecule_type == "nucleic":
        if n >= 30:
            return 0.6 * np.sqrt(n - 0.5) - 2.5
        return 0.3 if n < 12 else 0.4 if n < 16 else 0.5 if n < 20 else 0.6 if n < 24 else 0.7
    return max(0.5, 1.24 * np.cbrt(max(n - 15, 0)) - 1.8)


def tm_superposition(mobile: np.ndarray, target: np.ndarray, d0: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t that maximise the TM-score of ``mobile @ R.T + t``
    against ``target`` (residues paired by index; one anchor atom per residue).

    The search of the TM-score program (Zhang & Skolnick 2004): superpositions of
    fragments of length L, L/2, L/4, ... are each refined by superposing again on
    the residues closer than a search cutoff, and the best TM-score is kept.
    """
    n = len(mobile)
    cutoff = min(max(d0, 4.5), 8.0)
    best, best_rt = -1.0, kabsch_transform(mobile, target)
    length = n
    while True:
        step = max(1, length // 2)
        for start in range(0, n - length + 1, step):
            idx = np.arange(start, start + length)
            for _ in range(20):
                r, t = kabsch_transform(mobile[idx], target[idx])
                d = np.linalg.norm(mobile @ r.T + t - target, axis=1)
                score = float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))
                if score > best:
                    best, best_rt = score, (r, t)
                cut = cutoff
                new = np.flatnonzero(d < cut)
                while len(new) < 3 and cut < 4 * cutoff:
                    cut += 0.5
                    new = np.flatnonzero(d < cut)
                if len(new) < 3 or np.array_equal(new, idx):
                    break
                idx = new
        if length <= 4:
            break
        length = max(4, length // 2)
    return best_rt


def superpose(mobile: np.ndarray, target: np.ndarray, metric: str = "rmsd",
              d0: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Superposition as the metric defines it: least squares (Kabsch) for rmsd,
    TM-score maximisation for tm (``d0`` required)."""
    if metric == "tm":
        return tm_superposition(mobile, target, d0)
    return kabsch_transform(mobile, target)


def _rmsd_row(centered: np.ndarray, sq_norms: np.ndarray, i: int) -> np.ndarray:
    """RMSD of member i to members i+1.. from the singular values of the covariance.

    Uses RMSD^2 = (|A|^2 + |B|^2 - 2 * sum(sigma_k)) / M, with the smallest
    singular value sign-flipped when the optimal transform would be a reflection
    (Kabsch 1976); no rotation matrix has to be formed.
    """
    others = centered[i + 1:]
    h = np.einsum("kad,ae->kde", others, centered[i])
    sigma = np.linalg.svd(h, compute_uv=False)
    det_sign = np.sign(np.linalg.det(h))
    sigma[:, -1] *= np.where(det_sign == 0, 1.0, det_sign)
    msd = (sq_norms[i + 1:] + sq_norms[i] - 2.0 * sigma.sum(axis=1)) / centered.shape[1]
    return np.sqrt(np.maximum(msd, 0.0))


def rmsd_matrix(coords: np.ndarray, n_jobs: int = -1, groups: Optional[list] = None
                ) -> np.ndarray:
    """All-pairs Kabsch RMSD for an (n, M, 3) coordinate array; with ``groups``
    (equivalent_atom_groups), minimised over the naming of equivalent atoms."""
    n = coords.shape[0]
    centered = coords - coords.mean(axis=1, keepdims=True)
    sq_norms = np.einsum("kad,kad->k", centered, centered)
    out = np.zeros((n, n))
    if groups:
        def row_of(i):
            return i, _equivalent_rmsd_row(centered, sq_norms, i, groups)
    else:
        def row_of(i):
            return i, _rmsd_row(centered, sq_norms, i)
    with ThreadPoolExecutor(_n_workers(n_jobs)) as pool:
        for i, row in pool.map(row_of, range(n - 1)):
            out[i, i + 1:] = row
            out[i + 1:, i] = row
    return out


# ---------------------------------------------------------------------------
# Chemically equivalent atoms
# ---------------------------------------------------------------------------

# Atoms that the heavy-atom graph of a residue cannot tell apart: the automorphisms
# of the graph with elements as labels, hydrogens and stereochemistry ignored
# (checked against the PDB Chemical Component Dictionary by
# validation/equivalent_atoms.py). Each group lists atom pairs that change places
# together.
EQUIVALENT_ATOMS = {
    "ARG": [[("NH1", "NH2")]],
    "ASP": [[("OD1", "OD2")]],
    "GLU": [[("OE1", "OE2")]],
    "LEU": [[("CD1", "CD2")]],
    "VAL": [[("CG1", "CG2")]],
    "PHE": [[("CD1", "CD2"), ("CE1", "CE2")]],
    "TYR": [[("CD1", "CD2"), ("CE1", "CE2")]],
}
C_TERMINUS = ("O", "OXT")             # free carboxylate of a chain's last amino acid
PHOSPHATE = ("OP1", "OP2")            # non-bridging phosphate oxygens
TERMINAL_PHOSPHATE = ("OP1", "OP2", "OP3")   # 5'-terminal phosphate


def _pair_group(pairs) -> tuple[tuple[str, ...], list[tuple[int, ...]]]:
    names = tuple(a for pair in pairs for a in pair)
    swapped = tuple(i ^ 1 for i in range(len(names)))      # (0 1)(2 3)...
    return names, [tuple(range(len(names))), swapped]


def equivalent_atom_groups(atoms: Atoms) -> list[tuple[np.ndarray, np.ndarray]]:
    """Groups of chemically equivalent atoms present in ``atoms``, batched by shape.

    Returns [(index, orders), ...]: ``index`` (g, s) holds the atom indices of g
    groups of s atoms, ``orders`` (a, s) the a admissible orders of a group's
    positions (the first is the identity). Groups with an atom missing are left out.
    """
    position = {(int(r), str(n)): i for i, (r, n) in enumerate(zip(atoms.residue, atoms.name))}
    batches: dict[tuple, list] = {}
    for r in np.unique(atoms.residue):
        first = int(np.flatnonzero(atoms.residue == r)[0])
        here = {str(n) for n in atoms.name[atoms.residue == r]}
        groups = []
        if atoms.kind[first] == "protein":
            groups += [_pair_group(g) for g in
                       EQUIVALENT_ATOMS.get(canonical_resname(str(atoms.resname[first])), [])]
            if set(C_TERMINUS) <= here:
                groups.append(_pair_group([C_TERMINUS]))
        elif set(TERMINAL_PHOSPHATE) <= here:
            groups.append((TERMINAL_PHOSPHATE, list(itertools.permutations(range(3)))))
        elif set(PHOSPHATE) <= here:
            groups.append(_pair_group([PHOSPHATE]))
        for names, orders in groups:
            if set(names) <= here:
                batches.setdefault(tuple(orders), []).append(
                    [position[(int(r), a)] for a in names])
    return [(np.array(index), np.array(orders)) for orders, index in batches.items()]


MAX_RENAMING_ROUNDS = 50


def _equivalent_rmsd_row(centered: np.ndarray, sq_norms: np.ndarray, i: int,
                         groups: list) -> np.ndarray:
    """RMSD of member i to members i+1.., minimised over the naming of equivalent
    atoms by alternating superposition and renaming.

    Each round superposes every member on member i and then, group by group, gives
    the member's equivalent atoms the order closest to member i. Both steps never
    increase the deviation and a renaming is made only if it lowers it, so the
    rounds end at a naming that no single group can improve.
    """
    x = centered[i]
    y = centered[i + 1:]
    k, m = y.shape[:2]
    perm = np.tile(np.arange(m), (k, 1))
    rows = np.arange(k)[:, None, None]
    for _ in range(MAX_RENAMING_ROUNDS):
        yp = y[np.arange(k)[:, None], perm]
        h = np.einsum("kad,ae->kde", yp, x)
        u, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
        u[:, :, -1] *= d[:, None]
        rot = np.einsum("kji,klj->kil", vt, u)          # R = V diag(1, 1, d) U^T
        z = np.einsum("kad,ked->kae", yp, rot)
        changed = False
        new = perm.copy()
        for index, orders in groups:
            options = index[:, orders]                   # (g, a, s) positions
            cost = ((z[:, options] - x[index][None, :, None]) ** 2).sum(axis=(-1, -2))
            best = cost.argmin(axis=2)                   # (k, g); 0 = keep
            if not best.any():
                continue
            changed = True
            chosen = options[np.arange(len(index))[None, :], best]   # (k, g, s)
            new[rows, index[None]] = perm[rows, chosen]
        if not changed:
            break
        perm = new
    else:
        logger.warning("Equivalent-atom renaming did not settle within %d rounds",
                       MAX_RENAMING_ROUNDS)
    yp = y[np.arange(k)[:, None], perm]
    h = np.einsum("kad,ae->kde", yp, x)
    sigma = np.linalg.svd(h, compute_uv=False)
    det_sign = np.sign(np.linalg.det(h))
    sigma[:, -1] *= np.where(det_sign == 0, 1.0, det_sign)
    msd = (sq_norms[i + 1:] + sq_norms[i] - 2.0 * sigma.sum(axis=1)) / m
    return np.sqrt(np.maximum(msd, 0.0))


# ---------------------------------------------------------------------------
# TM-score via US-align
# ---------------------------------------------------------------------------

def find_usalign(hint: str = "USalign") -> str:
    exe = shutil.which(hint) or (hint if Path(hint).is_file() else None)
    if exe is None:
        raise FileNotFoundError(
            f"US-align executable {hint!r} not found. Install it "
            "(https://github.com/pylelab/USalign) and put it on PATH, or set metric.usalign_exe."
        )
    return exe


def parse_usalign_table(stdout: str) -> float:
    """Mean of TM1/TM2 from ``USalign -outfmt 2`` output."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    header = next(ln for ln in lines if ln.startswith("#")).lstrip("#").split("\t")
    values = next(ln for ln in lines if not ln.startswith("#")).split("\t")
    row = dict(zip(header, values))
    return 0.5 * (float(row["TM1"]) + float(row["TM2"]))


def _usalign_pair(exe: str, a: str, b: str, flags: list[str]) -> float:
    cmd = [exe, a, b, *flags, "-outfmt", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    try:
        return parse_usalign_table(proc.stdout)
    except (StopIteration, KeyError, ValueError) as exc:
        raise RuntimeError(
            f"US-align failed on {Path(a).name} vs {Path(b).name} "
            f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()[:300]}"
        ) from exc


def tm_matrix(ens: Ensemble, exe_hint: str = "USalign", n_jobs: int = -1) -> np.ndarray:
    """All-pairs TM-score (similarity in [0, 1]); pairs run as parallel subprocesses."""
    exe = find_usalign(exe_hint)
    multichain = len(set(ens.atoms.chain.tolist())) > 1
    flags = ["-TMscore", "2" if multichain else "1"]
    if ens.molecule_type == "nucleic":
        flags += ["-mol", "RNA"]
    elif ens.molecule_type == "protein":
        flags += ["-mol", "prot"]

    n = len(ens)
    with tempfile.TemporaryDirectory(prefix="struct2clust_") as tmp:
        # Every member is written from the shared atom table, so US-align sees
        # exactly the atoms the rest of the pipeline uses.
        paths = []
        for k in range(n):
            p = Path(tmp) / f"m{k}.pdb"
            write_pdb(p, ens.atoms, ens.coords[k], renumber=True)
            paths.append(str(p))
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sim = np.eye(n)
        with ThreadPoolExecutor(_n_workers(n_jobs)) as pool:
            scores = pool.map(
                lambda ij: _usalign_pair(exe, paths[ij[0]], paths[ij[1]], flags), pairs
            )
            for (i, j), s in zip(pairs, scores):
                sim[i, j] = sim[j, i] = s
    return sim


# ---------------------------------------------------------------------------
# Entry point and de-duplication
# ---------------------------------------------------------------------------

DEFAULT_DEDUP = {"rmsd": 0.1, "tm": 1e-3}


def distance_matrix(
    ens: Ensemble,
    metric: str,
    atom_mask: Optional[np.ndarray] = None,
    usalign_exe: str = "USalign",
    n_jobs: int = -1,
    equivalent_atoms: bool = False,
) -> np.ndarray:
    """Symmetric distance matrix with zero diagonal (0 = identical)."""
    if metric == "rmsd":
        coords = ens.coords if atom_mask is None else ens.coords[:, atom_mask]
        groups = None
        if equivalent_atoms:
            atoms = ens.atoms if atom_mask is None else ens.atoms.subset(atom_mask)
            groups = equivalent_atom_groups(atoms)
            logger.info("RMSD minimised over the naming of %d groups of equivalent atoms",
                        sum(len(index) for index, _ in groups))
        d = rmsd_matrix(coords, n_jobs, groups)
    elif metric == "tm":
        d = 1.0 - tm_matrix(ens, usalign_exe, n_jobs)
    else:
        raise ValueError(f"Unknown metric {metric!r}")
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def deduplicate(
    dist: np.ndarray, threshold: float, description: str = "",
    names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Removal of near-duplicates, independent of the input order (Butina 1999).

    Neighbours are structures within ``threshold`` (distance <= threshold). Structures
    are taken in order of decreasing number of neighbours; ties go to the smaller sum
    of distances to those neighbours (the more central structure), then to the name.
    A structure not yet assigned is kept as a centre, and its unassigned neighbours
    are removed as its copies. Every removed structure is therefore within the
    threshold of its centre (``duplicate_of``); two copies of one centre can be up to
    twice the threshold apart.

    Returns (kept indices in input order, [(centre, removed, distance), ...]).
    """
    n = dist.shape[0]
    close = dist <= threshold
    np.fill_diagonal(close, False)
    counts = close.sum(axis=1)
    spread = np.where(close, dist, 0.0).sum(axis=1)      # summed distance to neighbours
    key = names if names is not None else [f"{i:09d}" for i in range(n)]
    order = sorted(range(n), key=lambda i: (-counts[i], round(float(spread[i]), 9), key[i]))
    assigned = np.zeros(n, dtype=bool)
    kept, log = [], []
    for i in order:
        if assigned[i]:
            continue
        assigned[i] = True
        kept.append(i)
        copies = np.flatnonzero(close[i] & ~assigned)
        assigned[copies] = True
        log.extend((i, int(j), float(dist[i, j])) for j in copies)
    if log:
        logger.info("Removed %d near-duplicate structures (%s)", len(log),
                    description or f"distance <= {threshold:.3g}")
    return np.array(sorted(kept)), sorted(log)
