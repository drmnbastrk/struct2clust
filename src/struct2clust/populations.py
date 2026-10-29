"""Trajectory mode: uncertainty of cluster populations and their agreement between parts.

Populations are the fractions of frames in every final cluster (noise excluded from the
clusters, included in the total). Their sampling uncertainty is estimated by a block
bootstrap (Künsch 1989): blocks of every trajectory (about three times the longer of
its geometric and membership autocorrelation time, sampling.population_blocks) are
drawn with replacement, as
many as the trajectory has, and the populations are recomputed; the quantiles at
LEVEL give the interval. With few transitions block-bootstrap intervals are too
narrow, so LEVEL is the smallest nominal level whose intervals reached about 95%
coverage in simulations (validation/population_calibration.py); the report calls
them approximate 95% intervals, with the coverage measured. Resampling blocks rather
than frames is the remedy for correlated data of Grossfield et al. (2018, section
7.6). The partition is held fixed:
the interval describes how much the populations of these clusters depend on the frames
sampled, not the uncertainty of the clustering itself.

Convergence (Grossfield et al. 2018, section 4.4; Lyman & Zuckerman 2006): the frames
of all trajectories are clustered together and the populations are compared between
the parts: the trajectories, or the two halves of a single trajectory (first and second
half of its blocks). For every pair of parts and every cluster the difference of the
populations gets a bootstrap interval (each part resampled on its own blocks); the
intervals are Bonferroni-adjusted over clusters and pairs to FLAG_LEVEL family-wise.
FLAG_LEVEL stays at 0.95, below LEVEL: at 0.999 the flag missed parts that sampled
different states (8 blocks each), and a missed difference is worse than a false
warning. A cluster whose interval
excludes zero has populations that differ between the parts by more than their sampling
uncertainty: the parts have not sampled the same distribution, and the populations are
those of the simulations, not of an equilibrium (docs/METHODS.md, section 9).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

N_BOOTSTRAP = 2000
LEVEL = 0.999       # nominal; covered the true population in 92-99.5% (METHODS.md, section 9)
FLAG_LEVEL = 0.95   # family-wise level of the comparison of the parts


def _counts(labels: np.ndarray, blocks: np.ndarray, n_clusters: int):
    """(block ids, frames per block in every cluster, frames per block)."""
    ids, inv = np.unique(blocks, return_inverse=True)
    per = np.zeros((len(ids), n_clusters))
    member = labels >= 0
    np.add.at(per, (inv[member], labels[member]), 1)
    return ids, per, np.bincount(inv, minlength=len(ids)).astype(float)


def _bootstrap(rng: np.random.Generator, labels, blocks, n_clusters: int, n: int) -> np.ndarray:
    """(n, n_clusters) populations from n block-bootstrap samples of these frames."""
    _, per, size = _counts(labels, blocks, n_clusters)
    k = len(size)
    weights = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n)])
    return (weights @ per) / (weights @ size)[:, None]


def _parts(replica: np.ndarray, blocks: np.ndarray) -> dict[str, np.ndarray]:
    """Member mask per part: every trajectory, or the halves (by block) of a single one."""
    names = list(dict.fromkeys(replica))
    if len(names) > 1:
        return {n: replica == n for n in names}
    ids = np.unique(blocks)
    first = np.isin(blocks, ids[: len(ids) // 2])
    return {"first_half": first, "second_half": ~first}


def population_uncertainty(labels: np.ndarray, blocks: np.ndarray, replica: list[str],
                           seed: int, n_bootstrap: int = N_BOOTSTRAP,
                           level: float = LEVEL, flag_level: float = FLAG_LEVEL) -> dict:
    """Bootstrap intervals of the populations at ``level``, overall and per part, and
    the agreement of the parts at family-wise ``flag_level`` (module docstring)."""
    labels, blocks, replica = np.asarray(labels), np.asarray(blocks), np.asarray(replica)
    k = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    rng = np.random.default_rng([seed, 31])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    # overall: blocks drawn within every trajectory, so each keeps its share of the frames
    boots = []
    for name in dict.fromkeys(replica):
        m = replica == name
        b = _bootstrap(rng, labels[m], blocks[m], k, n_bootstrap)
        boots.append(b * m.sum())
    overall = sum(boots) / len(labels)
    out = {"fraction_low": np.quantile(overall, lo, axis=0).tolist(),
           "fraction_high": np.quantile(overall, hi, axis=0).tolist(), "parts": {},
           "comparisons": []}
    parts = _parts(replica, blocks)
    samples = {}
    for name, m in parts.items():
        samples[name] = _bootstrap(rng, labels[m], blocks[m], k, n_bootstrap)
        out["parts"][name] = {
            "n_frames": int(m.sum()), "n_blocks": int(len(np.unique(blocks[m]))),
            "population": [float(np.mean(labels[m] == c)) for c in range(k)],
            "low": np.quantile(samples[name], lo, axis=0).tolist(),
            "high": np.quantile(samples[name], hi, axis=0).tolist()}
    pairs = list(combinations(parts, 2))
    alpha = (1 - flag_level) / max(1, len(pairs) * k)  # Bonferroni over pairs and clusters
    for a, b in pairs:
        diff = samples[a] - samples[b]
        pa, pb = np.array(out["parts"][a]["population"]), np.array(out["parts"][b]["population"])
        d_lo = np.quantile(diff, alpha / 2, axis=0)
        d_hi = np.quantile(diff, 1 - alpha / 2, axis=0)
        for c in range(k):
            out["comparisons"].append({
                "part_a": a, "part_b": b, "cluster": c + 1,
                "population_a": round(float(pa[c]), 4), "population_b": round(float(pb[c]), 4),
                "difference": round(float(pa[c] - pb[c]), 4),
                "difference_low": round(float(d_lo[c]), 4),
                "difference_high": round(float(d_hi[c]), 4),
                "differs": bool(d_lo[c] > 0 or d_hi[c] < 0)})
    out["interval_level"], out["flag_level"] = level, flag_level
    return out
