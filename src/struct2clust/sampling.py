"""How many independent samples a trajectory holds.

Neighbouring MD frames are correlated, so resampling uses contiguous blocks of
frames (docs/METHODS.md, section 9). For every trajectory this module reports
the integrated autocorrelation time of its slowest geometric motion and the
effective number of independent frames, so that the reader can judge how much
independent information the trajectory and its resampling blocks carry.

The motion is summarised by the first principal coordinate of the trajectory's
distance matrix (classical scaling; Gower 1966), the one-dimensional coordinate
that preserves most of the distances. Its integrated autocorrelation time
tau = 1 + 2 sum_t rho(t) is estimated with Sokal's automatic window: the sum
stops at the first lag W >= c tau(W) (Sokal 1997). Its relative statistical error
is about sqrt(2 (2W + 1) / n) (Madras & Sokal 1988; Wolff 2004), so a trajectory
that is short compared with its correlation time gives an uncertain tau. The
effective number of frames is n / tau (Lyman & Zuckerman 2007). Slower
motions along other coordinates are not seen, so tau is itself a lower bound on
the slowest correlation time of the trajectory.

By default every trajectory is resampled in as many blocks as it holds independent
frames (blocks of about tau frames), between MIN_BLOCKS and MAX_BLOCKS: resampling
only as many units as there are independent samples is the remedy for correlated data
of Grossfield et al. (2018, section 7.6).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse.linalg import eigsh

WINDOW_C = 6.0   # window factor c; 6-10 is the usual range
MIN_BLOCKS = 8   # fewest automatic blocks: as for resampling groups (hierarchy.MIN_GROUPS)
MAX_BLOCKS = 50  # most automatic blocks; more add resampling units, not independence
# population intervals use blocks of this many autocorrelation times (docs/METHODS.md,
# section 9, "How approximate": blocks of one tau gave intervals that were too narrow)
POPULATION_BLOCK_TAUS = 3


def principal_coordinate(dist: np.ndarray) -> np.ndarray:
    """First principal coordinate of a distance matrix (classical scaling).

    Works on one copy of the matrix (double centring in place)."""
    n = len(dist)
    if n <= 3:
        return np.zeros(n)
    b = np.square(dist)                     # the only n x n copy
    row = b.mean(axis=1)
    b -= row[:, None]
    b -= row[None, :]
    b += row.mean()
    b *= -0.5
    _, vec = eigsh(b, k=1, which="LA")
    return vec[:, 0]


def integrated_autocorrelation_time(x: np.ndarray) -> tuple[float, float]:
    """(tau, relative error): tau = 1 + 2 sum rho(t) with Sokal's automatic window W,
    and its relative statistical error sqrt(2 (2W + 1) / n) (Madras & Sokal 1988)."""
    n = len(x)
    x = np.asarray(x, float) - np.mean(x)
    if n < 4 or not np.any(x):
        return 1.0, float("inf") if n < 4 else 0.0
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau, window = 1.0, n - 1
    for t in range(1, n):
        tau += 2.0 * acf[t]
        if t >= WINDOW_C * tau:
            window = t
            break
    return float(max(tau, 1.0)), float(np.sqrt(2 * (2 * window + 1) / n))


def automatic_blocks(n_frames: int, tau: float) -> int:
    """Blocks of about one autocorrelation time: the number of independent frames
    n / tau, between MIN_BLOCKS and MAX_BLOCKS (never more than the frames)."""
    return int(min(n_frames, MAX_BLOCKS, max(MIN_BLOCKS, np.floor(n_frames / tau))))


def population_blocks(sampling: list[dict], membership: Optional[dict] = None) -> dict:
    """Blocks per trajectory for the population intervals: blocks of about
    POPULATION_BLOCK_TAUS times the longer of the geometric autocorrelation time and the
    longest membership autocorrelation time in that trajectory (``membership``, from
    membership_autocorrelation: the populations are means of the membership indicators),
    between MIN_BLOCKS and MAX_BLOCKS; or the fixed number of a trajectory with fixed
    blocks."""
    out = {}
    for t in sampling:
        if t["blocks_rule"] == "fixed":
            out[t["name"]] = t["n_blocks"]
            continue
        tau = max([t["autocorrelation_time"], *(membership or {}).get(t["name"], {}).values()])
        out[t["name"]] = automatic_blocks(t["n_frames"], POPULATION_BLOCK_TAUS * tau)
    return out


def trajectory_sampling(replica: list[str], dist: np.ndarray,
                        n_blocks: Optional[int] = None) -> list[dict]:
    """Per trajectory: frames, integrated autocorrelation time (frames) with its relative
    error, effective number of independent frames, number of resampling blocks
    (``n_blocks``, or automatic_blocks if None) and their length (frames), and whether
    a block is shorter than one autocorrelation time (then the blocks hold fewer
    independent frames than there are blocks)."""
    labels = np.asarray(replica)
    out = []
    for name in dict.fromkeys(replica):
        idx = np.flatnonzero(labels == name)
        sub = dist if len(idx) == len(dist) else dist[np.ix_(idx, idx)]
        tau, rel_error = integrated_autocorrelation_time(principal_coordinate(sub))
        blocks = (automatic_blocks(len(idx), tau) if n_blocks is None
                  else min(n_blocks, len(idx)))
        block = len(idx) / blocks
        out.append({"name": name, "n_frames": len(idx),
                    "autocorrelation_time": round(tau, 1),
                    "autocorrelation_time_rel_error": round(rel_error, 2),
                    "effective_frames": round(len(idx) / tau, 1),
                    "n_blocks": blocks,
                    "blocks_rule": "automatic" if n_blocks is None else "fixed",
                    "block_length": round(block, 1),
                    "block_shorter_than_autocorrelation_time": bool(block < tau)})
    return out


def initial_positive_sequence_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time tau = 1 + 2 sum rho(t) by Geyer's initial positive
    sequence estimator: the sums of neighbouring pairs rho(2k) + rho(2k + 1) are added
    while they are positive (Geyer 1992). Unlike a window that grows with the estimate,
    it follows a slowly decaying correlation to its end, e.g. the membership of a state
    that is visited only a few times."""
    n = len(x)
    x = np.asarray(x, float) - np.mean(x)
    if n < 4 or not np.any(x):
        return 1.0
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    total = 0.0
    for k in range(n // 2):
        pair = acf[2 * k] + acf[2 * k + 1]
        if pair <= 0:
            break
        total += pair
    return float(max(2 * total - 1, 1.0))


def membership_autocorrelation(labels: np.ndarray, replica: list[str]) -> dict:
    """{trajectory: {cluster (1-based): tau}}: integrated autocorrelation time (frames) of
    the indicator "frame in cluster c" along every trajectory (the observable whose
    stability and population are reported), by initial_positive_sequence_time. Clusters a
    trajectory never leaves or never visits have no autocorrelation time there and are
    left out."""
    labels, rep = np.asarray(labels), np.asarray(replica)
    out = {}
    for name in dict.fromkeys(replica):
        lab = labels[rep == name]
        out[name] = {}
        for c in range(int(labels.max()) + 1):
            x = (lab == c).astype(float)
            if 0 < x.sum() < len(x):
                out[name][c + 1] = round(initial_positive_sequence_time(x), 1)
    return out


def cluster_transitions(labels: np.ndarray, replica: list[str]) -> dict:
    """{cluster (1-based): number of times a trajectory enters or leaves it}, summed over
    the trajectories, for every cluster that has frames. Noise frames are skipped, so a
    trajectory that stays in one cluster apart from noise frames never leaves it."""
    labels, rep = np.asarray(labels), np.asarray(replica)
    out = {c + 1: 0 for c in np.unique(labels[labels >= 0])}
    for name in dict.fromkeys(replica):
        lab = labels[rep == name]
        lab = lab[lab >= 0]
        for a, b in zip(lab[:-1], lab[1:]):
            if a != b:
                out[a + 1] += 1
                out[b + 1] += 1
    return out


def sampling_check(sampling: list[dict], membership: dict,
                   transitions: Optional[dict] = None,
                   members_of: Optional[dict] = None) -> dict:
    """Whether the resampling blocks can be treated as independent: every trajectory
    has blocks at least one autocorrelation time of the geometry long (it holds at least
    as many independent frames as blocks), no cluster's membership stays correlated
    for longer than one block, and no cluster is free of transitions (``transitions``,
    from cluster_transitions). A cluster that no trajectory enters or leaves consists of
    whole trajectories (``members_of``: {cluster: trajectories}), so the trajectory, not
    the block, is its independent unit, and a cluster drawn from one trajectory is found
    in every subsample that holds any of its blocks. A pre-specified quality-control rule
    (docs/METHODS.md, section 9); passing it does not establish convergence."""
    reasons, clusters = [], set()
    frozen = sorted(c for c, n in (transitions or {}).items() if n == 0)
    for c in frozen:
        where = ", ".join((members_of or {}).get(c, []))
        reasons.append(f"no trajectory enters or leaves cluster {c}"
                       + (f" (trajectory {where})" if where else "")
                       + ": its frames come from whole trajectories, not from repeated "
                       "visits, so its reproducibility cannot be assessed")
    for t in sampling:
        if t["block_shorter_than_autocorrelation_time"]:
            reasons.append(f"trajectory {t['name']}: blocks of {t['block_length']:.0f} frames "
                           "are shorter than its autocorrelation time "
                           f"({t['autocorrelation_time']:.0f} frames; "
                           f"{t['effective_frames']:.0f} independent frames)")
        for c, tau in membership.get(t["name"], {}).items():
            if tau > t["block_length"]:
                clusters.add(c)
                reasons.append(f"trajectory {t['name']}: membership of cluster {c} stays "
                               f"correlated for {tau:.0f} frames, longer than one block "
                               f"({t['block_length']:.0f} frames)")
    return {"sufficient": not reasons, "reasons": reasons,
            "clusters_longer_than_block": sorted(clusters),
            "clusters_without_transitions": frozen}


def membership_sampling(sampling: list[dict], membership: dict,
                        check: dict) -> Optional[list[dict]]:
    """Blocks of about one membership autocorrelation time, for a second resampling pass
    (input.membership_blocks), or None if the check did not fail for that reason alone.

    Eligible: the check failed, every trajectory has blocks at least one geometric
    autocorrelation time long, every cluster has transitions, and every trajectory is at
    least MIN_BLOCKS times as long as the longest membership autocorrelation time in it
    (or its geometric one). The new blocks of a trajectory are automatic_blocks with that
    time; the returned list has the fields of trajectory_sampling, blocks_rule
    "membership" and membership_autocorrelation_time."""
    if check["sufficient"] or check.get("clusters_without_transitions"):
        return None
    if any(t["block_shorter_than_autocorrelation_time"] for t in sampling):
        return None
    out = []
    for t in sampling:
        tau = max([t["autocorrelation_time"], *membership.get(t["name"], {}).values()])
        if t["n_frames"] < MIN_BLOCKS * tau:
            return None
        blocks = automatic_blocks(t["n_frames"], tau)
        block = t["n_frames"] / blocks
        out.append({**t, "n_blocks": blocks, "blocks_rule": "membership",
                    "block_length": round(block, 1),
                    "block_shorter_than_autocorrelation_time":
                        bool(block < t["autocorrelation_time"]),
                    "membership_autocorrelation_time": round(tau, 1)})
    return out
