"""HDBSCAN on a precomputed distance matrix, candidate settings, resampling helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import HDBSCAN

# Cluster extraction: excess of mass assigns structures to the most persistent
# clusters instead of trimming clusters to their cores, so structures stay
# assigned (docs/METHODS.md, section 6, "Why this rule").
METHODS = ("eom",)


def hdbscan_labels(dist: np.ndarray, mcs: int, ms: int, method: str = "eom") -> np.ndarray:
    """HDBSCAN labels (-1 = noise).

    ``method`` is the cluster-extraction rule: "eom" (excess of mass, the
    HDBSCAN default) or "leaf" (finest clusters of the condensed tree).
    """
    model = HDBSCAN(
        min_cluster_size=mcs, min_samples=ms, metric="precomputed",
        cluster_selection_method=method, copy=True,
    )
    return model.fit(dist).labels_


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 0, 1, ... from largest to smallest; noise stays -1."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.lexsort((ids, -counts))]
    out = np.full_like(labels, -1)
    for rank, label in enumerate(order):
        out[labels == label] = rank
    return out


def candidate_grid(
    n: int,
    mcs_values: Optional[list[int]] = None,
    ms_values: Optional[list[int]] = None,
) -> list[tuple[int, int, str]]:
    """(min_cluster_size, min_samples, method) candidates.

    Default min_cluster_size: every integer from 5 to 20, then steps of ~10% up
    to max(10, N/10). Default min_samples: {1, 2, 3, 5, 8, min_cluster_size}.
    """
    if mcs_values is None:
        hi = max(10, n // 10)
        values, v = set(range(5, min(20, hi) + 1)), 20.0
        while v < hi:
            v *= 1.1
            values.add(min(hi, int(round(v))))
        mcs_values = sorted(values)
    grid = []
    for mcs in sorted(set(mcs_values)):
        if mcs < 2 or mcs > n // 2:
            continue
        ms_set = ms_values if ms_values is not None else [1, 2, 3, 5, 8, mcs]
        for ms in sorted({m for m in ms_set if 1 <= m <= mcs}):
            for method in METHODS:
                grid.append((mcs, ms, method))
    if not grid:
        raise ValueError(f"No valid (min_cluster_size, min_samples) candidates for N={n}")
    return grid


def subsample(rng: np.random.Generator, blocks: np.ndarray, fraction: float) -> np.ndarray:
    """Sorted member indices drawn without replacement, whole blocks at a time.

    ``blocks`` assigns each member to a block; ensembles use one block per
    member, trajectories contiguous runs of frames, so time-correlated frames
    are kept or dropped together.
    """
    ids = np.unique(blocks)
    chosen = rng.choice(ids, size=max(2, int(round(fraction * len(ids)))), replace=False)
    return np.flatnonzero(np.isin(blocks, chosen))


def trajectory_blocks(n: int, n_blocks: int = 50) -> np.ndarray:
    return np.minimum(np.arange(n) * n_blocks // n, n_blocks - 1)


def medoids(dist: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Member with the smallest summed distance to the rest of its cluster."""
    out = {}
    for c in range(labels.max() + 1):
        idx = np.flatnonzero(labels == c)
        out[c] = int(idx[np.argmin(dist[np.ix_(idx, idx)].sum(axis=1))])
    return out
