"""Cluster-wise stability by resampling (Hennig 2007, Comput. Stat. Data Anal. 52:258).

A clustering is compared with clusterings of many random subsamples made with
the same HDBSCAN settings. For every cluster C and subsample S, the Jaccard
similarity between C restricted to S and its best-matching subsample cluster is
recorded. Hennig's interpretation of the mean over subsamples:

    >= 0.75    stable cluster                     -> tier HIGH
    0.60-0.75  a pattern, membership doubtful     -> tier MODERATE
    <  0.60    should not be trusted              -> tier LOW
    <= 0.50    counts as "dissolved" in that subsample

Subsampling without replacement is used instead of the bootstrap because
duplicated points create artificial zero-distance density for HDBSCAN.

The same resamples give a membership probability for every structure: the
fraction of subsamples containing it in which it falls inside the best match of
its cluster.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from struct2clust.clustering import hdbscan_labels, subsample

logger = logging.getLogger(__name__)

DISSOLVED = 0.5


def subsample_params(params: tuple, fraction: float) -> tuple:
    """HDBSCAN settings scaled to a subsample of relative size ``fraction``.

    A subsample is a scaled-down copy of the data: a cluster with s members has
    about fraction * s of them in it. Without scaling, clusters whose size is
    close to min_cluster_size could not be found in any subsample and would
    look unstable for purely arithmetic reasons.
    """
    mcs, ms, method = params
    return (max(2, round(mcs * fraction)), max(1, round(ms * fraction)), method)


def one_resample(dist, labels, blocks, params, fraction, seed, absent_zero=False):
    """Jaccard per cluster of ``labels`` and matched membership for one subsample.

    Returns (subsample indices, Jaccard per cluster, boolean (n_clusters,
    len(indices)) array: member lies in the cluster's match). A cluster none of
    whose members is in the subsample gets NaN (ignored), or 0 with
    ``absent_zero`` (resampling by group: the cluster was not reproduced from the
    independent groups drawn).
    """
    rng = np.random.default_rng(seed)
    idx = subsample(rng, blocks, fraction)
    sub = hdbscan_labels(dist[np.ix_(idx, idx)], *subsample_params(params, len(idx) / len(labels)))
    k = labels.max() + 1
    jac = np.full(k, np.nan)
    in_match = np.zeros((k, len(idx)), dtype=bool)
    orig = labels[idx]
    sub_masks = [sub == d for d in np.unique(sub[sub >= 0])]
    for c in range(k):
        mask_c = orig == c
        if not mask_c.any():
            if absent_zero:
                jac[c] = 0.0
            continue
        best, best_mask = 0.0, None
        for mask_d in sub_masks:
            inter = np.count_nonzero(mask_c & mask_d)
            if inter:
                j = inter / np.count_nonzero(mask_c | mask_d)
                if j > best:
                    best, best_mask = j, mask_d
        jac[c] = best
        if best_mask is not None:
            in_match[c] = best_mask
    return idx, jac, in_match


def resample_jaccard(dist, labels, blocks, params, n_resamples, fraction, seed, n_jobs=1,
                     absent_zero=False):
    """(B, n_clusters) Jaccard matrix and (n, n_clusters) membership frequencies."""
    seeds = np.random.SeedSequence(seed).generate_state(n_resamples)
    results = Parallel(n_jobs=n_jobs)(
        delayed(one_resample)(dist, labels, blocks, params, fraction, int(s), absent_zero)
        for s in seeds
    )
    n, k = len(labels), labels.max() + 1
    hits, seen = np.zeros((n, k)), np.zeros(n)
    for idx, _, in_match in results:
        seen[idx] += 1
        hits[idx] += in_match.T
    jac = np.array([r[1] for r in results])
    return jac, hits / np.maximum(seen, 1)[:, None]


@dataclass
class Stability:
    clusters: pd.DataFrame     # one row per cluster
    membership: np.ndarray     # (n_members, n_clusters) co-assignment frequency
    n_resamples: int


def tier(mean_jaccard: float, high: float, moderate: float) -> str:
    return "HIGH" if mean_jaccard >= high else "MODERATE" if mean_jaccard >= moderate else "LOW"


def cluster_stability(
    dist: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    params: tuple,
    n_resamples: int = 200,
    fraction: float = 0.8,
    high: float = 0.75,
    moderate: float = 0.60,
    seed: int = 42,
    n_jobs: int = -1,
    quiet: bool = False,
    absent_zero: bool = False,
) -> Stability:
    """Final, precise stability estimate of the selected clustering."""
    n, k = len(labels), labels.max() + 1
    if k == 0:
        return Stability(pd.DataFrame(), np.zeros((n, 0)), n_resamples)
    jac, membership = resample_jaccard(
        dist, labels, blocks, params, n_resamples, fraction, seed + 1, n_jobs, absent_zero
    )
    rows = []
    for c in range(k):
        v = jac[:, c][~np.isnan(jac[:, c])]
        mean = float(v.mean())
        rows.append({
            "cluster": c + 1,
            "size": int((labels == c).sum()),
            "jaccard_mean": mean,
            "jaccard_se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            "jaccard_q025": float(np.quantile(v, 0.025)),
            "jaccard_q975": float(np.quantile(v, 0.975)),
            "dissolved_fraction": float(np.mean(v <= DISSOLVED)),
            "tier": tier(mean, high, moderate),
        })
    table = pd.DataFrame(rows)
    (logger.debug if quiet else logger.info)(
        "Cluster stability: %s",
        ", ".join(f"C{r.cluster}={r.jaccard_mean:.2f} ({r.tier})" for r in table.itertuples()),
    )
    return Stability(table, membership, n_resamples)
