"""Is there cluster structure at all?

Clustering always proceeds; the verdict limits the confidence tiers
(CLUSTERABLE: up to HIGH, WEAK: up to MODERATE, NOT_CLUSTERABLE: LOW).

Dip-dist: Hartigan's dip test of unimodality on the pairwise distances. A
multimodal distance distribution indicates groups (Adolfsson, Ackerman &
Brownstein 2019, Pattern Recognit. 88:13). Pairwise distances are not
independent, so the p-value is indicative rather than exact.

Spectral eigengap: largest gap among the first eigenvalues of the normalised
graph Laplacian of a k-nearest-neighbour graph with self-tuning Gaussian
affinities (Zelnik-Manor & Perona 2004; von Luxburg 2007). The affinity scale
adapts to local density, so the result does not depend on distance units.
A suggested k of 1 means no gap beyond the trivial eigenvalue.

What the verdict means. In a simulation study the eigengap suggested k >= 2
for every structureless but anisotropic sample (curved, elongated, density
gradient, drifting MD): for a continuum the eigenvalues grow steadily and the
largest gap falls at the end of the search window (von Luxburg 2007 notes that
the heuristic is reliable only for pronounced clusters). The spectral vote
therefore indicates a non-homogeneous neighbourhood graph, not groups; the
evidence for groups is the dip test. WEAK (spectral only) thus means that
weakly separated groups and a continuum cannot be told apart, which is why
WEAK caps tiers at MODERATE and is reported as "group structure not confirmed".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from diptest import diptest
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)


@dataclass
class Clusterability:
    dip_statistic: float
    dip_pvalue: float
    eigenvalues: np.ndarray
    spectral_k: int
    verdict: str   # CLUSTERABLE | WEAK | NOT_CLUSTERABLE


# diptest's p-value tables are exact up to 72,000 values; larger inputs are sampled.
MAX_DIP_PAIRS = 72_000
DENSE_EIGEN_LIMIT = 2000


def dip_dist(dist: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """Dip test on the pairwise distances (a random sample of pairs for very large N)."""
    n = len(dist)
    n_pairs = n * (n - 1) // 2
    if n_pairs <= MAX_DIP_PAIRS:
        iu = np.triu_indices(n, k=1)
        values = dist[iu]
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, MAX_DIP_PAIRS)
        j = rng.integers(0, n - 1, MAX_DIP_PAIRS)
        j += j >= i  # j != i
        values = dist[i, j]
    stat, p = diptest(values)
    return float(stat), float(p)


def spectral_eigengap(dist: np.ndarray, n_neighbors: int = 15, max_k: int = 10):
    """(eigenvalues, suggested k) with k in 1..max_k."""
    n = len(dist)
    k = min(n_neighbors, n - 1)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(d, nn, axis=1), axis=1), axis=1)
    scale_nn = min(7, k)
    sigma = np.take_along_axis(d, nn[:, scale_nn - 1:scale_nn], axis=1).ravel()
    sigma = np.maximum(sigma, 1e-12)
    rows = np.repeat(np.arange(n), k)
    cols = nn.ravel()
    w = np.exp(-dist[rows, cols] ** 2 / (sigma[rows] * sigma[cols]))
    a = csr_matrix((w, (rows, cols)), shape=(n, n))
    a = a.maximum(a.T)
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    norm_adj = dinv @ a @ dinv  # eigenvalues of the Laplacian are 1 - these
    m = min(max_k + 1, n)
    if n <= DENSE_EIGEN_LIMIT:
        eig = np.sort(np.linalg.eigvalsh((identity(n) - norm_adj).toarray()))[:m]
    else:
        top = eigsh(norm_adj, k=m, which="LA", return_eigenvectors=False)
        eig = np.sort(1.0 - top)
    eig = np.clip(eig, 0.0, None)
    # A graph with c components has c zero eigenvalues, so the largest gap also
    # counts disconnected groups (plus any further structure inside them).
    # gaps[j] = lambda_{j+1} - lambda_j, so the gap after k eigenvalues is gaps[k-1]
    gaps = np.diff(eig)
    return eig, int(np.argmax(gaps[:max_k]) + 1)


def assess(dist: np.ndarray, alpha: float = 0.05, n_neighbors: int = 15, max_k: int = 10,
           quiet: bool = False):
    stat, p = dip_dist(dist)
    eig, k = spectral_eigengap(dist, n_neighbors, max_k)
    votes = int(p < alpha) + int(k >= 2)
    verdict = ["NOT_CLUSTERABLE", "WEAK", "CLUSTERABLE"][votes]
    (logger.debug if quiet else logger.info)(
        "Clusterability: dip p=%.3g, spectral k=%d -> %s", p, k, verdict)
    return Clusterability(stat, p, eig, k, verdict)
