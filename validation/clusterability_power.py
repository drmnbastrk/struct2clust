"""Power and false-positive rate of the group-structure test, and of three alternatives,
on partitions with a known answer (docs/METHODS.md, section 4, "Power of the dip test").
No resolution selection: each test is applied to a given partition.

Structureless data (no test should confirm groups): Gaussian, uniform, elongated,
curved (banana) and density-gradient clouds (300 points) and an AR(1) drift (500
frames), each cut by k-means into 2 and 3 parts, or into its outer 10% / 20% along the
first principal axis and the rest (an unbalanced cut).
Groups (the tests should confirm them; the true groups are the partition): two groups
6 sigma apart with the smaller one holding 50/20/10/5% of the points, and 8 sigma apart
with 10/5%; three groups on a line 4/5/6 sigma apart; 4 and 6 groups with every pair 6
sigma apart (50 points each).

Tests (confirm group structure if ...)
  dipdist         dip test on all pairwise distances, p < 0.05 (used by Struct2Clust).
  baldip          for every cluster and its nearest cluster (by medoid distance): the
                  dip test on the distances within their union, the larger subsampled to
                  the size of the smaller; every pair p < 0.05 / m (m pairs).
  proj            for the same pairs: the dip test on their members projected onto the
                  line through the two medoids, t = (d(x,a)^2 - d(x,b)^2) / (2 d(a,b));
                  every pair p < 0.05 / m. The same data define and test the pair.
  proj_split      as proj, with sample splitting: medoids from a random half, the other
                  half assigned to the nearest medoid and projected.
  any_*           some pair instead of every pair; dip|*: either test.

    python validation/clusterability_power.py validation/results/clusterability_power.csv 100
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from diptest import diptest
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios as sc  # noqa: E402  (validation/scenarios.py)

from struct2clust.clusterability import dip_dist  # noqa: E402

ALPHA = 0.05
D = 6


def medoid(dist, idx):
    return idx[np.argmin(dist[np.ix_(idx, idx)].sum(axis=1))]


def pair_p(dist, a_idx, b_idx, ma, mb):
    dab = dist[ma, mb]
    pts = np.r_[a_idx, b_idx]
    t = (dist[pts, ma] ** 2 - dist[pts, mb] ** 2) / (2 * dab)
    return float(diptest(t)[1]) if len(pts) >= 4 else 1.0


def nearest_pairs(dist, meds):
    k = len(meds)
    md = np.array([[dist[meds[i], meds[j]] for j in range(k)] for i in range(k)])
    np.fill_diagonal(md, np.inf)
    return sorted({tuple(sorted((i, int(np.argmin(md[i]))))) for i in range(k)})


def proj_pvalues(dist, labels):
    k = labels.max() + 1
    idx = [np.flatnonzero(labels == c) for c in range(k)]
    meds = [medoid(dist, i) for i in idx]
    return [pair_p(dist, idx[a], idx[b], meds[a], meds[b]) for a, b in nearest_pairs(dist, meds)]


def proj_split_pvalues(dist, labels, rng):
    k = labels.max() + 1
    n = len(labels)
    half = np.zeros(n, bool)
    half[rng.choice(n, n // 2, replace=False)] = True
    idx_a = [np.flatnonzero((labels == c) & half) for c in range(k)]
    if any(len(i) == 0 for i in idx_a):
        return [1.0]
    meds = np.array([medoid(dist, i) for i in idx_a])
    test = np.flatnonzero(~half & (labels >= 0))
    assign = np.argmin(dist[np.ix_(test, meds)], axis=1)
    out = []
    for a, b in nearest_pairs(dist, list(meds)):
        out.append(pair_p(dist, test[assign == a], test[assign == b], meds[a], meds[b]))
    return out


def balanced_dip_pvalues(dist, labels, rng):
    """For every cluster and its nearest cluster: the dip-dist test on their union, the
    larger one subsampled to the size of the smaller, so that within- and between-pair
    distances have comparable mass whatever the cluster sizes."""
    k = labels.max() + 1
    idx = [np.flatnonzero(labels == c) for c in range(k)]
    meds = [medoid(dist, i) for i in idx]
    out = []
    for a, b in nearest_pairs(dist, meds):
        n = min(len(idx[a]), len(idx[b]))
        pts = np.r_[rng.choice(idx[a], n, replace=False), rng.choice(idx[b], n, replace=False)]
        out.append(dip_dist(dist[np.ix_(pts, pts)])[1] if n >= 3 else 1.0)
    return out


def gates(dist, labels, rng):
    p_dip = dip_dist(dist)[1]
    pp = proj_pvalues(dist, labels)
    ps = proj_split_pvalues(dist, labels, rng)
    pb = balanced_dip_pvalues(dist, labels, rng)
    m, ms, mb = len(pp), len(ps), len(pb)
    return {
        "dip_p": p_dip,
        "dipdist": p_dip < ALPHA,
        "baldip": all(p < ALPHA / mb for p in pb),
        "any_baldip": any(p < ALPHA / mb for p in pb),
        "dip|baldip": p_dip < ALPHA or all(p < ALPHA / mb for p in pb),
        "proj": all(p < ALPHA / m for p in pp),
        "any_proj": any(p < ALPHA / m for p in pp),
        "proj_split": all(p < ALPHA / ms for p in ps),
        "any_proj_split": any(p < ALPHA / ms for p in ps),
        "dip|proj_split": p_dip < ALPHA or all(p < ALPHA / ms for p in ps),
    }


def two_groups(minor, sep, n=300):
    def make(rng):
        m = int(round(n * minor))
        x = np.vstack([rng.normal(size=(n - m, D)),
                       rng.normal(size=(m, D)) + np.r_[sep, np.zeros(D - 1)]])
        return x, np.r_[np.zeros(n - m, int), np.ones(m, int)]
    return make


def line(k, sep):
    def make(rng):
        return sc.blobs(rng, k, sep), np.repeat(np.arange(k), 100)
    return make


def simplex(k, sep=6.0, n=50):
    def make(rng):
        c = np.eye(D)[:k] * sep / np.sqrt(2)
        return (np.vstack([rng.normal(size=(n, D)) + c[i] for i in range(k)]),
                np.repeat(np.arange(k), n))
    return make


def null(kind, k):
    def make(rng):
        x = sc.ar1(rng, 500) if kind == "md_drift" else sc.SHAPES[kind](rng, 300)
        lab = KMeans(k, n_init=10, random_state=int(rng.integers(1 << 31))).fit_predict(x)
        return x, lab
    return make


def null_tail(kind, q):
    """A continuum cut unevenly: the outer (1 - q) of the data along the first principal
    axis against the rest (an unbalanced cut, the hard case for a balanced test)."""
    def make(rng):
        x = sc.ar1(rng, 500) if kind == "md_drift" else sc.SHAPES[kind](rng, 300)
        xc = x - x.mean(axis=0)
        pc = xc @ np.linalg.svd(xc, full_matrices=False)[2][0]
        return x, (pc > np.quantile(pc, q)).astype(int)
    return make


SCEN = {}
for kind in ("gauss", "uniform", "elongated", "banana", "gradient", "md_drift"):
    for k in (2, 3):
        SCEN[f"null_{kind}_k{k}"] = null(kind, k)
    for q in (0.8, 0.9):
        SCEN[f"null_{kind}_tail{round(100 * (1 - q))}"] = null_tail(kind, q)
for minor in (0.5, 0.2, 0.1, 0.05):
    SCEN[f"pos_2groups_6s_minor{minor}"] = two_groups(minor, 6)
for minor in (0.1, 0.05):
    SCEN[f"pos_2groups_8s_minor{minor}"] = two_groups(minor, 8)
for s in (4, 5, 6):
    SCEN[f"pos_line3_{s}s"] = line(3, s)
for k in (4, 6):
    SCEN[f"pos_simplex{k}_6s"] = simplex(k)

if __name__ == "__main__":
    out, reps = sys.argv[1], int(sys.argv[2])
    rows = []
    for name, make in SCEN.items():
        for r in range(reps):
            rng = np.random.default_rng([zlib.crc32(name.encode()), r])
            x, lab = make(rng)
            g = gates(cdist(x, x), np.asarray(lab), rng)
            rows.append({"scenario": name, "rep": r, **g})
        df = pd.DataFrame([w for w in rows if w["scenario"] == name])
        print(name, {c: round(float(df[c].mean()), 2) for c in df.columns
                     if c not in ("scenario", "rep", "dip_p")}, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
