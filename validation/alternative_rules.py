"""The resolution rule compared with other ways of choosing an HDBSCAN setting.

    python validation/alternative_rules.py --samples 5 --jobs 2

Three synthetic controls with a known answer (6 dimensions, Euclidean distances):

  nested      two groups 40 sigma apart, each made of three sub-groups 8 sigma apart
              (50 points each): 6 states
  noise       300 points from one Gaussian: no states
  background  three groups 8 sigma apart (80 points each) and 60 points spread
              uniformly over their bounding box: 3 states, background as noise

Rules, all applied to the same candidate settings (clustering.candidate_grid):

  used            finest partition whose clusters all have mean Jaccard >= 0.75,
                  decided by the sequential test (selection.select_partition)
  jaccard_0.6     the same with the threshold 0.6 (100 subsamples per candidate)
  most_stable     the candidate with the most clusters of mean Jaccard >= 0.75,
                  unstable clusters ignored (100 subsamples)
  ari             highest mean adjusted Rand index between the clusterings of two
                  subsamples, on their shared points (Ben-Hur et al. 2002; 50 pairs)
  dbcv            highest density-based clustering validation index (Moulavi et al.
                  2014; hdbscan.validity.validity_index, needs the hdbscan package)

Writes validation/results/alternative_rules.csv (one row per control, sample and
rule: number of clusters chosen, ARI to the truth, noise) and prints a summary.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DIM = 6


def nested(rng):
    centres = []
    for top in (0.0, 40.0):
        for k in range(3):
            centres.append(np.r_[top, 8.0 * k, np.zeros(DIM - 2)])
    x = np.vstack([rng.normal(c, 1.0, (50, DIM)) for c in centres])
    return x, np.repeat(np.arange(6), 50)


def noise(rng):
    return rng.normal(0, 1, (300, DIM)), np.zeros(300, dtype=int)


def background(rng):
    centres = [np.r_[8.0 * k, np.zeros(DIM - 1)] for k in range(3)]
    groups = np.vstack([rng.normal(c, 1.0, (80, DIM)) for c in centres])
    lo, hi = groups.min(0), groups.max(0)
    back = rng.uniform(lo, hi, (60, DIM))
    return np.vstack([groups, back]), np.r_[np.repeat(np.arange(3), 80), np.full(60, -1)]


CONTROLS = {"nested": nested, "noise": noise, "background": background}


def scores(dist, labels, params, seed, n_jaccard=100, n_pairs=50):
    """Per-cluster mean Jaccard (n_jaccard subsamples) and mean pairwise subsample ARI."""
    from sklearn.metrics import adjusted_rand_score

    from struct2clust.clustering import hdbscan_labels, subsample
    from struct2clust.stability import one_resample, subsample_params

    blocks = np.arange(len(dist))
    seeds = np.random.SeedSequence(seed).generate_state(n_jaccard + 2 * n_pairs)
    jac = np.nanmean([one_resample(dist, labels, blocks, params, 0.8, int(s))[1]
                      for s in seeds[:n_jaccard]], axis=0)
    aris = []
    for p in range(n_pairs):
        sets = []
        for s in seeds[n_jaccard + 2 * p:n_jaccard + 2 * p + 2]:
            idx = subsample(np.random.default_rng(int(s)), blocks, 0.8)
            lab = hdbscan_labels(dist[np.ix_(idx, idx)], *subsample_params(params, 0.8))
            sets.append(dict(zip(idx.tolist(), lab.tolist())))
        shared = sorted(set(sets[0]) & set(sets[1]))
        aris.append(adjusted_rand_score([sets[0][i] for i in shared],
                                        [sets[1][i] for i in shared]))
    return jac, float(np.mean(aris))


def run_one(control: str, sample: int) -> list[dict]:
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score

    from struct2clust.clustering import candidate_grid, hdbscan_labels, relabel_by_size
    from struct2clust.selection import select_partition

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    x, truth = CONTROLS[control](np.random.default_rng(sample))
    dist = cdist(x, x)
    grid = candidate_grid(len(dist))
    labelings = [relabel_by_size(hdbscan_labels(dist, *g)) for g in grid]
    rows = []
    for i, (g, lab) in enumerate(zip(grid, labelings)):
        if lab.max() < 1:
            continue
        jac, ari = scores(dist, lab, g, seed=1000 * sample + i)
        try:
            from hdbscan.validity import validity_index
            dbcv = float(validity_index(x, lab)) if (lab >= 0).sum() > 2 else np.nan
        except ImportError:
            dbcv = np.nan
        rows.append({"i": i, "k": int(lab.max() + 1), "noise": float(np.mean(lab < 0)),
                     "weakest": float(jac.min()), "mean": float(jac.mean()),
                     "n_stable": int((jac >= 0.75).sum()), "ari": ari, "dbcv": dbcv})
    cand = pd.DataFrame(rows)
    picks = {}
    sel = select_partition(dist, np.arange(len(dist)), seed=sample, n_jobs=1, quiet=True)
    picks["used"] = (grid.index(sel.params) if sel.stable and sel.found_clusters else None)
    q = cand[cand["weakest"] >= 0.6]
    picks["jaccard_0.6"] = (int(q.sort_values(["k", "mean"], ascending=False).iloc[0]["i"])
                            if len(q) else None)
    picks["most_stable"] = (int(cand.sort_values(["n_stable", "mean"], ascending=False)
                                .iloc[0]["i"]) if cand["n_stable"].max() > 0 else None)
    picks["ari"] = int(cand.sort_values("ari", ascending=False).iloc[0]["i"])
    picks["dbcv"] = (int(cand.sort_values("dbcv", ascending=False).iloc[0]["i"])
                     if cand["dbcv"].notna().any() else None)
    out = []
    for rule, i in picks.items():
        if i is None:
            out.append({"control": control, "sample": sample, "rule": rule, "n_clusters": 0,
                        "ari_to_truth": np.nan, "noise": np.nan})
            continue
        lab = labelings[i]
        out.append({"control": control, "sample": sample, "rule": rule,
                    "n_clusters": int(lab.max() + 1),
                    "ari_to_truth": round(float(adjusted_rand_score(truth, lab)), 3),
                    "noise": round(float(np.mean(lab < 0)), 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "alternative_rules.csv")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    tasks = [(c, s) for c in CONTROLS for s in range(args.samples)]
    rows = [r for res in Parallel(n_jobs=args.jobs)(delayed(run_one)(*t) for t in tasks)
            for r in res]
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = table.groupby(["rule", "control"]).agg(
        clusters=("n_clusters", lambda v: "/".join(map(str, v))),
        ari=("ari_to_truth", "median")).unstack("control")
    print(summary.to_string())


if __name__ == "__main__":
    main()
