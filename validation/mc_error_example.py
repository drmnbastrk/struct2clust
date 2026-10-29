"""How large Monte Carlo error is in a real ensemble: few versus many subsamples.

    python validation/mc_error_example.py example/results/endothiapepsin

For every candidate HDBSCAN setting of a finished run (on its distance matrix of
clustered structures), the weakest-cluster mean Jaccard is estimated from the first
50 and from all 1000 of the same subsamples, and the sequential test of the
resolution selection is applied. Writes <run>/../../../validation/results/
mc_error_<run>.csv and prints the candidates whose estimate from 50 subsamples is
on the other side of 0.75 than the estimate from 1000.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
FEW, MANY, THRESHOLD = 50, 1000, 0.75


def one(dist, labels, params, seed, alpha):
    from struct2clust.selection import evaluate
    from struct2clust.stability import one_resample

    blocks = np.arange(len(dist))
    seeds = np.random.SeedSequence(seed).generate_state(MANY)
    jac = np.array([one_resample(dist, labels, blocks, params, 0.8, int(s))[1] for s in seeds])
    few = np.nanmean(jac[:FEW], axis=0)
    many = np.nanmean(jac, axis=0)
    se = np.nanstd(jac, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(jac), axis=0))
    w = int(np.argmin(many))
    ev = evaluate(dist, labels, blocks, params, 0.8, seed, THRESHOLD, alpha)
    return {"weakest_50": round(float(few.min()), 3), "weakest_1000": round(float(many.min()), 3),
            "weakest_1000_se": round(float(se[w]), 3),
            "sequential_decision": ("qualifies" if ev.qualifies else
                                    "borderline" if ev.borderline else "does not qualify"),
            "sequential_subsamples": ev.n_subsamples}


def main() -> None:
    from joblib import Parallel, delayed

    from struct2clust.clustering import candidate_grid, hdbscan_labels, relabel_by_size

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--min-clusters", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    dist = np.load(args.run / "distance_matrix.npy")
    grid = candidate_grid(len(dist))
    labelings = [relabel_by_size(hdbscan_labels(dist, *g)) for g in grid]
    todo = [i for i, lab in enumerate(labelings) if lab.max() + 1 >= args.min_clusters]
    # the top-level budget (half the default risk) shared by all candidates with >= 2 clusters
    alpha = 0.001 / 2 / sum(lab.max() >= 1 for lab in labelings)
    res = Parallel(n_jobs=args.jobs)(delayed(one)(dist, labelings[i], grid[i], 7 + i, alpha)
                                     for i in todo)
    table = pd.DataFrame([{"min_cluster_size": grid[i][0], "min_samples": grid[i][1],
                           "n_clusters": int(labelings[i].max() + 1), **r}
                          for i, r in zip(todo, res)])
    out = HERE / "results" / f"mc_error_{args.run.name}.csv"
    table.to_csv(out, index=False)
    flip = table[(table["weakest_50"] >= THRESHOLD) != (table["weakest_1000"] >= THRESHOLD)]
    print(f"{len(table)} candidates with >= {args.min_clusters} clusters; "
          f"{len(flip)} on the other side of {THRESHOLD} with {FEW} subsamples:")
    print(flip.to_string(index=False))


if __name__ == "__main__":
    main()
