"""Sensitivity of the sequential test to its bet limit (selection.BET_LIMIT).

    python validation/bet_limit_sensitivity.py synthetic --reps 500 --jobs 2
    python validation/bet_limit_sensitivity.py examples example/results/endothiapepsin \
        example/results/eralpha --jobs 2

The bet limit caps the share of the wealth staked on one value (default 0.75;
Waudby-Smith & Ramdas 2024 suggest values between 1/2 and 3/4). Any limit below 1
keeps the wealth positive, so the type-I guarantee (Ville's inequality) holds for
every limit; the limit changes how fast a decision is reached.

synthetic  One cluster, per-subsample values drawn from a known distribution with
           mean mu around the threshold 0.75: a Beta distribution (mean mu,
           concentration 4) and a mixture (1 with probability q, otherwise
           uniform on [0.3, 1]). For each limit, mean and level alpha, the test is
           run as in selection.evaluate (batches of 10, at least 20, at most 1000
           values) and the decision and number of values are recorded. Wrong
           decisions are those on the wrong side of the true mean; their rate is
           compared with alpha (a finite simulation checks, it does not prove,
           the bound).
examples   The resolution selection of finished runs (their distance matrices of
           clustered structures) repeated with each limit: selected setting,
           agreement with the partition of the default limit, borderline
           candidates, subsamples used and time.

Writes validation/results/bet_limit_<mode>.csv (or --out) and prints a summary.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
LIMITS = [0.5, 0.75, 0.9]
MEANS = [0.65, 0.70, 0.73, 0.74, 0.75, 0.76, 0.77, 0.80, 0.90]
ALPHAS = [0.05, 1e-5]          # 1e-5: about the per-candidate level of a default run
THRESHOLD = 0.75


def draw(dist: str, mu: float, n: int, rng) -> np.ndarray:
    if dist == "beta":
        c = 4.0
        return rng.beta(mu * c, (1 - mu) * c, n)
    q = (mu - 0.65) / 0.35                       # mean of q*1 + (1-q)*U(0.3, 1)
    return np.where(rng.random(n) < q, 1.0, rng.uniform(0.3, 1.0, n))


def one_sequence(limit, dist, mu, alpha, seed):
    from struct2clust import selection

    selection.BET_LIMIT = limit
    rng = np.random.default_rng(seed)
    x = draw(dist, mu, 1000, rng)
    test = selection.BettingTest(THRESHOLD)
    decision, n = "borderline", 1000
    for start in range(0, 1000, selection.BATCH):
        for v in x[start:start + selection.BATCH]:
            test.update(float(v))
        if start + selection.BATCH < 20:
            continue
        if test.below(alpha):
            decision, n = "below", start + selection.BATCH
            break
        if test.above(alpha):
            decision, n = "above", start + selection.BATCH
            break
    wrong = (decision == "above" and mu < THRESHOLD) or (decision == "below" and mu >= THRESHOLD)
    return {"limit": limit, "distribution": dist, "mean": mu, "alpha": alpha, "seed": seed,
            "decision": decision, "n_values": n, "wrong": wrong}


def synthetic(reps: int, jobs: int) -> pd.DataFrame:
    tasks = [(lim, d, mu, a, s) for lim in LIMITS for d in ("beta", "mixture") for mu in MEANS
             for a in ALPHAS for s in range(reps)]
    rows = Parallel(n_jobs=jobs, batch_size=200)(delayed(one_sequence)(*t) for t in tasks)
    return pd.DataFrame(rows)


def examples(runs: list[Path], jobs: int) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score

    from struct2clust import selection
    from struct2clust.clustering import hdbscan_labels
    from struct2clust.pipeline import top_level_risk

    rows = []
    for run in runs:
        dist = np.load(run / "distance_matrix.npy")
        reference = None
        for limit in [0.75] + [x for x in LIMITS if x != 0.75]:
            selection.BET_LIMIT = limit
            t0 = time.time()
            sel = selection.select_partition(dist, np.arange(len(dist)), seed=42, n_jobs=jobs,
                                             quiet=True, resampling_risk=top_level_risk(0.001))
            seconds = time.time() - t0
            labels = hdbscan_labels(dist, *sel.params) if sel.found_clusters else None
            if limit == 0.75:
                reference = labels
            table = sel.candidates
            rows.append({
                "run": run.name, "limit": limit, "n_structures": len(dist),
                "n_clusters": sel.n_clusters, "reproducible": sel.stable,
                "min_cluster_size": sel.min_cluster_size, "min_samples": sel.min_samples,
                "ari_with_limit_0.75": round(float(adjusted_rand_score(reference, labels)), 4)
                if labels is not None and reference is not None else None,
                "candidates_evaluated": int(table["evaluated"].sum()),
                "borderline_candidates": int(table["borderline"].sum()),
                "subsamples_total": int(table["n_subsamples"].sum()),
                "seconds": round(seconds, 1)})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["synthetic", "examples"])
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, help="Output CSV (default: validation/results/).")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    out = args.out or HERE / "results" / f"bet_limit_{args.mode}.csv"
    if args.mode == "synthetic":
        t = synthetic(args.reps, args.jobs)
        t.to_csv(out, index=False)
        s = t.groupby(["alpha", "distribution", "mean", "limit"]).agg(
            wrong=("wrong", "mean"), borderline=("decision", lambda d: (d == "borderline").mean()),
            mean_values=("n_values", "mean")).round(4)
        print(s.to_string())
    else:
        t = examples(args.runs, args.jobs)
        t.to_csv(out, index=False)
        print(t.to_string())


if __name__ == "__main__":
    main()
