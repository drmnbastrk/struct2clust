"""Sensitivity of trajectory results to the resampling block length.

    python validation/block_sensitivity.py -c example/1rkj_rna_bound_vs_free.yaml \
        example/results/1RKJ_rna --jobs 2

Two analyses of a finished trajectory run (its distance matrix is reused):

blocking   For every trajectory, the coordinate from which the autocorrelation
           time is estimated (sampling.principal_coordinate) is cut into blocks of
           b = 1, 2, 4, ... frames, and the standard error of its mean is computed
           from the block means (Flyvbjerg & Petersen 1989). For correlated frames
           the error grows with b until the blocks are longer than the correlation
           time, where (SE(b) / SE(1))^2 levels off at 2 tau. Without a plateau the
           largest value is only a lower bound on 2 tau. The table compares this with
           the integrated autocorrelation time of the run.
blocks     The clustering is repeated with input.trajectory_blocks = 5 ... 250
           (block lengths from 100 to 2 frames per trajectory of 500): conclusion,
           clusters, the selection's and the independent stability estimates, and
           subsamples used.

Writes validation/results/block_sensitivity_<run>_{blocking,blocks}.csv.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
N_BLOCKS = [5, 10, 25, 50, 100, 250]


def blocking(x: np.ndarray) -> list[dict]:
    x = np.asarray(x, float)
    se1 = x.std(ddof=1) / np.sqrt(len(x))
    rows, b = [], 1
    while len(x) // b >= 8:
        m = len(x) // b
        means = x[:m * b].reshape(m, b).mean(axis=1)
        se = means.std(ddof=1) / np.sqrt(m)
        rows.append({"block_frames": b, "n_blocks": m, "standard_error": se,
                     "se_ratio": se / se1, "two_tau_lower_bound": (se / se1) ** 2})
        b *= 2
    return rows


def main() -> None:
    from struct2clust.cli import build_config
    from struct2clust.pipeline import Run, load_input, run_conclusion
    from struct2clust.pipeline import cluster as run_cluster
    from struct2clust.sampling import integrated_autocorrelation_time, principal_coordinate

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="Finished run (distance_matrix.npy and labels).")
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    dist = np.load(args.run / "distance_matrix.npy")
    names = (args.run / "distance_matrix_labels.txt").read_text().split()
    cfg = build_config(args.config)
    ens = load_input(cfg, None)
    if list(ens.names) != names:
        raise SystemExit("the run's distance matrix does not match the input order")
    out = HERE / "results"

    rows = []
    replica = np.asarray(ens.replica)
    for name in ens.replicas:
        idx = np.flatnonzero(replica == name)
        x = principal_coordinate(dist[np.ix_(idx, idx)])
        tau, err = integrated_autocorrelation_time(x)
        for r in blocking(x):
            rows.append({"trajectory": name, "tau_integrated": round(tau, 1),
                         "tau_rel_error": round(err, 2), **{k: round(v, 5) if isinstance(v, float)
                                                            else v for k, v in r.items()}})
    pd.DataFrame(rows).to_csv(out / f"block_sensitivity_{args.run.name}_blocking.csv",
                              index=False)
    print(pd.DataFrame(rows).to_string())

    rows = []
    for nb in N_BLOCKS:
        c = cfg.model_copy(deep=True)
        c.input.trajectory_blocks = nb
        c.n_jobs = args.jobs
        c.biology.enabled = False
        c.figures.enabled = False
        run = Run(config=c, out_dir=None, ensemble=ens, distances=dist, metric_label="",
                  distance_unit="Å", n_input=len(ens))
        t0 = time.time()
        run_cluster(run)
        tab = run.stability.clusters if run.stability is not None else None
        rows.append({
            "n_blocks": nb, "block_frames": round(len(ens) / len(ens.replicas) / nb, 1),
            "conclusion": run_conclusion(run)[0],
            "n_clusters": int(run.n_clusters),
            "noise": round(float(np.mean(run.labels < 0)), 4),
            "selection_estimates": run.selection.cluster_estimates,
            "independent_estimates": [round(v, 3) for v in tab["jaccard_mean"]]
            if tab is not None else [],
            "independent_low": [round(v, 3) for v in tab["jaccard_q025"]]
            if tab is not None and "jaccard_q025" in tab else [],
            "borderline": run.selection.borderline,
            "subsamples_selection": int(run.selection.candidates["n_subsamples"].sum()),
            "seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / f"block_sensitivity_{args.run.name}_blocks.csv", index=False)


if __name__ == "__main__":
    main()
