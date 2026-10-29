"""Sensitivity of a finished run to the stability threshold and the subsample fraction.

    python validation/threshold_sensitivity.py example/results/tar --jobs 2

The clustering of a finished run (its distance matrix and configuration from
run_summary.json are reused) is repeated with the Jaccard threshold stability.high at
0.70, 0.75 and 0.80 and with the subsample fraction at 0.5 and 0.8 (for ensembles
without groups, selection and stability; with groups or automatic trajectory blocks,
input.group_fraction). Per setting: conclusion, clusters, noise, the adjusted Rand
index to the partition of the run, and whether the setting is the run's own. A
conclusion that holds at every setting does not depend on these conventions
(docs/METHODS.md, section 13, "Resampling conventions"); one that changes is close to
the threshold and should be read with that in mind.

Writes validation/results/threshold_sensitivity_<run>.csv.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
THRESHOLDS = [0.70, 0.75, 0.80]
FRACTIONS = [0.5, 0.8]


def main() -> None:
    from sklearn.metrics import adjusted_rand_score

    from struct2clust.config import Config
    from struct2clust.pipeline import Run, cluster, run_conclusion
    from struct2clust.structures import Ensemble

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="Finished run directory.")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    summary = json.loads((args.run / "run_summary.json").read_text(encoding="utf-8"))
    dist = np.load(args.run / "distance_matrix.npy")
    table = pd.read_csv(args.run / "clusters.csv")
    names = (args.run / "distance_matrix_labels.txt").read_text().split()
    if list(table["structure"]) != names:
        raise SystemExit("clusters.csv and the distance matrix list different structures")
    replica = list(table["replica"].astype(str)) if "replica" in table else None
    frame = list(table["frame"]) if "frame" in table else None
    ens = Ensemble(names=names, sources=[None] * len(names), atoms=None,
                   coords=np.zeros((len(names), 1, 3)), replica=replica, frame=frame)
    reported = np.asarray(table["cluster"]) - 1          # 0-based, noise -1
    base = Config.model_validate(summary["config"])
    own = (base.stability.high, base.selection_fraction)

    rows = []
    for t in THRESHOLDS:
        for f in FRACTIONS:
            c = base.model_copy(deep=True)
            c.stability.high = t
            if c.by_unit:
                c.input.group_fraction = f
            else:
                c.selection.subsample_fraction = f
                c.stability.subsample_fraction = f
            c.n_jobs = args.jobs
            c.biology.enabled = False
            c.figures.enabled = False
            run = Run(config=c, out_dir=None, ensemble=ens, distances=dist, metric_label="",
                      distance_unit="", n_input=len(ens))
            t0 = time.time()
            cluster(run)
            rows.append({
                "threshold": t, "fraction": f,
                "own_setting": bool(np.isclose(t, own[0]) and np.isclose(f, own[1])),
                "conclusion": run_conclusion(run)[0], "n_clusters": int(run.n_clusters),
                "noise": round(float(np.mean(run.labels < 0)), 4),
                "ari_to_run": round(float(adjusted_rand_score(reported, run.labels)), 3),
                "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"threshold_sensitivity_{args.run.name}.csv", index=False)


if __name__ == "__main__":
    main()
