"""Monte Carlo reproducibility: the same data clustered with different random seeds.

    python validation/seed_reproducibility.py --data-seeds 3 --mc-seeds 8 --jobs 2 \
        --out validation/results/seed_reproducibility.csv

For every scenario and data sample (``--data-seeds``), the clustering is run with
``--mc-seeds`` different random seeds of the resampling. The resampling rule
promises that, apart from candidates flagged borderline, the result does not
depend on that seed; the table records, per run, the final partition (as a
hash of its labels), the number of clusters, the subsamples used by the
resolution selection and the time.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT = ["top_null_banana", "top_pos_3groups_4sigma", "refine_pos_5sigma"]


def run_one(scenario: str, data_seed: int, mc_seed: int) -> dict:
    from scenarios import SCENARIOS
    from scipy.spatial.distance import cdist

    from struct2clust.config import Config
    from struct2clust.pipeline import Run, cluster, run_conclusion
    from struct2clust.structures import Ensemble

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    x, _, replica = SCENARIOS[scenario](np.random.default_rng(data_seed))
    n = len(x)
    frame = [i - replica.index(r) for i, r in enumerate(replica)] if replica else None
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=replica, frame=frame)
    cfg = Config.model_validate({"seed": mc_seed, "n_jobs": 1, "biology": {"enabled": False}})
    run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
              metric_label="", distance_unit="", n_input=n)
    cluster(run)
    sel = run.selection
    sub = [e for e in run.refinement if e.get("n_subsamples")]
    return {
        "scenario": scenario, "data_seed": data_seed, "mc_seed": mc_seed,
        "conclusion": run_conclusion(run)[0],
        "n_final": int(run.n_clusters),
        "partition": hashlib.sha1(
            np.asarray(run.labels, dtype=np.int64).tobytes()).hexdigest()[:10],
        "noise": round(float(np.mean(run.labels < 0)), 4),
        "borderline": bool(sel.borderline),
        "selection_subsamples": int(sel.candidates["n_subsamples"].sum()),
        "refinement_subsamples": int(sum(e["n_subsamples"] for e in sub)),
        "seconds": round(time.time() - t0, 1),
    }


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per scenario and data sample: how many different partitions the seeds gave."""
    rows = []
    for (s, d), g in table.groupby(["scenario", "data_seed"]):
        counts = g["partition"].value_counts()
        rows.append({"scenario": s, "data_seed": d, "runs": len(g),
                     "partitions": len(counts), "agree_with_most": int(counts.iloc[0]),
                     "n_final": "/".join(map(str, sorted(set(g["n_final"])))),
                     "borderline_runs": int(g["borderline"].sum()),
                     "median_seconds": float(g["seconds"].median())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT)
    ap.add_argument("--data-seeds", type=int, default=3)
    ap.add_argument("--mc-seeds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "seed_reproducibility.csv")
    args = ap.parse_args()
    tasks = [(s, d, m) for s in args.scenarios for d in range(args.data_seeds)
             for m in range(args.mc_seeds)]
    from joblib import Parallel, delayed
    records = []
    for r in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(run_one)(*t) for t in tasks):
        records.append(r)
        print(f"{r['scenario']} data {r['data_seed']} seed {r['mc_seed']}: "
              f"{r['n_final']} clusters, {r['seconds']} s", flush=True)
    table = pd.DataFrame(records).sort_values(["scenario", "data_seed", "mc_seed"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    summary = summarise(table)
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
