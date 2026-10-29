"""Simulation study of the minimum number of groups for refinement (input.groups).

    python validation/min_groups_study.py run --seeds 20 --jobs 2
    python validation/min_groups_study.py table

A cluster of 180 points next to a far-away group (150 points from 10 runs, at 80 on
the first axis, as in the refine_pos scenarios of scenarios.py), resampled by run.
The inner cluster is made of 3r runs, r = 2, 3, 4 (6, 9 or 12 groups):

  mg_pos_r<r>         3 sub-groups 6 sigma apart, each from r runs (run offsets SD 0.7,
                      spread inside a run SD 0.7): refinement should find the 3
  mg_null_r<r>        one structureless group from 3r runs (SD 0.7, 0.7): no split
  mg_nullstrong_r<r>  the same with stronger run effects (run SD 1.0, inside 0.5)

Every run is made without a minimum (MIN_GROUPS = 1) and records every refinement
node with its number of groups. The result for a minimum T follows from the same
runs: a node splits when it split in the run, has >= T groups and its parent split
(node seeds depend only on the node's path, so this equals running with T).
Runs: runs_min_groups/; table: results/min_groups.csv.

Per run and T, `table` counts
  false split (null)  the inner group ends in more than one final cluster;
  found (pos)         each sub-group is >= 90% of one final cluster, and that cluster
                      holds >= 90% of the sub-group.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs_min_groups"
SCENARIOS = [f"mg_{k}_r{r}" for k in ("pos", "null", "nullstrong") for r in (2, 3, 4)]
THRESHOLDS = [1, 4, 6, 8, 10]


def make(scenario, seed):
    """(points, truth, run): truth 0-2 sub-groups (or 0 for the null), last label far group."""
    import group_study as gs

    rng = np.random.default_rng(seed)
    kind, r = scenario[3:].split("_r")
    r = int(r)
    if kind == "pos":
        x, truth, run = gs.batches(rng, gs.centres(3, 6.0), r, 60 // r, 0.7, 0.7)
    else:
        sd_run, sd_in = (1.0, 0.5) if kind == "nullstrong" else (0.7, 0.7)
        x, _, run = gs.batches(rng, gs.centres(1, 0), 3 * r, 60 // r, sd_run, sd_in)
        truth = np.zeros(len(x), int)
    fx, _, frun = gs.batches(rng, [np.r_[80.0, np.zeros(gs.D - 1)]], 10, 15, 0.7, 0.7)
    return (np.vstack([x, fx]), np.r_[truth, np.full(len(fx), truth.max() + 1)],
            np.r_[run, frun + run.max() + 1])


def run_one(scenario: str, seed: int) -> dict:
    from scipy.spatial.distance import cdist

    from struct2clust import hierarchy, pipeline
    from struct2clust.config import Config
    from struct2clust.pipeline import Run, cluster, run_conclusion
    from struct2clust.structures import Ensemble

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    hierarchy.MIN_GROUPS = pipeline.MIN_GROUPS = 1
    t0 = time.time()
    x, truth, group = make(scenario, seed)
    n = len(x)
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=None, frame=None)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "groups.csv"
        path.write_text("structure,group\n" + "".join(f"{i},{g}\n" for i, g in enumerate(group)))
        cfg = Config.model_validate({"seed": seed, "n_jobs": 1, "biology": {"enabled": False},
                                     "input": {"groups": str(path), "group_fraction": 0.5}})
        run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
                  metric_label="", distance_unit="", n_input=n)
        cluster(run)
    return {
        "scenario": scenario, "seed": seed, "n_groups": int(len(np.unique(group))),
        "conclusion": run_conclusion(run)[0],
        "n_top": int(run.groups.max() + 1), "n_final": int(run.n_clusters),
        "nodes": [{k: e.get(k) for k in ("cluster", "size", "clusterability", "n_groups",
                                         "split", "n_subclusters", "reason")}
                  for e in run.refinement],
        # truth counts per cluster, to tell which node is the inner group
        "top_truth": {int(g): np.bincount(truth[run.groups == g]).tolist()
                      for g in np.unique(run.groups) if g >= 0},
        "final_truth": {int(c): np.bincount(truth[run.labels == c]).tolist()
                        for c in np.unique(run.labels) if c >= 0},
        "seconds": round(time.time() - t0, 1),
    }


def final_counts(rec, t):
    """Truth counts of the final clusters with minimum t. These scenarios have one level
    of refinement (checked), so a refused split leaves the top level as it was."""
    split = [e for e in rec["nodes"] if e["split"]]
    assert all("." not in e["cluster"] for e in split), "deeper refinement not handled"
    if all((e["n_groups"] or 0) >= t for e in split):
        return list(rec["final_truth"].values())
    return list(rec["top_truth"].values())


def found(counts):
    def ok(k):
        return any(len(c) > k and c[k] >= 0.9 * 60 and c[k] >= 0.9 * sum(c) for c in counts)
    return all(ok(k) for k in range(3))


def false_split(counts):
    return sum(1 for c in counts if c and c[0] > 0) > 1


def clopper_pearson(k, n):
    """95% Clopper-Pearson interval of a rate k / n."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def table():
    import pandas as pd

    rows = []
    for sc in SCENARIOS:
        recs = [json.loads(f.read_text()) for f in sorted(RUNS.glob(f"{sc}__seed*.json"))]
        measure = found if "_pos_" in sc else false_split
        row = {"scenario": sc, "measure": "found_all_3" if "_pos_" in sc else "false_split",
               "inner_groups": 3 * int(sc[-1]), "n": len(recs)}
        for t in THRESHOLDS:
            k = sum(measure(final_counts(r, t)) for r in recs)
            lo, hi = clopper_pearson(k, len(recs))
            row[f"T{t}"] = f"{k}/{len(recs)} ({100 * lo:.0f}-{100 * hi:.0f}%)"
        rows.append(row)
    out = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "min_groups.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["run", "table"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--scenarios", nargs="*")
    args = ap.parse_args()
    if args.what == "table":
        table()
        return
    RUNS.mkdir(exist_ok=True)
    todo = [(s, k) for s in (args.scenarios or SCENARIOS) for k in range(args.seeds)
            if not (RUNS / f"{s}__seed{k:03d}.json").exists()]
    print(f"min_groups: {len(todo)} runs to go", flush=True)

    def save(r):
        path = RUNS / f"{r['scenario']}__seed{r['seed']:03d}.json"
        path.with_suffix(".tmp").write_text(json.dumps(r, indent=1))
        path.with_suffix(".tmp").replace(path)
        print(f"{r['scenario']} seed {r['seed']}: {r['conclusion']} {r['n_final']} "
              f"({r['seconds']} s)", flush=True)

    if args.jobs > 1:
        from joblib import Parallel, delayed
        for r in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(run_one)(s, k) for s, k in todo):
            save(r)
    else:
        for s, k in todo:
            save(run_one(s, k))


if __name__ == "__main__":
    main()
