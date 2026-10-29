"""Simulation study of resampling by group (input.groups) and of the subsample fraction.

    python validation/group_study.py grouped --seeds 20 --jobs 2
    python validation/group_study.py frac05  --seeds 20 --jobs 2

grouped   every scenario with groups: the dependence scenarios below (group =
          entry or run), and scenarios of run_study.py whose points are given
          30 random sources unrelated to their structure (the cost of
          grouping when sources carry no structure). Runs: runs_grouped/.
frac05    scenarios of run_study.py without groups, with the subsample
          fraction of the selection and the stability at 0.5 instead of 0.8
          (Hennig's subsetting uses half of the data). Runs: runs_frac05/.

Dependence scenarios (group labels in brackets):
  dep_null_copies    40 structureless entries x 6 near-identical copies [entry];
                     the top_null_copies scenario of run_study.py
  dep_null_batches   one structureless group from 30 runs of 10; every run is
                     offset (SD 1) and tight inside (SD 0.3), like the models of
                     one prediction run [run]: no distinct states
  dep_pos_copies     3 groups 6 sigma apart, each of 12 entries x 6 copies
                     [entry]: the 3 groups
  dep_pos_batches    3 groups 6 sigma apart, each from 10 runs of 10 with run
                     offsets (SD 0.5, inside 0.5) [run]: the 3 groups
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
D = 6
TIERS = ["LOW", "MODERATE", "HIGH"]
N_RANDOM_SOURCES = 30


def batches(rng, centres, n_runs, per_run, sd_run, sd_in):
    x, run, truth = [], [], []
    for t, c in enumerate(centres):
        for _ in range(n_runs):
            offset = rng.normal(0, sd_run, D)
            x.append(c + offset + rng.normal(0, sd_in, (per_run, D)))
            run += [len(set(run))] * per_run
            truth += [t] * per_run
    return np.vstack(x), np.array(truth), np.array(run)


def copies(rng, centres, n_entries, n_copies, sd_entry=1.0, sd_copy=0.05):
    return batches(rng, centres, n_entries, n_copies, sd_entry, sd_copy)


def centres(k, sep):
    return [np.r_[i * sep, np.zeros(D - 1)] for i in range(k)]


DEPENDENCE = {
    "dep_null_copies": lambda rng: copies(rng, centres(1, 0), 40, 6),
    "dep_null_batches": lambda rng: batches(rng, centres(1, 0), 30, 10, 1.0, 0.3),
    "dep_pos_copies": lambda rng: copies(rng, centres(3, 6.0), 12, 6),
    "dep_pos_batches": lambda rng: batches(rng, centres(3, 6.0), 10, 10, 0.5, 0.5),
}
WITH_RANDOM_SOURCES = ["top_null_gauss", "top_null_elongated", "top_null_banana",
                       "top_pos_2groups_6sigma", "top_pos_3groups_5sigma",
                       "top_pos_3groups_6sigma", "refine_null_gauss", "refine_pos_5sigma",
                       "refine_pos_6sigma"]
FRAC05 = ["top_null_gauss", "top_null_uniform", "top_null_elongated", "top_null_banana",
          "top_null_gradient", "top_null_md_drift", "top_null_copies",
          "top_pos_2groups_6sigma", "top_pos_3groups_4sigma", "top_pos_3groups_5sigma",
          "top_pos_3groups_6sigma", "top_pos_3elongated_6sigma", "refine_null_gauss",
          "refine_null_banana", "refine_pos_4sigma", "refine_pos_5sigma", "refine_pos_6sigma"]


def make(scenario, seed, mode):
    """(points, truth, replica or None, groups or None)."""
    from scenarios import SCENARIOS
    rng = np.random.default_rng(seed)
    if scenario in DEPENDENCE:
        x, truth, group = DEPENDENCE[scenario](rng)
        return x, truth, None, group
    x, truth, replica = SCENARIOS[scenario](rng)
    if mode == "grouped":
        group = np.random.default_rng(10_000 + seed).integers(0, N_RANDOM_SOURCES, len(x))
        return x, truth, replica, group
    return x, truth, replica, None


def run_one(scenario: str, seed: int, mode: str) -> dict:
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score

    from struct2clust.config import Config
    from struct2clust.pipeline import Run, cluster, run_conclusion
    from struct2clust.structures import Ensemble

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    x, truth, replica, group = make(scenario, seed, mode)
    n = len(x)
    frame = [i - replica.index(r) for i, r in enumerate(replica)] if replica else None
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=replica, frame=frame)
    raw = {"seed": seed, "n_jobs": 1, "biology": {"enabled": False}}
    with tempfile.TemporaryDirectory() as tmp:
        if group is not None:
            path = Path(tmp) / "groups.csv"
            path.write_text("structure,group\n"
                            + "".join(f"{i},{g}\n" for i, g in enumerate(group)))
            raw["input"] = {"groups": str(path)}
        if mode == "frac05":
            raw["selection"] = {"subsample_fraction": 0.5}
            raw["stability"] = {"subsample_fraction": 0.5}
        cfg = Config.model_validate(raw)
        run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
                  metric_label="", distance_unit="", n_input=n)
        cluster(run)
    ca, sel = run.clusterability, run.selection
    tiers = list(run.stability.clusters["tier"]) if run.stability is not None else []
    return {
        "mode": mode, "scenario": scenario, "seed": seed,
        "n_groups": int(len(np.unique(group))) if group is not None else None,
        "verdict": ca.verdict if ca else None,
        "reproducible": bool(sel.stable),
        "conclusion": run_conclusion(run)[0],
        "n_selected": int(sel.n_clusters),
        "n_top": int(run.groups.max() + 1) if run.groups is not None else int(sel.n_clusters),
        "n_final": int(run.n_clusters),
        "splits": [e["cluster"] for e in run.refinement if e["split"]],
        "tiers": tiers, "best_tier": max(tiers, key=TIERS.index) if tiers else "",
        "ari_top": float(adjusted_rand_score(truth, run.groups)),
        "ari_final": float(adjusted_rand_score(truth, run.labels)),
        "noise_final": float(np.mean(run.labels < 0)),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["grouped", "frac05"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--scenarios", nargs="*")
    args = ap.parse_args()
    runs = HERE / f"runs_{args.mode}"
    runs.mkdir(exist_ok=True)
    names = args.scenarios or (list(DEPENDENCE) + WITH_RANDOM_SOURCES
                               if args.mode == "grouped" else FRAC05)
    todo = [(s, k) for s in names for k in range(args.seeds)
            if not (runs / f"{s}__seed{k:03d}.json").exists()]
    print(f"{args.mode}: {len(todo)} runs to go", flush=True)

    def save(r):
        path = runs / f"{r['scenario']}__seed{r['seed']:03d}.json"
        path.with_suffix(".tmp").write_text(json.dumps(r, indent=1))
        path.with_suffix(".tmp").replace(path)
        print(f"{r['scenario']} seed {r['seed']}: {r['conclusion']} {r['n_final']} "
              f"({r['seconds']} s)", flush=True)

    if args.jobs > 1:
        from joblib import Parallel, delayed
        for r in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(run_one)(s, k, args.mode) for s, k in todo):
            save(r)
    else:
        for s, k in todo:
            save(run_one(s, k, args.mode))


if __name__ == "__main__":
    main()
