"""Simulation study: every scenario x seed through the full clustering step.

    python validation/run_study.py --seeds 10 --jobs 2          # workstation
    python validation/run_study.py --seeds 50 --index $SLURM_ARRAY_TASK_ID --chunk 5
                                                               # one slice on a cluster

Every run is written to validation/runs/<scenario>__seed<NNN>.json as soon as it
finishes; runs that already exist are skipped, so an interrupted study can be
restarted. Summarise with validation/summarise.py.

    python validation/run_study.py --without-gate --scenarios refine_null_gauss ...

runs the same scenarios with the clusterability test switched off (stability
alone decides every split) and writes them to validation/runs_without_gate/.

    python validation/run_study.py --single-pass --scenarios traj_pos_markov ...

runs trajectory scenarios with input.membership_blocks off (no second pass with blocks of
one membership autocorrelation time) and writes them to validation/runs_single_pass/.

Scenarios with replica labels are run as trajectories (input.type trajectory: automatic
blocks, resampled like groups, and the sampling check), as struct2clust run does.

The trajectory scenarios take the longest (5-15 CPU-minutes per run). The study keeps
every core it is given busy for a long time, so use a compute cluster or a machine with
adequate cooling.
"""
from __future__ import annotations

import os

# One thread per process: parallelism comes from running several runs at once.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "runs"
# scenario-specific settings; everything else runs with the defaults
SELECTION = {"refine_scale": {"refine_min_separation_ratio": 1.0}}   # the optional rule
RUNS_WITHOUT_GATE = HERE / "runs_without_gate"
RUNS_SINGLE_PASS = HERE / "runs_single_pass"
# trajectory mode without files: the paths are never read, run_one passes the frames
TRAJECTORY_INPUT = {"type": "trajectory", "topology": "frames.pdb", "trajectory": "frames.xtc"}
TIERS = ["LOW", "MODERATE", "HIGH"]


def run_one(scenario: str, seed: int, gate: bool = True,
            membership_blocks: bool = True) -> dict:
    from scenarios import SCENARIOS
    from scipy.spatial.distance import cdist
    from sklearn.metrics import adjusted_rand_score

    from struct2clust.config import Config
    from struct2clust.pipeline import (
        Run,
        cluster,
        conclusion_if_sampling_sufficient,
        run_conclusion,
    )
    from struct2clust.structures import Ensemble

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    x, truth, replica = SCENARIOS[scenario](np.random.default_rng(seed))
    n = len(x)
    frame = [i - replica.index(r) for i, r in enumerate(replica)] if replica else None
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=replica, frame=frame)
    inp = {**TRAJECTORY_INPUT, "membership_blocks": membership_blocks} if replica else {}
    cfg = Config.model_validate({"seed": seed, "n_jobs": 1, "biology": {"enabled": False},
                                 "clusterability": {"enabled": gate}, "input": inp,
                                 "selection": SELECTION.get(scenario.rsplit("_", 1)[0], {})})
    run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
              metric_label="", distance_unit="", n_input=n)
    cluster(run)
    ca, sel = run.clusterability, run.selection
    tiers = list(run.stability.clusters["tier"]) if run.stability is not None else []
    check = run.sampling_check or {}
    first = run.first_pass or {}
    return {
        "scenario": scenario, "seed": seed,
        "verdict": ca.verdict if ca else None,
        "dip_p": float(ca.dip_pvalue) if ca else None,
        "spectral_k": int(ca.spectral_k) if ca else None,
        "reproducible": bool(sel.stable),
        "conclusion": run_conclusion(run)[0],
        # trajectories: the sampling check and what it replaced
        "sampling_sufficient": check.get("sufficient"),
        "longer_than_block": len(check.get("clusters_longer_than_block", [])),
        "without_transitions": len(check.get("clusters_without_transitions", [])),
        "conclusion_if_sampling_sufficient": conclusion_if_sampling_sufficient(run),
        "blocks": [t["n_blocks"] for t in run.sampling],
        "second_pass": bool(first),
        "first_pass_conclusion": first.get("conclusion"),
        "n_selected": int(sel.n_clusters),
        "n_top": int(run.groups.max() + 1) if run.groups is not None else int(sel.n_clusters),
        "n_final": int(run.n_clusters),
        "top_level_merges": len(run.top_level_merges),
        "substructure": [e["cluster"] for e in run.refinement if e.get("substructure")],
        "splits": [e["cluster"] for e in run.refinement if e["split"]],
        "tiers": tiers, "best_tier": max(tiers, key=TIERS.index) if tiers else "",
        "ari_top": float(adjusted_rand_score(truth, run.groups)),
        "ari_final": float(adjusted_rand_score(truth, run.labels)),
        "noise_top": float(np.mean(run.groups < 0)),
        "noise_final": float(np.mean(run.labels < 0)),
        "seconds": round(time.time() - t0, 1),
    }


def save(record: dict, runs: Path) -> None:
    path = runs / f"{record['scenario']}__seed{record['seed']:03d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)


def main() -> None:
    from scenarios import SCENARIOS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--scenarios", nargs="*", help="Default: all.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel runs (one core each).")
    ap.add_argument("--index", type=int, help="Run only slice number INDEX ...")
    ap.add_argument("--chunk", type=int, default=5, help="... of CHUNK runs.")
    ap.add_argument("--without-gate", action="store_true",
                    help="Switch the clusterability test off (written to runs_without_gate/).")
    ap.add_argument("--single-pass", action="store_true",
                    help="input.membership_blocks off for trajectory scenarios "
                         "(written to runs_single_pass/).")
    args = ap.parse_args()
    runs, gate = (RUNS_WITHOUT_GATE, False) if args.without_gate else (RUNS, True)
    if args.single_pass:
        runs = RUNS_SINGLE_PASS

    names = args.scenarios or list(SCENARIOS)
    tasks = [(s, k) for s in names for k in range(args.seeds)]
    if args.index is not None:
        tasks = tasks[args.index * args.chunk:(args.index + 1) * args.chunk]
    runs.mkdir(exist_ok=True)
    todo = [(s, k) for s, k in tasks if not (runs / f"{s}__seed{k:03d}.json").exists()]
    print(f"{len(tasks) - len(todo)} runs done, {len(todo)} to go", flush=True)
    if args.jobs > 1:
        from joblib import Parallel, delayed
        for record in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(run_one)(s, k, gate, not args.single_pass) for s, k in todo):
            save(record, runs)
            print(f"{record['scenario']} seed {record['seed']}: {record['seconds']} s",
                  flush=True)
    else:
        for s, k in todo:
            record = run_one(s, k, gate, not args.single_pass)
            save(record, runs)
            print(f"{s} seed {k}: {record['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
