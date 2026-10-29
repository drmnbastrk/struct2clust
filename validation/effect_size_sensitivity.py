"""Sensitivity of the effect-size rule to its ratio (METHODS.md, section 8).

    python validation/effect_size_sensitivity.py

Two tight sub-groups (SD 0.1 per dimension) inside one group, with medoids f
state scales apart, next to one ordinary Gaussian group; each pair (f, ratio)
on 5 samples. Prints, per sample, the clusters selected -> clusters reported,
with "m" when clusters were merged by the rule and "s" when sub-structure was
reported. Writes validation/results/effect_size_sensitivity.csv.
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    from scenarios import refine_scale

    from struct2clust.config import Config
    from struct2clust.pipeline import Run, cluster
    from struct2clust.structures import Ensemble

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    rows = []
    for f in (0.5, 0.8, 1.5):
        for ratio in (0.5, 1.0, 2.0):
            for seed in range(5):
                x, _, _ = refine_scale(f, sub_sd=0.1)(np.random.default_rng(seed))
                n = len(x)
                ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                               coords=np.zeros((n, 1, 3)), replica=None, frame=None)
                cfg = Config.model_validate({
                    "seed": seed, "n_jobs": 2, "biology": {"enabled": False},
                    "selection": {"refine_min_separation_ratio": ratio}})
                run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
                          metric_label="", distance_unit="", n_input=n)
                cluster(run)
                rows.append({"f": f, "ratio": ratio, "seed": seed,
                             "n_selected": run.selection.n_clusters, "n_final": run.n_clusters,
                             "merged": bool(run.top_level_merges),
                             "substructure": any(e.get("substructure") for e in run.refinement)})
            r = [x for x in rows if x["f"] == f and x["ratio"] == ratio]
            print(f"f={f} ratio={ratio}: " + " ".join(
                f"{x['n_selected']}->{x['n_final']}{'m' if x['merged'] else ''}"
                f"{'s' if x['substructure'] else ''}" for x in r), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "effect_size_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
