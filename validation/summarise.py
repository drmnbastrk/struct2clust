"""Summarise the simulation study.

    python validation/summarise.py                 # reads validation/runs/*.json
    python validation/summarise.py --from-csv      # reads validation/results/runs.csv
    python validation/summarise.py --without-gate  # runs_without_gate/ -> results/without_gate/
    python validation/summarise.py --single-pass   # runs_single_pass/ -> results/single_pass/

Writes runs.csv (one row per run) and summary.csv (one row per scenario) to
validation/results/, and prints the summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.assign(split=runs["n_splits"] > 0, any_high=runs["best_tier"] == "HIGH",
                       dip_significant=runs["dip_p"] < 0.05, eigengap_k2=runs["spectral_k"] >= 2)
    g = runs.groupby("scenario")
    table = pd.DataFrame({
        "runs": g.size(),
        "dip_significant": g["dip_significant"].mean(),
        "eigengap_k2": g["eigengap_k2"].mean(),
        "clusterable": g["verdict"].apply(lambda v: (v == "CLUSTERABLE").mean()),
        "weak": g["verdict"].apply(lambda v: (v == "WEAK").mean()),
        "reproducible": g["reproducible"].mean(),
        "distinct_states": g["conclusion"].apply(lambda v: (v == "distinct_states").mean()),
        "partition_not_confirmed":
            g["conclusion"].apply(lambda v: (v == "partition_not_confirmed").mean()),
        "insufficient_sampling":
            g["conclusion"].apply(lambda v: (v == "insufficient_sampling").mean()),
        # trajectories: what the resampling gives when the sampling check is set aside
        "distinct_if_sufficient": g["conclusion_if_sampling_sufficient"].apply(
            lambda v: (v == "distinct_states").mean())
        if "conclusion_if_sampling_sufficient" in runs else np.nan,
        "second_pass_rate": g["second_pass"].mean() if "second_pass" in runs else np.nan,
        "any_high": g["any_high"].mean(),
        "n_top_mean": g["n_top"].mean(),
        "n_final_mean": g["n_final"].mean(),
        "split_rate": g["split"].mean(),
        "substructure_rate": g["n_substructure"].apply(lambda v: (v > 0).mean())
        if "n_substructure" in runs else np.nan,
        "top_merge_rate": g["top_level_merges"].apply(lambda v: (v > 0).mean())
        if "top_level_merges" in runs else np.nan,
        "ari_final": g["ari_final"].mean(),
        "noise_final": g["noise_final"].mean(),
        "seconds_median": g["seconds"].median(),
    })
    return table.round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--without-gate", action="store_true",
                    help="Summarise the runs without the clusterability test.")
    ap.add_argument("--single-pass", action="store_true",
                    help="Summarise the trajectory runs with input.membership_blocks off.")
    args = ap.parse_args()
    variant = ("without_gate" if args.without_gate else
               "single_pass" if args.single_pass else "")
    source = HERE / (f"runs_{variant}" if variant else "runs")
    results = RESULTS / variant if variant else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        runs = pd.read_csv(results / "runs.csv")
    else:
        rows = [json.loads(p.read_text()) for p in sorted(source.glob("*.json"))]
        runs = pd.DataFrame(rows)
        runs["n_splits"] = runs.pop("splits").apply(len)
        if "substructure" in runs:
            runs["n_substructure"] = runs.pop("substructure").apply(len)
        runs["tiers"] = runs["tiers"].apply(" ".join)
        if "blocks" in runs:
            runs["blocks"] = runs["blocks"].apply(lambda b: " ".join(map(str, b)))
        runs = runs.sort_values(["scenario", "seed"])
        runs.to_csv(results / "runs.csv", index=False)
    table = summary(runs)
    table.to_csv(results / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table)


if __name__ == "__main__":
    main()
