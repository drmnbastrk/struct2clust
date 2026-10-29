"""Random halves of clusters, measured with the effect-size ratio.

    python validation/random_halves.py example/results/* --out validation/results/random_halves.csv

The effect-size rule (docs/METHODS.md, section 6) compares the distance between
two groups' medoids with the state scale: the median distance of the members of
the other top-level clusters to their own medoid. This script cuts every
cluster of finished runs into two random halves (``--cuts`` times) and records
the same ratio for the halves. The halves are not different states, so the
ratios show what the rule alone would see between parts of one state; the
real sub-groups that refinement examined are listed for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def medoid(dist: np.ndarray, idx: np.ndarray) -> int:
    return int(idx[np.argmin(dist[np.ix_(idx, idx)].sum(axis=1))])


def halves(run: Path, cuts: int, seed: int) -> list[dict]:
    dist = np.load(run / "distance_matrix.npy")
    names = (run / "distance_matrix_labels.txt").read_text().split("\n")[: len(dist)]
    cl = pd.read_csv(run / "clusters.csv").set_index("structure").loc[names]
    top = cl["top_level"].to_numpy() if "top_level" in cl else cl["cluster"].to_numpy()
    final = cl["cluster"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for c in sorted(set(final) - {0}):
        members = np.flatnonzero(final == c)
        t = top[members[0]]
        others = [np.flatnonzero(top == o) for o in sorted(set(top) - {0, t})]
        if not others or len(members) < 10:
            continue
        scale = float(np.median(np.concatenate([dist[o, medoid(dist, o)] for o in others])))
        for _ in range(cuts):
            perm = rng.permutation(members)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
            rows.append({"run": run.name, "cluster": int(c), "size": len(members),
                         "ratio": dist[medoid(dist, a), medoid(dist, b)] / scale})
    return rows


def real_subgroups(run: Path) -> list[dict]:
    """Ratios of the sub-groups refinement found reproducible (split or merged)."""
    summary = json.loads((run / "run_summary.json").read_text())
    rows = []
    for e in summary["refinement"]["attempts"]:
        for p in e.get("medoid_separation") or []:
            if p.get("ratio_to_state_scale") is not None:
                rows.append({"run": run.name, "clusters": e["clusters"],
                             "ratio": p["ratio_to_state_scale"], "split": e["split"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--cuts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("validation/results/random_halves.csv"))
    args = ap.parse_args()
    runs = [r for r in args.runs if (r / "run_summary.json").exists()]
    table = pd.DataFrame([row for r in runs for row in halves(r, args.cuts, args.seed)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    per = table.groupby(["run", "cluster", "size"])["ratio"].agg(
        median="median", p95=lambda v: v.quantile(0.95), max="max",
        above_1=lambda v: (v >= 1).mean()).round(3)
    print("Random halves (ratio of medoid distance to state scale):")
    print(per.to_string())
    print(f"\nAll {len(table)} cuts: median {table['ratio'].median():.2f}, "
          f"95% quantile {table['ratio'].quantile(0.95):.2f}, "
          f"at least 1 in {100 * (table['ratio'] >= 1).mean():.0f}%")
    real = pd.DataFrame([row for r in runs for row in real_subgroups(r)])
    if not real.empty:
        print("\nReproducible sub-groups examined by refinement:")
        print(real.to_string(index=False))


if __name__ == "__main__":
    main()
