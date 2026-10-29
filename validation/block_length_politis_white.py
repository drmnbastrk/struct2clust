"""Automatic block length (Politis & White 2004; Patton, Politis & White 2009) for the
trajectories of a finished run, compared with the resampling block length used.

    python validation/block_length_politis_white.py example/results/1RKJ_rna
    python validation/block_length_politis_white.py --self-check

For every trajectory, the coordinate from which the autocorrelation time is estimated
(sampling.principal_coordinate of its distance matrix) is given to the Politis-White
rule for the circular block bootstrap of the mean:

    b_opt = (2 G^2 / D)^(1/3) n^(1/3),   D = (4/3) g(0)^2,
    G = sum_k lambda(k/M) |k| R(k),      g(0) = sum_k lambda(k/M) R(k),

with R the sample autocovariance and lambda the flat-top window (1 up to 1/2, linear
to 0 at 1). M = 2 m, where m is the smallest lag after which K_N = max(5, sqrt(log10 n))
consecutive autocorrelations lie within 2 sqrt(log10 n / n); M is at most n / 2.
The rule targets the variance of a mean, not the clustering itself: it is a reference
for the block length, not a replacement of input.trajectory_blocks.

--self-check compares the estimate with the exact value for AR(1) series
(b_opt = (1.5 (2 phi / (1 - phi^2))^2)^(1/3) n^(1/3)).

Writes validation/results/block_length_politis_white_<run>.csv or
validation/results/block_length_politis_white_selfcheck.csv.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent


def flat_top(t: np.ndarray) -> np.ndarray:
    t = np.abs(t)
    return np.where(t <= 0.5, 1.0, np.where(t <= 1.0, 2.0 * (1.0 - t), 0.0))


def politis_white(x: np.ndarray) -> dict:
    """Optimal block length for the circular block bootstrap of the mean of x."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    if acov[0] <= 0:
        return {"n": n, "m": 0, "M": 0, "b_opt": 1.0}
    rho = acov / acov[0]
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    bound = 2.0 * math.sqrt(math.log10(n) / n)
    m = None
    for lag in range(1, n - kn):
        if np.all(np.abs(rho[lag:lag + kn]) < bound):
            m = lag - 1
            break
    if m is None:
        m = n // 4
    big_m = int(min(max(2 * m, 1), n // 2))
    k = np.arange(-big_m, big_m + 1)
    lam = flat_top(k / big_m)
    r = acov[np.abs(k)]
    g0 = float(np.sum(lam * r))
    g = float(np.sum(lam * np.abs(k) * r))
    d = (4.0 / 3.0) * g0 ** 2
    b = (2.0 * g ** 2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0) if d > 0 else 1.0
    return {"n": n, "m": int(m), "M": big_m, "b_opt": float(min(max(b, 1.0), n))}


def self_check(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for phi in (0.5, 0.8, 0.9, 0.95):
        for n in (500, 5000, 50000):
            est = []
            for _ in range(20):
                x = lfilter([1.0], [1.0, -phi], rng.standard_normal(n + 1000))
                est.append(politis_white(x[1000:])["b_opt"])
            exact = (1.5 * (2 * phi / (1 - phi ** 2)) ** 2) ** (1 / 3) * n ** (1 / 3)
            rows.append({"phi": phi, "n": n, "exact": round(exact, 2),
                         "estimate_median": round(float(np.median(est)), 2),
                         "estimate_q25": round(float(np.quantile(est, 0.25)), 2),
                         "estimate_q75": round(float(np.quantile(est, 0.75)), 2),
                         "median_ratio": round(float(np.median(est)) / exact, 3)})
    return pd.DataFrame(rows)


def run(run_dir: Path) -> pd.DataFrame:
    from struct2clust.config import Config
    from struct2clust.sampling import integrated_autocorrelation_time, principal_coordinate

    dist = np.load(run_dir / "distance_matrix.npy")
    labels = (run_dir / "distance_matrix_labels.txt").read_text().split()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    n_blocks = summary.get("config", {}).get("input", {}).get(
        "trajectory_blocks", Config().input.trajectory_blocks)
    used_length = {t["name"]: t["block_length"] for t in summary["input"].get("replicas", [])
                   if "block_length" in t}
    traj = np.array([lab.split(":")[0] for lab in labels])
    rows = []
    for name in dict.fromkeys(traj):
        idx = np.flatnonzero(traj == name)
        x = principal_coordinate(dist[np.ix_(idx, idx)].copy())
        tau, rel = integrated_autocorrelation_time(x)
        pw = politis_white(x)
        used = used_length.get(name) or len(idx) / min(n_blocks or 50, len(idx))
        rows.append({"trajectory": name, "frames": len(idx),
                     "autocorrelation_time": round(tau, 1),
                     "autocorrelation_time_rel_error": round(rel, 2),
                     "block_used": round(used, 1),
                     "politis_white_block": round(pw["b_opt"], 1),
                     "politis_white_M": pw["M"],
                     "used_over_politis_white": round(used / pw["b_opt"], 2)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    out = HERE / "results"
    if args.self_check:
        t = self_check()
        t.to_csv(out / "block_length_politis_white_selfcheck.csv", index=False)
    else:
        t = run(args.run_dir)
        t.to_csv(out / f"block_length_politis_white_{args.run_dir.name}.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
