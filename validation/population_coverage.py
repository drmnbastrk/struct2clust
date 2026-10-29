"""Coverage of the population intervals and size of the halves comparison for several
block lengths, on stationary two-state Markov chains with known populations
(docs/METHODS.md, section 9, "How approximate"). Frames are 6-D: the state (6 units
apart along one axis) plus AR(1) noise (phi = 0.9). The true state labels are the
clusters, so only the interval procedure is tested.
Nominal level 0.95 throughout; the level used by the pipeline is chosen by
validation/population_calibration.py.

Block rules (blocks per trajectory, at least 8 and at most 50 as in the pipeline, unless stated):
  pc1_1tau   n / tau(PC1)                         (the blocks of the clustering)
  pc1_2tau   n / (2 tau(PC1))
  pc1_3tau   n / (3 tau(PC1))
  mem_1tau   n / tau(membership), tau of the cluster indicators (Geyer), largest
  mem_2tau   n / (2 tau(membership))
  mem_2tau_min4  as mem_2tau but at least 4 blocks

    python validation/population_coverage.py validation/results/population_coverage.csv 200
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from struct2clust.clustering import trajectory_blocks
from struct2clust.populations import population_uncertainty
from struct2clust.sampling import (
    MAX_BLOCKS,
    MIN_BLOCKS,
    initial_positive_sequence_time,
    integrated_autocorrelation_time,
    principal_coordinate,
)


def chain(rng, n, a, b):
    s = np.empty(n, int)
    s[0] = rng.random() < a / (a + b)
    u = rng.random(n)
    for t in range(1, n):
        s[t] = (u[t] < a) if s[t - 1] == 0 else (u[t] >= b)
    return s


def n_blocks(n, length, lo=MIN_BLOCKS):
    return int(min(n, MAX_BLOCKS, max(lo, np.floor(n / length))))


CONFIGS = [(1000, 0.02, 0.02), (1000, 0.01, 0.01), (1000, 0.005, 0.02), (2000, 0.01, 0.01),
           (1000, 0.05, 0.05), (4000, 0.01, 0.01)]

if __name__ == "__main__":
    out, reps = sys.argv[1], int(sys.argv[2])
    rows = []
    for n, a, b in CONFIGS:
        pi1 = a / (a + b)
        tau_true = (2 - a - b) / (a + b)
        for r in range(reps):
            rng = np.random.default_rng([n, int(1e4 * a), int(1e4 * b), r])
            s = chain(rng, n, a, b)
            if s.min() == s.max():
                continue
            e = np.zeros((n, 6))
            z = rng.normal(size=(n, 6))
            for t in range(1, n):
                e[t] = 0.9 * e[t - 1] + np.sqrt(1 - 0.81) * z[t]
            x = e + np.outer(s, [6, 0, 0, 0, 0, 0])
            tau_pc1 = integrated_autocorrelation_time(principal_coordinate(cdist(x, x)))[0]
            tau_mem = max(initial_positive_sequence_time((s == c).astype(float)) for c in (0, 1))
            rules = {"pc1_1tau": n_blocks(n, tau_pc1), "pc1_2tau": n_blocks(n, 2 * tau_pc1),
                     "pc1_3tau": n_blocks(n, 3 * tau_pc1), "mem_1tau": n_blocks(n, tau_mem),
                     "mem_2tau": n_blocks(n, 2 * tau_mem),
                     "mem_2tau_min4": n_blocks(n, 2 * tau_mem, lo=4)}
            for rule, nb in rules.items():
                blocks = trajectory_blocks(n, nb)
                pop = population_uncertainty(s, blocks, ["A"] * n, seed=r, level=0.95)
                lo, hi = pop["fraction_low"][1], pop["fraction_high"][1]
                rows.append({"n": n, "a": a, "b": b, "pi1": pi1, "tau_true": tau_true,
                             "tau_pc1": tau_pc1, "tau_mem": tau_mem, "rule": rule,
                             "n_blocks": nb, "covered": lo <= pi1 <= hi, "width": hi - lo,
                             "halves_differ": any(c["differs"] for c in pop["comparisons"])})
        df = pd.DataFrame([w for w in rows if (w["n"], w["a"], w["b"]) == (n, a, b)])
        print(n, a, b, df.groupby("rule")[["n_blocks", "covered", "halves_differ", "width"]]
              .mean().round(2).to_dict("index"), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
