"""Calibration of the level of the population intervals (docs/METHODS.md, section 9,
"How approximate").

The intervals are block-bootstrap quantiles over blocks of three times the longer of
the geometric and the membership autocorrelation time (sampling.population_blocks).
With few transitions, nominal 95% intervals still cover the true population less often
than 95%. This study finds the nominal level whose actual coverage is 95%: on
stationary two-state Markov chains with known populations
(the state, 6 units apart along one axis, plus AR(1) noise with phi 0.9; the true state
labels are the clusters, so only the interval procedure is tested), it records for
every nominal level whether the interval of state 1 covers the true population, the
interval width, and whether the comparison of the two halves flags a difference
(nominal family-wise 1 - level; the halves of a stationary chain do not differ).
Column ``rule``: "membership_3tau" is the pipeline's rule; "pc1_3tau" (three times the
geometric time only) is kept for comparison.

    python validation/population_calibration.py validation/results/population_calibration.csv 200
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

from struct2clust.clustering import trajectory_blocks  # noqa: E402
from struct2clust.populations import population_uncertainty  # noqa: E402
from struct2clust.sampling import (  # noqa: E402
    POPULATION_BLOCK_TAUS,
    automatic_blocks,
    integrated_autocorrelation_time,
    membership_autocorrelation,
    population_blocks,
    principal_coordinate,
)

LEVELS = [0.95, 0.975, 0.99, 0.995, 0.999]
# (frames, probability of leaving state 0, of leaving state 1)
CONFIGS = [(1000, 0.02, 0.02), (1000, 0.01, 0.01), (1000, 0.005, 0.02), (2000, 0.01, 0.01),
           (1000, 0.05, 0.05), (4000, 0.01, 0.01), (2000, 0.005, 0.005), (1000, 0.1, 0.1)]


def chain(rng, n, a, b):
    s = np.empty(n, int)
    s[0] = rng.random() < a / (a + b)
    u = rng.random(n)
    for t in range(1, n):
        s[t] = (u[t] < a) if s[t - 1] == 0 else (u[t] >= b)
    return s


if __name__ == "__main__":
    out, reps = sys.argv[1], int(sys.argv[2])
    rows = []
    for n, a, b in CONFIGS:
        pi1 = a / (a + b)
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
            tau = integrated_autocorrelation_time(principal_coordinate(cdist(x, x)))[0]
            sampling = [{"name": "A", "n_frames": n, "autocorrelation_time": tau,
                         "n_blocks": 0, "blocks_rule": "automatic"}]
            membership = membership_autocorrelation(s, ["A"] * n)
            tau_mem = max(membership["A"].values())
            rules = {"membership_3tau": population_blocks(sampling, membership)["A"],
                     "pc1_3tau": automatic_blocks(n, POPULATION_BLOCK_TAUS * tau)}
            for rule, nb in rules.items():
                blocks = trajectory_blocks(n, nb)
                for level in LEVELS:
                    pop = population_uncertainty(s, blocks, ["A"] * n, seed=r, level=level,
                                                 flag_level=level)
                    lo, hi = pop["fraction_low"][1], pop["fraction_high"][1]
                    rows.append({"n": n, "a": a, "b": b, "pi1": pi1, "tau_pc1": tau,
                                 "tau_membership": tau_mem, "rule": rule, "n_blocks": nb,
                                 "level": level, "covered": lo <= pi1 <= hi,
                                 "width": hi - lo,
                                 "halves_differ": any(c["differs"]
                                                      for c in pop["comparisons"])})
        df = pd.DataFrame([w for w in rows if (w["n"], w["a"], w["b"]) == (n, a, b)])
        print(n, a, b, df.groupby(["rule", "level"])[["covered", "halves_differ", "width"]]
              .mean().round(3).to_dict("index"), flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
