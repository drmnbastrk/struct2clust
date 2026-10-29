"""Choosing the clustering resolution: the finest partition whose clusters are all stable.

Every candidate HDBSCAN setting (min_cluster_size, min_samples, extraction
method) yields a partition. Every setting whose partition has at least two
clusters is checked by resampling (Hennig 2007), even when two settings give the
same partition, because they behave differently on subsamples. A setting
qualifies when every one of its clusters has a mean Jaccard of at least
``threshold`` (0.75, Hennig's "stable"). Among qualifying settings the one with
the most clusters is selected.

- Over-clustering is excluded because a cluster that is not reproducible
  (e.g. a density fluctuation split off a larger group) disqualifies the
  partition.
- Under-clustering is excluded because a finer partition whose parts are all
  reproducible always wins over a coarser one.

Ties (same number of clusters) go to the lower noise fraction, i.e. the
setting that assigns the most structures to its stable clusters, then to the
higher mean Jaccard, then to the larger min_cluster_size. If no partition qualifies,
the data show no reproducible cluster structure; the partition with the highest
weakest-cluster Jaccard is returned for inspection and flagged. If no setting
gives two or more clusters, the selection has no clusters (``n_clusters`` 0).

Monte Carlo precision. A cluster's mean Jaccard is estimated from random
subsamples, so a decision made from a fixed, small number of them changes with
the random seed whenever the true value is close to the threshold. Each
candidate is therefore decided by a sequential test by betting (Waudby-Smith &
Ramdas 2024), the bounded-value counterpart of the confidence-sequence method
for Monte Carlo tests (Gandy 2009; Ding, Gandy & Hahn 2020). For every cluster,
two wealth processes bet on the per-subsample Jaccard values (which lie in
[0, 1]): one that the mean Jaccard is above the threshold, one that it is
below. A wealth that reaches 1/alpha establishes its side; by Ville's
inequality this happens to the wrong side with probability at most alpha,
however often the evidence is looked at, and without any normal approximation.

- A candidate qualifies when every cluster is established above the threshold
  (an intersection-union test, Berger 1982: each cluster at the candidate's
  level gives the candidate that level).
- It does not qualify when some cluster is established below the threshold
  (each cluster at the candidate's level divided by its number of clusters).
- The risk given to the selection (its share of the run's resampling risk, see
  pipeline.top_level_risk and hierarchy.node_risk) is shared by its candidates
  (Bonferroni), so with probability at least 1 - risk every decision of the
  selection equals the one infinitely many subsamples would give.

The risk concerns Monte Carlo error only: the probability that another random
seed would change the result on the same data. It is not an error rate of the
clusters with respect to the population the structures were drawn from.

A candidate still undecided after ``max_resamples`` is marked borderline and
does not qualify: its stability could not be established, so the partition it
proposes is not claimed. A mean exactly at the threshold can never be decided
by any sequential procedure (Gandy 2009), so such a limit is unavoidable; it
is the convention of Monte Carlo testing that a null hypothesis is not
rejected unless the evidence is sufficient.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from struct2clust.clustering import candidate_grid, hdbscan_labels, relabel_by_size
from struct2clust.stability import one_resample

logger = logging.getLogger(__name__)


@dataclass
class Selection:
    min_cluster_size: int
    min_samples: int
    method: str
    n_clusters: int
    min_jaccard: float
    mean_jaccard: float
    stable: bool               # False: no candidate had all clusters >= threshold
    candidates: pd.DataFrame   # one row per candidate setting
    borderline: bool = False   # a candidate that could change the result was undecided
    cluster_estimates: list = None   # selected candidate: mean Jaccard per cluster
    tie_partitions_differ: bool = False   # qualifying settings tied on clusters and noise
    #                                       give different partitions (choice by estimate)

    @property
    def params(self) -> tuple:
        return (self.min_cluster_size, self.min_samples, self.method)

    @property
    def found_clusters(self) -> bool:
        """False when no candidate setting gave two or more clusters."""
        return self.n_clusters > 0


BATCH = 10
# truncation level c of the bet (Waudby-Smith & Ramdas 2024: "reasonable defaults" 1/2 or 3/4)
BET_LIMIT = 0.75


class BettingTest:
    """Sequential test of whether the mean of values in [0, 1] lies above or below
    ``threshold`` (testing by betting; Waudby-Smith & Ramdas 2024).

    ``up`` is the wealth of betting that the mean is above the threshold, ``down``
    that it is below. Each starts at 1 and is multiplied by 1 + lambda (x - t)
    (resp. 1 - lambda (x - t)) for every new value x; lambda is chosen from the
    earlier values only (approximate growth-rate adaptive betting, aGRAPA) and
    capped so that the wealth never becomes negative. Under the hypothesis that
    the bet is wrong, each wealth is a nonnegative supermartingale, so it reaches
    1/alpha with probability at most alpha at any time (Ville's inequality).
    """

    def __init__(self, threshold: float):
        self.t = threshold
        self.log_up = 0.0
        self.log_down = 0.0
        self.n = 0
        self.total = 0.0
        self.sq_dev = 0.0

    def _lambda(self) -> float:
        # mean and variance of the earlier values with one pseudo-value of mean 1/2
        # and variance 1/4, as in Waudby-Smith & Ramdas (2024)
        mean = (0.5 + self.total) / (self.n + 1)
        var = (0.25 + self.sq_dev) / (self.n + 1)
        gap = mean - self.t
        return gap / (var + gap ** 2)

    def update(self, x: float) -> None:
        if np.isnan(x):            # cluster absent from this subsample: no bet
            return
        lam = self._lambda()
        up = min(max(lam, 0.0), BET_LIMIT / self.t)
        down = min(max(-lam, 0.0), BET_LIMIT / (1.0 - self.t))
        self.log_up += np.log1p(up * (x - self.t))
        self.log_down += np.log1p(-down * (x - self.t))
        self.n += 1
        self.total += x
        self.sq_dev += (x - (0.5 + self.total) / (self.n + 1)) ** 2

    def above(self, alpha: float) -> bool:
        return self.log_up >= np.log(1.0 / alpha)

    def below(self, alpha: float) -> bool:
        return self.log_down >= np.log(1.0 / alpha)


@dataclass
class Evaluation:
    weakest: float       # smallest per-cluster mean Jaccard
    mean: float          # mean of the per-cluster mean Jaccards
    weakest_se: float    # standard error of the weakest cluster's mean (descriptive)
    n_subsamples: int
    cluster_means: list  # mean Jaccard per cluster (descriptive)
    qualifies: bool      # every cluster established above the threshold at the risk
    borderline: bool     # still undecided after max_resamples (then qualifies is False)


def _mean_se(jac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster mean Jaccard and its standard error (NaN = cluster absent from a
    subsample, ignored)."""
    n = np.sum(~np.isnan(jac), axis=0)
    mean = np.nanmean(jac, axis=0)
    sd = np.nanstd(jac, axis=0, ddof=1) if len(jac) > 1 else np.zeros(jac.shape[1])
    return mean, sd / np.sqrt(np.maximum(n, 1))


def evaluate(dist, labels, blocks, params, fraction, seed, threshold, alpha,
             min_resamples=20, max_resamples=1000, absent_zero=False) -> Evaluation:
    """Sequential decision whether every cluster of one partition has a mean
    Jaccard of at least ``threshold``, at Monte Carlo risk ``alpha``.

    Subsamples are added in batches of 10 (at least ``min_resamples``). The
    candidate qualifies once every cluster is established above the threshold at
    level alpha, and does not qualify once some cluster is established below it
    at level alpha / k (k clusters); a candidate undecided after
    ``max_resamples`` is marked borderline and does not qualify.
    """
    k = int(labels.max()) + 1
    tests = [BettingTest(threshold) for _ in range(k)]
    seeds = np.random.SeedSequence(seed).generate_state(max_resamples)
    rows = []
    decided = None
    for start in range(0, max_resamples, BATCH):
        for s in seeds[start:start + BATCH]:
            jac = one_resample(dist, labels, blocks, params, fraction, int(s), absent_zero)[1]
            rows.append(jac)
            for test, x in zip(tests, jac):
                test.update(float(x))
        if len(rows) < min_resamples:
            continue
        if any(t.below(alpha / k) for t in tests):
            decided = False
            break
        if all(t.above(alpha) for t in tests):
            decided = True
            break
    mean, se = _mean_se(np.array(rows))
    weakest = int(np.argmin(mean))
    return Evaluation(
        weakest=float(mean[weakest]), mean=float(mean.mean()), weakest_se=float(se[weakest]),
        n_subsamples=len(rows), cluster_means=[round(float(v), 4) for v in mean],
        qualifies=bool(decided),
        borderline=decided is None,
    )


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two labelings group the structures identically (same noise, and the
    clusters equal up to their numbering)."""
    if not np.array_equal(a < 0, b < 0):
        return False
    pairs = set(zip(a[a >= 0].tolist(), b[b >= 0].tolist()))
    return len(pairs) == len({x for x, _ in pairs}) == len({y for _, y in pairs})


def select_partition(
    dist: np.ndarray,
    blocks: np.ndarray,
    threshold: float = 0.75,
    fraction: float = 0.8,
    mcs_values: Optional[list[int]] = None,
    ms_values: Optional[list[int]] = None,
    seed: int = 42,
    n_jobs: int = -1,
    quiet: bool = False,
    resampling_risk: float = 0.001,
    min_resamples: int = 20,
    max_resamples: int = 1000,
    absent_zero: bool = False,
) -> Selection:
    log = logger.debug if quiet else logger.info
    grid = candidate_grid(len(dist), mcs_values, ms_values)
    labelings = Parallel(n_jobs=n_jobs)(
        delayed(hdbscan_labels)(dist, mcs, ms, method) for mcs, ms, method in grid
    )
    labelings = [relabel_by_size(lab) for lab in labelings]
    table = pd.DataFrame([
        {"min_cluster_size": mcs, "min_samples": ms, "method": method,
         "n_clusters": int(lab.max() + 1), "noise": float(np.mean(lab < 0))}
        for (mcs, ms, method), lab in zip(grid, labelings)
    ])

    # Settings that give the same partition still behave differently on
    # subsamples, so every setting with >= 2 clusters is assessed on its own.
    todo = [i for i, lab in enumerate(labelings) if lab.max() >= 1]
    log(
        "Resolution selection: %d settings (%d with >= 2 clusters), %d-%d subsamples each "
        "(resampling risk %g)", len(grid), len(todo), min_resamples, max_resamples,
        resampling_risk,
    )
    seeds = dict(zip(todo, np.random.SeedSequence(seed).generate_state(max(1, len(todo)))))
    # ``resampling_risk`` is this selection's budget (docs/METHODS.md, section 6),
    # shared by its candidates (Bonferroni)
    alpha = resampling_risk / max(1, len(todo))
    table["evaluated"] = False
    table["min_jaccard"] = np.nan
    table["min_jaccard_se"] = np.nan
    table["mean_jaccard"] = np.nan
    table["n_subsamples"] = 0
    table["qualifies"] = False
    table["borderline"] = False
    estimates: dict[int, list] = {}
    # Candidates are assessed from the finest to the coarsest number of clusters.
    # Once one with k clusters qualifies, coarser candidates cannot change the
    # result (the finest qualifying partition is selected) and are not assessed;
    # the result is the same as assessing all of them.
    for k in sorted({int(labelings[i].max()) + 1 for i in todo}, reverse=True):
        group = [i for i in todo if labelings[i].max() + 1 == k]
        scores = Parallel(n_jobs=n_jobs)(
            delayed(evaluate)(dist, labelings[i], blocks, grid[i], fraction, int(seeds[i]),
                              threshold, alpha, min_resamples, max_resamples, absent_zero)
            for i in group
        )
        for i, ev in zip(group, scores):
            estimates[i] = ev.cluster_means
            table.loc[i, ["evaluated", "min_jaccard", "min_jaccard_se", "mean_jaccard",
                          "n_subsamples", "qualifies", "borderline"]] = [
                True, ev.weakest, ev.weakest_se, ev.mean, ev.n_subsamples, ev.qualifies,
                ev.borderline]
        if any(ev.qualifies for ev in scores):
            break

    scored = table[table["n_clusters"] >= 2]
    table["selected"] = False
    if scored.empty:
        (logger.debug if quiet else logger.warning)(
            "No candidate setting gives two or more clusters: HDBSCAN finds no cluster "
            "structure in these data.")
        return Selection(0, 0, "", 0, float("nan"), float("nan"), False, table)
    qualifying = scored[scored["qualifies"]]
    stable = not qualifying.empty
    if stable:
        pick = qualifying.sort_values(
            ["n_clusters", "noise", "mean_jaccard", "min_cluster_size"],
            ascending=[False, True, False, False],
        ).iloc[0]
    else:
        pick = scored[scored["evaluated"]].sort_values(
            ["min_jaccard", "mean_jaccard"], ascending=[False, False]
        ).iloc[0]
        (logger.debug if quiet else logger.warning)(
            "No partition has all clusters reproducible (Jaccard >= %.2f); best weakest-cluster "
            "Jaccard is %.2f. The data show no reproducible cluster structure.",
            threshold, pick["min_jaccard"],
        )
    table["selected"] = table.index == pick.name
    # Settings tied with the selected one on clusters and noise are ordered by their
    # estimated mean Jaccard, which is not covered by the Monte Carlo guarantee
    # (docs/METHODS.md, section 6); that matters only if their partitions differ.
    tie_differs = False
    if stable:
        tied = qualifying[(qualifying["n_clusters"] == pick["n_clusters"])
                          & (qualifying["noise"] == pick["noise"])]
        tie_differs = any(not same_partition(labelings[i], labelings[int(pick.name)])
                          for i in tied.index)
    # A borderline (undecided, hence not qualifying) candidate matters only if it
    # could have changed the result: it has more clusters than the selected one
    # (or, when nothing qualifies, it could have qualified).
    finer = scored[scored["n_clusters"] > pick["n_clusters"]] if stable else scored
    borderline = bool(pick["borderline"] or finer["borderline"].any())
    sel = Selection(
        int(pick["min_cluster_size"]), int(pick["min_samples"]), str(pick["method"]),
        int(pick["n_clusters"]), float(pick["min_jaccard"]), float(pick["mean_jaccard"]),
        stable, table, borderline, estimates.get(int(pick.name)), tie_differs,
    )
    log(
        "Selected %d clusters (min_cluster_size=%d, min_samples=%d, %s); "
        "weakest cluster Jaccard %.3f +- %.3f",
        sel.n_clusters, sel.min_cluster_size, sel.min_samples, sel.method, sel.min_jaccard,
        float(pick["min_jaccard_se"]),
    )
    if tie_differs:
        (logger.debug if quiet else logger.warning)(
            "Several qualifying settings have the selected number of clusters and noise "
            "fraction but different partitions; the one with the highest estimated mean "
            "Jaccard was selected, a choice that can change with the random seed.")
    if borderline:
        (logger.debug if quiet else logger.warning)(
            "Resolution at the stability threshold: a finer candidate's weakest-cluster Jaccard "
            "is within Monte Carlo error of %.2f after %d subsamples; its stability was not "
            "established, so it was not selected.", threshold, max_resamples)
    return sel
