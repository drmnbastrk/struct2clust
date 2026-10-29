"""Hierarchical refinement: the selection rule applied again inside every cluster.

With excess-of-mass extraction, a few dominant, well-separated groups can hide
the states inside them: every candidate setting returns the dominant groups,
because splitting one of them never has more "excess of mass" than keeping it.
Refinement therefore treats each selected cluster as a data set of its own:

1. Group structure. Its members are tested for clusterability. Only a cluster
   whose members are CLUSTERABLE (both tests agree) can be split, because a
   structureless but curved, elongated or drifting cluster can be cut
   reproducibly (Ben-David et al. 2006); such clusters are WEAK. The dip test of
   a cluster with n_j of the N structures uses the significance level
   alpha * (n_j - 1) / (N - 1), the level that controls the family-wise error rate
   of a sequence of tests (Meinshausen 2008; Kimes et al. 2017). The control is
   nominal here: dip-test p-values on pairwise distances are approximate, and the
   clusters tested were selected from the same data (docs/METHODS.md, section 8).
2. Reproducibility. The resolution selection is run on its members only, with
   the same rule and Monte Carlo precision as at the top level: a
   sub-partition qualifies if every sub-cluster's mean Jaccard is established
   to be >= stability.high at the resampling risk. A sub-partition that cannot
   be resolved within max_resamples (borderline) is not established.
3. Effect size (optional; off by default, when only the ratio of every pair of
   final clusters is reported). Steps 1-2 establish that sub-groups exist; they
   do not say whether they are as distinct as the states of the ensemble. Tight groups
   can be reproducible and separated although they differ far less than the
   members of a single top-level state differ from each other (e.g. two
   modelling protocols applied to one template). Following the
   separation-versus-scatter logic of cluster validity indices (Davies &
   Bouldin 1979), the separation of two sub-clusters is the distance between
   their medoids, and the reference is the scatter of the ensemble's states:
   the median distance of the members of all *other* top-level clusters to
   their own medoid, pooled (the "state scale"). Sub-clusters whose medoids are
   closer than ``selection.refine_min_separation_ratio`` times the state
   scale, or closer than the optional absolute floor
   ``selection.refine_min_separation``, are merged (single linkage). If one
   group remains, the cluster is not split and its reproducible sub-groups are
   reported as sub-structure (output column ``substructure``); otherwise it is
   split into the merged groups. Medoid distance is used rather than the
   median distance between members, because the latter contains the members'
   own scatter: by that measure, random halves of a wide structureless cluster
   are as far apart as real sub-groups.
4. Repeat for every sub-cluster until nothing splits (the state scale of the
   top-level parent is used at every depth).

The stability of a sub-cluster is measured within its parent: the parent's
members are subsampled and re-clustered with the sub-cluster's settings. Its
tier is capped by both the data-set and the parent-level clusterability
verdicts, and its membership probability is the parent's membership
probability times the within-parent one. A group formed by merging
sub-clusters takes the statistics of its least stable part and the summed
membership probability of its parts. Members that are noise at the
sub-cluster level become noise but keep their top-level group.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from struct2clust.clusterability import assess
from struct2clust.clustering import hdbscan_labels, medoids, relabel_by_size, subsample
from struct2clust.selection import select_partition
from struct2clust.stability import Stability, cluster_stability

logger = logging.getLogger(__name__)

TIER_ORDER = ["LOW", "MODERATE", "HIGH"]
TIER_CAP = {"CLUSTERABLE": "HIGH", "WEAK": "MODERATE", "NOT_CLUSTERABLE": "LOW"}


def cap_tiers(tiers: list[str], verdict: Optional[str]) -> list[str]:
    """Limit cluster tiers by a clusterability verdict.

    Resampling stability is conditional on there being structure to find: in
    structureless data HDBSCAN can still return small, reproducible groups.
    Data judged NOT_CLUSTERABLE therefore get no cluster above LOW, and WEAK
    data none above MODERATE.
    """
    if verdict is None:
        return tiers
    cap = TIER_ORDER.index(TIER_CAP[verdict])
    return [TIER_ORDER[min(TIER_ORDER.index(t), cap)] for t in tiers]


def stricter(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """The verdict with the lower tier cap."""
    if a is None or b is None:
        return a or b
    return min(a, b, key=lambda v: TIER_ORDER.index(TIER_CAP[v]))


@dataclass
class Leaf:
    members: np.ndarray      # global indices
    prob: np.ndarray         # membership probability, length n
    row: dict                # stability statistics
    path: str                # "2" top-level, "2.1" first sub-cluster of cluster 2


@dataclass
class Refinement:
    labels: np.ndarray
    stability: Stability
    groups: np.ndarray       # top-level cluster per structure (-1 = top-level noise)
    paths: np.ndarray        # deepest cluster containing each structure ("" = top-level noise)
    log: list[dict] = field(default_factory=list)   # one entry per refinement attempt
    substructure: Optional[np.ndarray] = None   # reproducible sub-group below the state scale


def node_risk(risk: float, level: int, n_node: int, n_total: int) -> float:
    """Monte Carlo risk budget of the refinement selection at one node.

    The run's risk R is split as R/2 for the top-level selection and
    (R/2) * 2^-(level+1) * n_node / n_total for a node at ``level`` (0 = a top-level
    cluster) holding n_node of the n_total structures. Nodes of one level are
    disjoint, so their n_node add up to at most n_total, and the budgets of all
    nodes add up to at most R/2: the probability that any decision of the run
    differs from the one infinitely many subsamples would give is at most R
    (docs/METHODS.md, section 6).
    """
    return risk / 2 * 2.0 ** -(level + 1) * n_node / n_total


def _seed(seed: int, path: str) -> int:
    parts = [int(p) for p in path.split(".")]
    return int(np.random.SeedSequence([seed, *parts]).generate_state(1)[0])


def state_scales(dist: np.ndarray, labels: np.ndarray) -> dict[int, Optional[float]]:
    """State scale per top-level cluster c: the median distance of the members of all
    other top-level clusters to their own medoid (pooled). None if c is the only one."""
    meds = medoids(dist, labels)
    to_medoid = {c: dist[np.flatnonzero(labels == c), m] for c, m in meds.items()}
    out = {}
    for c in meds:
        pooled = [v for t, v in to_medoid.items() if t != c]
        out[c] = float(np.median(np.concatenate(pooled))) if pooled else None
    return out


def sub_label(path: str, j: int) -> str:
    """Internal label of the j-th (0-based) sub-group of a hierarchy node: 2a, 2b, ...
    The output numbers the sub-groups of every final cluster 1, 2, ..."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return f"{path}{letters[j]}" if j < 26 else f"{path}_{j + 1}"


RANGE_SUBSAMPLES = 200
# With input.groups or automatic trajectory blocks, a node whose members come from fewer
# units is not split: half of 8 units gives 70 different subsamples, half of 4 only 6.
# A pre-specified quality-control rule, not a statistical threshold.
MIN_GROUPS = 8


def separation_range(dist: np.ndarray, a: np.ndarray, b: np.ndarray,
                     reference: list[np.ndarray], blocks: np.ndarray, fraction: float,
                     seed: int, n: int = RANGE_SUBSAMPLES) -> Optional[list[float]]:
    """How much the ratio of medoid distance to state scale varies between subsamples.

    In each of ``n`` subsamples of the structures (whole resampling blocks, as for
    the stability), the medoids of groups ``a`` and ``b`` and of every reference
    group are found again; the ratio is their medoid distance over the median
    distance of the reference members to their own medoid. Returns the 2.5% and
    97.5% quantiles. This is the spread between subsamples of the same data, not a
    confidence interval: subsamples of 80% of the data vary less than new data
    would. None without reference groups.
    """
    if not reference:
        return None
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(dist), dtype=bool)
    ratios = []
    for _ in range(n):
        keep[:] = False
        keep[subsample(rng, blocks, fraction)] = True
        parts = [g[keep[g]] for g in (a, b, *reference)]
        if any(len(p) == 0 for p in parts):
            continue
        meds = [p[np.argmin(dist[np.ix_(p, p)].sum(axis=1))] for p in parts]
        scale = float(np.median(np.concatenate(
            [dist[p, m] for p, m in zip(parts[2:], meds[2:])])))
        if scale > 0:
            ratios.append(dist[meds[0], meds[1]] / scale)
    if not ratios:
        return None
    return [round(float(q), 3) for q in np.quantile(ratios, [0.025, 0.975])]


def separation_pairs(dist: np.ndarray, labels: np.ndarray, blocks: np.ndarray,
                     fraction: float, seed: int,
                     n: int = RANGE_SUBSAMPLES) -> list[dict]:
    """Effect size of every pair of clusters, reported whether or not a rule uses it.

    For clusters a and b: the distance between their medoids, each cluster's own
    scatter (median distance of its members to its medoid), the state scale (the
    median distance of the members of all *other* clusters to their own medoid,
    pooled; None with only two clusters), their ratio, and the 2.5% and 97.5%
    quantiles of the ratio over ``n`` subsamples (whole resampling blocks; medoids
    found again in each). The quantiles are the spread between subsamples of the
    same data, not a confidence interval.
    """
    k = int(labels.max()) + 1
    if k < 2:
        return []
    members = [np.flatnonzero(labels == c) for c in range(k)]

    def describe(parts):
        meds = [p[np.argmin(dist[np.ix_(p, p)].sum(axis=1))] for p in parts]
        return meds, [dist[p, m] for p, m in zip(parts, meds)]

    def scale(to_med, a, b):
        others = [v for c, v in enumerate(to_med) if c not in (a, b)]
        return float(np.median(np.concatenate(others))) if others else None

    meds, to_med = describe(members)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(dist), dtype=bool)
    ratios = {(a, b): [] for a in range(k) for b in range(a + 1, k)}
    if k >= 3:
        for _ in range(n):
            keep[:] = False
            keep[subsample(rng, blocks, fraction)] = True
            parts = [m[keep[m]] for m in members]
            if any(len(p) == 0 for p in parts):
                continue
            sm, st = describe(parts)
            for a, b in ratios:
                s = scale(st, a, b)
                if s:
                    ratios[a, b].append(dist[sm[a], sm[b]] / s)
    out = []
    for a, b in ratios:
        d, s = float(dist[meds[a], meds[b]]), scale(to_med, a, b)
        r = ratios[a, b]
        out.append({
            "a": a + 1, "b": b + 1, "medoid_distance": round(d, 4),
            "scatter_a": round(float(np.median(to_med[a])), 4),
            "scatter_b": round(float(np.median(to_med[b])), 4),
            "state_scale": round(s, 4) if s else None,
            "ratio_to_state_scale": round(d / s, 3) if s else None,
            "ratio_subsampling_range": [round(float(q), 3) for q in
                                        np.quantile(r, [0.025, 0.975])] if r else None,
        })
    return out


def merge_close(md: np.ndarray, threshold: float) -> np.ndarray:
    """Group index per sub-cluster: single-linkage components of medoid distance < threshold."""
    k = len(md)
    group = np.arange(k)
    for i in range(k):
        for j in range(i + 1, k):
            if md[i, j] < threshold and group[i] != group[j]:
                group[group == group[j]] = group[i]
    return np.unique(group, return_inverse=True)[1]


def merge_states(dist: np.ndarray, labels: np.ndarray, stability: Stability, cfg,
                 blocks: Optional[np.ndarray] = None):
    """The effect-size rule at the top level: merge clusters whose medoids are closer
    than the state scale of the clusters not involved in the comparison.

    For a pair (a, b) the reference is the median distance of the members of all
    other clusters to their own medoid (pooled); with only two clusters there is no
    reference and nothing is merged (the absolute floor still applies). For every
    merged pair the ratio's spread between subsamples is recorded
    (``separation_range``). Returns (labels, stability, log, substructure or None).
    """
    s = cfg.selection
    k = int(labels.max()) + 1
    if k < 2 or not (s.refine_min_separation_ratio or s.refine_min_separation):
        return labels, stability, [], None
    meds = medoids(dist, labels)
    to_medoid = {c: dist[np.flatnonzero(labels == c), meds[c]] for c in range(k)}
    md = np.array([[dist[meds[a], meds[b]] for b in range(k)] for a in range(k)])
    thresholds = np.zeros((k, k))
    pairs = []
    for a in range(k):
        for b in range(a + 1, k):
            others = [to_medoid[c] for c in range(k) if c not in (a, b)]
            scale = float(np.median(np.concatenate(others))) if others else None
            t = max(s.refine_min_separation_ratio * scale if scale else 0.0,
                    s.refine_min_separation or 0.0)
            thresholds[a, b] = thresholds[b, a] = t
            if md[a, b] < t:
                members = [np.flatnonzero(labels == c) for c in range(k)]
                spread = separation_range(
                    dist, members[a], members[b],
                    [members[c] for c in range(k) if c not in (a, b)],
                    blocks if blocks is not None else np.arange(len(dist)),
                    cfg.stability_fraction,
                    # own stream, distinct from the refinement node "a.b"
                    _seed(_seed(cfg.seed, "0"), f"{a + 1}.{b + 1}"))
                pairs.append({"a": str(a + 1), "b": str(b + 1),
                              "distance": round(float(md[a, b]), 4),
                              "state_scale": round(scale, 4) if scale else None,
                              "ratio_to_state_scale": round(float(md[a, b] / scale), 3)
                              if scale else None,
                              "ratio_subsampling_range": spread})
    if not pairs:
        return labels, stability, [], None
    # single linkage over the pairs below their threshold
    group = np.arange(k)
    for p in pairs:
        a, b = int(p["a"]) - 1, int(p["b"]) - 1
        group[group == group[b]] = group[a]
    group = np.unique(group, return_inverse=True)[1]
    new_labels, tab, membership = _merge_groups(
        stability.clusters, stability.membership, labels, group)
    substructure = np.array([""] * len(labels), dtype=object)
    log = []
    for g in range(int(new_labels.max()) + 1):
        merged = sorted({int(c) for c in labels[new_labels == g]})
        if len(merged) > 1:
            for j, c in enumerate(merged):
                substructure[labels == c] = sub_label(str(g + 1), j)
            log.append({"merged_top_level_clusters": [str(c + 1) for c in merged],
                        "into": str(g + 1),
                        "pairs": [p for p in pairs if int(p["a"]) - 1 in merged]})
    logger.info("Top level: %d clusters closer than the state scale merged -> %d clusters",
                k, int(new_labels.max()) + 1)
    return (new_labels, Stability(tab, membership, stability.n_resamples), log, substructure)


def refine(
    dist: np.ndarray,
    labels: np.ndarray,
    stability: Stability,
    blocks: np.ndarray,
    verdict: Optional[str],
    cfg,
) -> Refinement:
    """Split every cluster that contains distinct, reproducible sub-states, recursively.

    ``labels`` and ``stability`` are the top-level result (tiers already
    capped); ``cfg`` is the run Config.
    """
    n = len(labels)
    log: list[dict] = []
    leaves: list[Leaf] = []
    table = stability.clusters
    paths = np.array([str(c + 1) if c >= 0 else "" for c in labels], dtype=object)
    substructure = np.array([""] * n, dtype=object)
    scales = state_scales(dist, labels)
    top = [np.flatnonzero(labels == c) for c in range(int(labels.max()) + 1)]
    for c in range(len(top)):
        leaf = Leaf(top[c], stability.membership[:, c].copy(), table.iloc[c].to_dict(),
                    str(c + 1))
        reference = [m for t, m in enumerate(top) if t != c]
        leaves += _split(dist, blocks, leaf, verdict, cfg, log, paths, scales[c], substructure,
                         reference)

    new_labels = np.full(n, -1)
    rows, probs = [], []
    for k, leaf in enumerate(leaves):
        new_labels[leaf.members] = k
        rows.append({**leaf.row, "cluster": k + 1, "size": len(leaf.members), "path": leaf.path,
                     "level": leaf.path.count(".")})
        probs.append(leaf.prob)
    clusters = pd.DataFrame(rows)
    front = ["cluster", "path", "level", "size"]
    clusters = clusters[front + [c for c in clusters.columns if c not in front]]
    groups = labels.copy()
    n_split = sum(e["split"] for e in log)
    if n_split:
        logger.info("Refinement: %d top-level clusters -> %d clusters (%d splits)",
                    int(labels.max()) + 1, len(leaves), n_split)
    else:
        logger.info("Refinement: no cluster split")
    merged = Stability(clusters, np.column_stack(probs), stability.n_resamples)
    return Refinement(new_labels, merged, groups, paths, log,
                      substructure if (substructure != "").any() else None)


def _merge_groups(tab: pd.DataFrame, membership: np.ndarray, child: np.ndarray,
                  group: np.ndarray):
    """Child labels, stability rows and membership after merging sub-clusters into
    groups (largest group first): a merged group takes the row of its least stable
    part and the summed membership of its parts."""
    sizes = [int(np.isin(child, np.flatnonzero(group == g)).sum()) for g in range(group.max() + 1)]
    order = np.argsort([-s for s in sizes], kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    new_child = np.where(child >= 0, rank[group[np.maximum(child, 0)]], -1)
    rows, cols = [], []
    for g in order:
        parts = np.flatnonzero(group == g)
        row = tab.iloc[parts[np.argmin(tab["jaccard_mean"].to_numpy()[parts])]].to_dict()
        row["size"] = sizes[g]
        rows.append(row)
        cols.append(np.minimum(membership[:, parts].sum(axis=1), 1.0))
    new_tab = pd.DataFrame(rows)
    new_tab["cluster"] = np.arange(1, len(rows) + 1)
    return new_child, new_tab, np.column_stack(cols)


def _split(dist, blocks, leaf: Leaf, verdict, cfg, log, paths, scale=None,
           substructure=None, reference=None) -> list[Leaf]:
    idx = leaf.members
    entry = {"cluster": leaf.path, "size": len(idx), "clusterability": None, "split": False,
             "n_subclusters": 1}
    log.append(entry)
    sub = dist[np.ix_(idx, idx)]
    if cfg.clusterability.enabled:
        c = cfg.clusterability
        # the FWER level along the hierarchy (Meinshausen 2008; Kimes et al. 2017), nominal
        # here: a node with n_j of the N structures is tested at alpha * (n_j - 1) / (N - 1).
        alpha = c.alpha * (len(idx) - 1) / (len(dist) - 1)
        entry["alpha"] = round(alpha, 5)
        local = assess(sub, alpha, c.n_neighbors, c.max_k, quiet=True).verdict
        entry["clusterability"] = local
        if local != "CLUSTERABLE":
            entry["reason"] = ("no group structure among its members" if local == "NOT_CLUSTERABLE"
                               else "group structure not confirmed (clusterability WEAK)")
            return [leaf]
    else:
        local = None

    s, st = cfg.selection, cfg.stability
    if cfg.by_unit:
        n_groups = len(np.unique(blocks[idx]))
        unit = "groups" if cfg.grouped else "trajectory blocks"
        entry["n_groups"] = n_groups
        if n_groups < MIN_GROUPS:
            entry["reason"] = (f"members from {n_groups} {unit} only: too few independent "
                               f"{unit} to resample (at least {MIN_GROUPS})")
            return [leaf]
    seed = _seed(cfg.seed, leaf.path)
    risk = node_risk(s.resampling_risk, leaf.path.count("."), len(idx), len(dist))
    entry["resampling_risk"] = float(f"{risk:.3g}")
    try:
        sel = select_partition(sub, blocks[idx], st.high, cfg.selection_fraction,
                               s.min_cluster_size, s.min_samples, seed=seed, n_jobs=cfg.n_jobs,
                               quiet=True, resampling_risk=risk,
                               min_resamples=s.min_resamples, max_resamples=s.max_resamples,
                               absent_zero=cfg.by_unit)
    except ValueError:   # too small for any candidate setting
        sel = None
    if sel is None or not sel.found_clusters:
        entry["reason"] = "no candidate setting gives two or more sub-clusters"
        return [leaf]
    pick = sel.candidates[sel.candidates["selected"]].iloc[0]
    entry["weakest_jaccard"] = round(sel.min_jaccard, 4)
    entry["weakest_jaccard_se"] = round(float(pick["min_jaccard_se"]), 4)
    entry["n_subsamples"] = int(pick["n_subsamples"])
    entry["borderline"] = sel.borderline
    entry["tie_partitions_differ"] = sel.tie_partitions_differ
    if not sel.stable:
        entry["reason"] = (
            "sub-partition at the stability threshold: reproducibility not established "
            f"after {s.max_resamples} subsamples (borderline)" if sel.borderline
            else "no sub-partition with all sub-clusters reproducible")
        return [leaf]

    child = relabel_by_size(hdbscan_labels(sub, *sel.params))
    stab = cluster_stability(sub, child, blocks[idx], sel.params, st.n_resamples,
                             cfg.stability_fraction, st.high, st.moderate, seed=seed,
                             n_jobs=cfg.n_jobs, quiet=True, absent_zero=cfg.by_unit)
    tab, membership = stab.clusters, stab.membership
    tab["selection_decision"] = "established"
    tab["selection_estimate"] = sel.cluster_estimates
    entry["subcluster_jaccard"] = [round(v, 4) for v in tab["jaccard_mean"]]
    entry["subcluster_sizes"] = [int(v) for v in tab["size"]]

    # Step 3: effect size -- are the sub-clusters as distinct as the ensemble's states?
    k = len(tab)
    meds = medoids(sub, child)
    md = np.array([[sub[meds[a], meds[b]] for b in range(k)] for a in range(k)])
    ratio, floor = s.refine_min_separation_ratio, s.refine_min_separation
    threshold = max(ratio * scale if (scale and ratio) else 0.0, floor or 0.0)
    entry["state_scale"] = round(scale, 4) if scale else None
    entry["separation_threshold"] = round(threshold, 4)
    parts = [idx[child == j] for j in range(k)]
    entry["medoid_separation"] = [
        {"a": a + 1, "b": b + 1,
         "distance": round(float(md[a, b]), 4),
         "ratio_to_state_scale": round(float(md[a, b] / scale), 3) if scale else None,
         "ratio_subsampling_range": separation_range(
             dist, parts[a], parts[b], reference or [], blocks, cfg.stability_fraction,
             _seed(seed, f"{a + 1}.{b + 1}")) if scale else None}
        for a in range(k) for b in range(a + 1, k)]
    group = merge_close(md, threshold) if threshold > 0 else np.arange(k)
    if group.max() == 0:
        if substructure is not None:
            for j in range(k):
                substructure[idx[child == j]] = sub_label(leaf.path, j)
        entry["substructure"] = True
        entry["reason"] = ("reproducible sub-groups closer than the state scale "
                           f"(medoid distance < {threshold:.3g}): one state with sub-structure")
        logger.info("Refinement: cluster %s has %d reproducible sub-groups below the state scale "
                    "(medoids %s apart, threshold %.3g); kept as one cluster", leaf.path, k,
                    ", ".join(f"{e['distance']:.3g}" for e in entry["medoid_separation"]),
                    threshold)
        return [leaf]
    if group.max() < k - 1:
        parts_of = child.copy()
        child, tab, membership = _merge_groups(tab, membership, child, group)
        entry["merged_subclusters"] = []
        for g in range(len(tab)):
            parts = sorted({int(j) for j in parts_of[child == g] if j >= 0})
            if len(parts) > 1:
                into = f"{leaf.path}.{g + 1}"
                entry["merged_subclusters"].append({"into": into, "parts": [j + 1 for j in parts]})
                if substructure is not None:
                    for r, j in enumerate(parts):
                        substructure[idx[parts_of == j]] = sub_label(into, r)

    cap = stricter(verdict, local)
    tab["jaccard_tier"] = tab["tier"]
    tab["tier"] = cap_tiers(tab["jaccard_tier"].tolist(), cap)
    entry.update(split=True, n_subclusters=len(tab), min_cluster_size=sel.min_cluster_size,
                 min_samples=sel.min_samples)
    logger.info(
        "Refinement: cluster %s (n=%d) splits into %d: %s", leaf.path, len(idx), len(tab),
        ", ".join(f"{leaf.path}.{r.cluster}={r.jaccard_mean:.2f}" for r in tab.itertuples()),
    )
    out = []
    for j in range(len(tab)):
        prob = np.zeros_like(leaf.prob)
        prob[idx] = leaf.prob[idx] * membership[:, j]
        child_leaf = Leaf(idx[child == j], prob, tab.iloc[j].to_dict(), f"{leaf.path}.{j + 1}")
        paths[child_leaf.members] = child_leaf.path
        out += _split(dist, blocks, child_leaf, cap, cfg, log, paths, scale, substructure,
                      reference)
    return out
