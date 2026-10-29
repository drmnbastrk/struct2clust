"""End-to-end run.

load -> distance matrix -> de-duplicate -> clusterability -> resolution selection
(finest partition with all clusters stable) -> HDBSCAN -> cluster-wise stability
(tiers, membership) -> effect size (clusters closer than the state scale are one
state) -> hierarchical refinement (the same rules inside every cluster) -> medoids
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from struct2clust import metrics
from struct2clust.biology import Biology, characterise, medoid_differences
from struct2clust.clusterability import Clusterability, assess
from struct2clust.clustering import (
    hdbscan_labels,
    medoids,
    relabel_by_size,
    trajectory_blocks,
)
from struct2clust.config import Config
from struct2clust.hierarchy import (
    MIN_GROUPS,
    TIER_CAP,
    cap_tiers,
    merge_states,
    refine,
    separation_pairs,
)
from struct2clust.populations import population_uncertainty
from struct2clust.resources import available_memory, peak_memory, worker_count
from struct2clust.sampling import (
    MIN_BLOCKS,
    cluster_transitions,
    membership_autocorrelation,
    membership_sampling,
    population_blocks,
    sampling_check,
    trajectory_sampling,
)
from struct2clust.selection import Selection, select_partition
from struct2clust.stability import Stability, cluster_stability
from struct2clust.structures import (
    Ensemble,
    atom_mask,
    describe_selection,
    load_ensemble,
    load_trajectory,
    restrict_residues,
)

logger = logging.getLogger(__name__)

ATOM_LABEL = {"heavy-atoms": "heavy-atom", "backbone": "backbone", "anchor": "C3′/Cα"}


@dataclass
class Run:
    """Everything a run produces."""

    config: Config
    out_dir: Path
    ensemble: Ensemble          # after de-duplication
    distances: np.ndarray       # matrix of the de-duplicated ensemble
    metric_label: str           # e.g. "RMSD (backbone)"
    distance_unit: str          # "Å" or "" (dimensionless)
    n_input: int
    duplicates: list = field(default_factory=list)  # (kept name, removed name, distance)
    compared_atoms: dict = field(default_factory=dict)  # see structures.describe_selection
    clusterability: Optional[Clusterability] = None
    selection: Optional[Selection] = None
    labels: Optional[np.ndarray] = None   # 0-based cluster per member, -1 = noise
    stability: Optional[Stability] = None
    groups: Optional[np.ndarray] = None        # top-level cluster per structure
    paths: Optional[np.ndarray] = None         # deepest cluster per structure, e.g. "2.1"
    refinement: list = field(default_factory=list)
    top_level_merges: list = field(default_factory=list)  # clusters merged below the state scale
    substructure: Optional[np.ndarray] = None  # sub-group below the state scale, per structure
    medoids: dict = field(default_factory=dict)
    medoid_difference: Optional[pd.DataFrame] = None  # per-residue share, per medoid pair
    separation: list = field(default_factory=list)    # effect size per pair of final clusters
    sampling: list = field(default_factory=list)  # per trajectory: autocorrelation, blocks
    populations: Optional[dict] = None  # trajectories: population intervals, parts compared
    membership_tau: dict = field(default_factory=dict)  # trajectories: tau of every cluster
    sampling_check: Optional[dict] = None  # trajectories: are the blocks independent?
    first_pass: Optional[dict] = None  # input.membership_blocks: the pass it replaced
    n_groups: Optional[int] = None  # input.groups: number of resampling groups
    biology: Optional[Biology] = None

    @property
    def n_clusters(self) -> int:
        return int(self.labels.max() + 1) if self.labels is not None else 0


CONCLUSIONS = {
    "distinct_states":
        "Distinct states: the partition is reproducible and the data show group "
        "structure (multimodal pairwise distances; clusterability CLUSTERABLE).",
    "partition_not_confirmed":
        "Reproducible partition, group structure not confirmed: the clusters reappear in "
        "subsamples, but the test for group structure (dip test on the pairwise distances) "
        "did not confirm it. Either the data are a continuum that is cut consistently, or "
        "the states are unequal in size or many, where the test has little power "
        "(docs/METHODS.md, section 4). Treat the clusters as reproducible regions of the "
        "ensemble; their separation (separation.csv) and the map help to tell the cases "
        "apart.",
    "no_group_structure":
        "No cluster structure: the data show no group structure (NOT_CLUSTERABLE), so there "
        "is no evidence for more than one state. The partition is shown for inspection only.",
    "no_reproducible_partition":
        "No reproducible cluster structure: no partition has all clusters reproducible. "
        "The partition shown is exploratory.",
    "no_clusters":
        "No clusters: no HDBSCAN setting divides the structures into two or more clusters, so "
        "there is no evidence for more than one state; every structure is reported as noise.",
    "insufficient_clusters":
        "Insufficient clusters: the clusters of the resolution selection are closer than the "
        "separation threshold and are reported as one state, so the final partition has fewer "
        "than two clusters and does not support distinct states. This is not evidence that the "
        "ensemble is one state.",
    "insufficient_sampling":
        "Insufficient sampling: the trajectory blocks cannot be treated as independent (a "
        "trajectory holds fewer independent frames than blocks, a cluster's membership "
        "stays correlated for longer than one block, or no trajectory enters or leaves a "
        "cluster), so reproducibility cannot be assessed "
        "from independent data. The partition is exploratory; this is evidence neither for "
        "nor against distinct states.",
}

# conclusions about several clusters; they need at least two clusters in the final partition
NEEDS_TWO_CLUSTERS = ("distinct_states", "partition_not_confirmed", "no_group_structure")


def conclusion(reproducible: bool, verdict: Optional[str], found: bool = True,
               n_final: Optional[int] = None) -> str:
    """Overall result, graded by the strength of the evidence (docs/METHODS.md).

    Stability shows that a partition is reproducible, not that the data consist
    of groups: a curved or elongated continuum can be cut reproducibly (von
    Luxburg 2010). Distinct states are therefore claimed only when the data
    also show group structure. Given ``n_final`` (clusters reported after the
    effect-size rule and refinement), a conclusion about several clusters
    becomes ``insufficient_clusters`` when fewer than two remain.
    """
    before = _conclusion(reproducible, verdict, found)
    if n_final is not None and n_final < 2 and before in NEEDS_TWO_CLUSTERS:
        return "insufficient_clusters"
    return before


def _conclusion(reproducible: bool, verdict: Optional[str], found: bool) -> str:
    if not found:
        return "no_clusters"
    if not reproducible:
        return "no_reproducible_partition"
    if verdict == "NOT_CLUSTERABLE":
        return "no_group_structure"
    if verdict == "CLUSTERABLE":
        return "distinct_states"
    return "partition_not_confirmed"   # WEAK, or clusterability not tested


def run_conclusion(run: "Run") -> tuple[str, str]:
    """(final conclusion, conclusion before the effect-size rule) of a clustered run.

    Both come from :func:`conclusion` with the same inputs; the final one also uses the
    number of clusters reported. The second is diagnostic, not the run's conclusion.
    """
    sel, ca = run.selection, run.clusterability
    args = (bool(sel.stable), ca.verdict if ca else None, bool(sel.found_clusters))
    final, before = conclusion(*args, n_final=run.n_clusters), conclusion(*args)
    if run.sampling_check and not run.sampling_check["sufficient"]:
        final, before = (c if c == "no_clusters" else "insufficient_sampling"
                         for c in (final, before))
    return final, before


def conclusion_if_sampling_sufficient(run: "Run") -> Optional[str]:
    """Diagnostic for trajectories whose sampling check failed: the conclusion the
    resampling would give if the blocks were independent."""
    if not run.sampling_check or run.sampling_check["sufficient"]:
        return None
    sel, ca = run.selection, run.clusterability
    return conclusion(bool(sel.stable), ca.verdict if ca else None, bool(sel.found_clusters),
                      n_final=run.n_clusters)


def load_input(cfg: Config, input_path: Optional[Path]) -> Ensemble:
    if cfg.input.type == "trajectory":
        ens = load_trajectory(
            cfg.input.topology, cfg.input.trajectory, cfg.input.selection,
            cfg.input.stride, cfg.input.max_frames,
        )
    else:
        ens = load_ensemble(input_path)
    return restrict_residues(ens, cfg.input.residues) if cfg.input.residues else ens


def compute_distances(cfg: Config, ens: Ensemble, cache_dir: Path) -> tuple[np.ndarray, str, str]:
    """Distance matrix for the configured metric, cached on disk by content hash."""
    name, atoms = cfg.metric.name, cfg.metric.atoms
    key = hashlib.sha1()
    key.update(ens.coords.tobytes())
    key.update(repr((name, atoms, cfg.metric.equivalent_atoms)).encode())
    cache = cache_dir / f"distances_{key.hexdigest()[:16]}.npy"
    if name == "rmsd":
        label, unit = f"{ATOM_LABEL[atoms].capitalize()} RMSD", "Å"
        label += (" (equivalent atoms matched)" if cfg.metric.equivalent_atoms
                  else " (atoms paired by name)")
    else:
        label, unit = "1-TM-score", ""

    if cache.exists():
        logger.info("Distance matrix loaded from cache (%s)", cache.name)
        return np.load(cache), label, unit

    t0 = time.time()
    dist = metrics.distance_matrix(
        ens, name, atom_mask=atom_mask(ens.atoms, atoms),
        usalign_exe=cfg.metric.usalign_exe, n_jobs=cfg.n_jobs,
        equivalent_atoms=cfg.metric.equivalent_atoms,
    )
    logger.info("Distance matrix %dx%d (%s) in %.1f s",
                len(dist), len(dist), label, time.time() - t0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, dist)
    return dist, label, unit


def clustering_atoms(cfg: Config) -> str:
    """Atom subset the distance is computed on (tm: one anchor atom per residue)."""
    return "anchor" if cfg.metric.name == "tm" else cfg.metric.atoms


@dataclass
class Superposition:
    """How structures are superposed for display and comparison: on the atoms and
    with the criterion of the clustering distance (metrics.superpose)."""

    mask: np.ndarray
    metric: str
    d0: Optional[float] = None


def superposition(cfg: Config, ens: Ensemble) -> Superposition:
    mask = atom_mask(ens.atoms, clustering_atoms(cfg))
    if cfg.metric.name == "tm":
        return Superposition(mask, "tm", metrics.tm_d0(int(mask.sum()), ens.molecule_type))
    return Superposition(mask, "rmsd")


def compared_atoms(cfg: Config, ens: Ensemble) -> dict:
    """Atoms entering the distance: {"compared": describe_selection(...)}."""
    return {"compared": describe_selection(ens.atoms, atom_mask(ens.atoms, clustering_atoms(cfg)))}


# what a run writes into its directory (output.write_results); .cache is kept and reused
RUN_FILES = ("clusters.csv", "duplicates.csv", "selection.csv", "cluster_summary.csv",
             "separation.csv", "populations.csv", "population_differences.csv",
             "medoid_difference.csv", "run_summary.json", "report.md", "distance_matrix.npy",
             "distance_matrix_labels.txt")
RUN_DIRS = ("medoids", "cluster_views", "biology", "figures")


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    """Refuse to write into a directory that holds results unless ``overwrite``.

    A run writes only the files of its own outcome, so files of an earlier run in the
    same directory (e.g. a third cluster's medoid, or a summary of clusters that a new run
    did not find) would remain next to the new ones. With ``overwrite`` the files and
    directories of an earlier run are removed first; anything else is left in place and
    named in the log. The distance cache (.cache) is kept.
    """
    import shutil

    present = [p for p in out_dir.iterdir() if p.name != ".cache"] if out_dir.is_dir() else []
    if not present:
        return
    if not overwrite:
        raise FileExistsError(
            f"{out_dir} already holds results ({', '.join(sorted(p.name for p in present)[:5])}"
            f"{', ...' if len(present) > 5 else ''}). Choose another run name or output "
            "directory, or pass --overwrite to replace them.")
    for p in present:
        if p.is_dir() and p.name in RUN_DIRS:
            shutil.rmtree(p)
        elif p.is_file() and p.name in RUN_FILES:
            p.unlink()
    other = sorted(p.name for p in out_dir.iterdir() if p.name != ".cache")
    if other:
        logger.warning("Kept in %s, not written by a run (may refer to the earlier results): "
                       "%s", out_dir, ", ".join(other))


def near_duplicates(cfg: Config, dist: np.ndarray, names: list[str]
                    ) -> tuple[np.ndarray, list[tuple[int, int, float]], str]:
    """(kept indices, removed [(centre, copy, distance)], rule) of the near-duplicate
    removal (docs/METHODS.md, section 3). Trajectory frames are all kept: they are
    samples in time, and a frame that returns close to an earlier one is a revisit
    that counts towards the populations and the autocorrelation, not a copy."""
    if cfg.input.type == "trajectory":
        return np.arange(len(dist)), [], ""
    threshold = cfg.metric.dedup_threshold
    if threshold is None:
        threshold = metrics.DEFAULT_DEDUP[cfg.metric.name]
    if cfg.metric.name == "tm":
        rule = f"1 - TM-score <= {threshold:.3g}, i.e. TM-score >= {1 - threshold:.4g}"
    else:
        rule = f"RMSD <= {threshold:.3g} A"
    kept, removed = metrics.deduplicate(dist, threshold, rule, names)
    return kept, removed, rule


def run_pipeline(
    cfg: Config, input_path: Optional[Path], out_dir: Path, write: bool = True,
    overwrite: bool = False,
) -> Run:
    """Run every stage; with ``write`` (default) also write tables, figures and report.
    An output directory that already holds results is refused unless ``overwrite``
    (prepare_out_dir)."""
    out_dir = Path(out_dir)
    if write:
        prepare_out_dir(out_dir, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    ens = load_input(cfg, input_path)
    dist, label, unit = compute_distances(cfg, ens, out_dir / ".cache")

    kept, removed, rule = near_duplicates(cfg, dist, ens.names)
    duplicates = [(ens.names[i], ens.names[j], d) for i, j, d in removed]
    n_input = len(ens)
    ens, dist = ens.subset(kept), dist[np.ix_(kept, kept)]
    if len(ens) < 10:
        raise ValueError(
            f"Only {len(ens)} distinct structures"
            + (f" remain after removing {len(removed)} near-duplicates ({n_input} read; "
               f"{rule})" if removed else "")
            + "; at least 10 are needed for clustering and resampling.")

    compared = compared_atoms(cfg, ens)
    for role, per_kind in compared.items():
        for kind, info in per_kind.items():
            logger.info(
                "%s atoms (%s): %s, %d atoms over %d residues; per residue type: %s",
                role.capitalize(), label, kind, info["n_atoms"], info["n_residues"],
                "; ".join(f"{rn}: {' '.join(nm)}" for rn, nm in info["atoms_by_residue"].items()),
            )

    np.save(out_dir / "distance_matrix.npy", dist)
    (out_dir / "distance_matrix_labels.txt").write_text("\n".join(ens.names) + "\n")

    run = Run(
        config=cfg, out_dir=out_dir, ensemble=ens, distances=dist,
        metric_label=label, distance_unit=unit, n_input=n_input, duplicates=duplicates,
        compared_atoms=compared,
    )
    cluster(run)
    if write:
        from struct2clust.output import write_results
        write_results(run)
    return run


# reported when the autocorrelation time is uncertain by more than half (a reporting choice)
TAU_ERROR_NOTE = 0.5


def sampling_notes(t: dict) -> list[str]:
    """Warnings about one trajectory's sampling (sampling.trajectory_sampling)."""
    notes = []
    if t["autocorrelation_time_rel_error"] > TAU_ERROR_NOTE:
        notes.append(
            f"the autocorrelation time is uncertain (relative error "
            f"{100 * t['autocorrelation_time_rel_error']:.0f}%): the trajectory is short "
            "compared with its correlation time")
    if t["block_shorter_than_autocorrelation_time"]:
        notes.append(
            f"resampling blocks of {t['block_length']:.0f} frames are shorter than the "
            f"autocorrelation time ({t['autocorrelation_time']:.0f} frames): the "
            f"{t['n_frames']} frames hold about {t['effective_frames']:.0f} independent "
            "ones, fewer than there are blocks, so stability and populations are less "
            "certain than they appear"
            + (" (fewer blocks: input.trajectory_blocks)" if t.get("blocks_rule") == "fixed"
               else f" (the automatic rule uses at least {MIN_BLOCKS} blocks)"))
    return notes


def population_notes(pop: dict) -> list[str]:
    """Warnings about populations that differ between trajectories (or halves)."""
    differ: dict = {}
    for c in pop["comparisons"]:
        if c["differs"]:
            differ.setdefault((c["part_a"], c["part_b"]), []).append(c["cluster"])
    return [f"Populations of {part_a} and {part_b} differ by more than their sampling "
            f"uncertainty for cluster(s) {', '.join(f'C{c}' for c in clusters)}: the parts "
            "have not sampled the same distribution, so the populations describe these "
            "simulations, not an equilibrium (METHODS.md, section 9)"
            for (part_a, part_b), clusters in differ.items()]


def top_level_risk(risk: float) -> float:
    """Monte Carlo risk budget of the top-level resolution selection: half of the run's
    risk; the other half is shared by the refinement selections (hierarchy.node_risk)."""
    return risk / 2


def resampling_blocks(ens: Ensemble, n_blocks) -> np.ndarray:
    """Block id per member: one block per structure for ensembles; for trajectories,
    contiguous runs of frames within each replica (a block never spans two replicas).
    ``n_blocks``: blocks per trajectory, one number for all or {replica: number}."""
    if not ens.replica:
        return np.arange(len(ens))
    blocks = np.empty(len(ens), dtype=int)
    offset = 0
    replica = np.asarray(ens.replica)
    for label in ens.replicas:
        idx = np.flatnonzero(replica == label)
        nb = n_blocks[label] if isinstance(n_blocks, dict) else n_blocks
        blocks[idx] = trajectory_blocks(len(idx), min(nb, len(idx))) + offset
        offset = blocks[idx].max() + 1
    return blocks


def read_groups(path) -> dict:
    """Group of every structure from a CSV file with the columns structure, group."""
    table = pd.read_csv(path, dtype=str)
    missing = {"structure", "group"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: columns {sorted(missing)} missing (need structure, group)")
    if table["structure"].duplicated().any():
        raise ValueError(f"{path}: structures listed twice: "
                         f"{sorted(table.loc[table['structure'].duplicated(), 'structure'])[:5]}")
    return dict(zip(table["structure"], table["group"]))


def group_blocks(names: list[str], groups: dict) -> np.ndarray:
    """Resampling block per structure: its group. Every structure must have one."""
    missing = [n for n in names if n not in groups]
    if missing:
        raise ValueError(f"{len(missing)} structures without a group, e.g. {missing[:5]}")
    return np.unique([groups[n] for n in names], return_inverse=True)[1]


def second_pass(run: Run) -> bool:
    """input.membership_blocks: if the sampling check failed only because cluster
    memberships stay correlated for longer than a block (sampling.membership_sampling),
    cluster again with blocks of about one membership autocorrelation time and keep that
    run, with the replaced pass summarised in ``run.first_pass``. Returns whether it ran."""
    blocks = membership_sampling(run.sampling, run.membership_tau, run.sampling_check)
    if blocks is None:
        return False
    sel, ca = run.selection, run.clusterability
    first = {
        "conclusion": run_conclusion(run)[0],
        "conclusion_if_sampling_sufficient": conclusion_if_sampling_sufficient(run),
        "reproducible": bool(sel.stable), "n_clusters": run.n_clusters,
        "verdict": ca.verdict if ca else None,
        "blocks": {t["name"]: t["n_blocks"] for t in run.sampling},
        "sampling_check": run.sampling_check}
    for t in blocks:
        logger.info("Trajectory %s: second pass with %d blocks of %.0f frames (membership "
                    "autocorrelation time %.0f frames)", t["name"], t["n_blocks"],
                    t["block_length"], t["membership_autocorrelation_time"])
    fresh = Run(config=run.config, out_dir=run.out_dir, ensemble=run.ensemble,
                distances=run.distances, metric_label=run.metric_label,
                distance_unit=run.distance_unit, n_input=run.n_input,
                duplicates=run.duplicates, compared_atoms=run.compared_atoms,
                sampling=blocks, first_pass=first)
    cluster(fresh)
    for f in fields(Run):
        setattr(run, f.name, getattr(fresh, f.name))
    return True


def cluster(run: Run) -> None:
    """Clusterability, resolution selection, final clustering, stability, refinement,
    medoids."""
    cfg, dist = run.config, run.distances
    available = available_memory()
    cfg.n_jobs = worker_count(cfg.n_jobs, len(dist), available)
    need = peak_memory(len(dist), cfg.n_jobs)
    logger.info("%d structures, %d workers: about %.1f GB of memory needed at the peak%s",
                len(dist), cfg.n_jobs, need / 1e9,
                f" ({available / 1e9:.1f} GB available)" if available else "")
    if available and need > available:
        logger.warning(
            "The run needs about %.1f GB, more than the %.1f GB available: it may be stopped "
            "for lack of memory. Use fewer workers (n_jobs) or fewer structures "
            "(input.stride, input.max_frames).", need / 1e9, available / 1e9)
    if run.ensemble.replica and not run.sampling:
        run.sampling = trajectory_sampling(run.ensemble.replica, dist,
                                           cfg.input.trajectory_blocks)
        for t in run.sampling:
            logger.info("Trajectory %s: autocorrelation time %.1f frames, %d resampling "
                        "blocks of %.0f frames (%s)", t["name"], t["autocorrelation_time"],
                        t["n_blocks"], t["block_length"], t["blocks_rule"])
            for text in sampling_notes(t):
                logger.warning("Trajectory %s: %s", t["name"], text)
    blocks = resampling_blocks(run.ensemble, {t["name"]: t["n_blocks"] for t in run.sampling})
    if cfg.grouped:
        blocks = group_blocks(run.ensemble.names, read_groups(cfg.input.groups))
        run.n_groups = len(np.unique(blocks))
        logger.info("Resampling by group (%s): %d groups; subsamples keep %.0f%% of the "
                    "groups; a cluster absent from a subsample counts as not reproduced",
                    cfg.input.groups, run.n_groups, 100 * cfg.input.group_fraction)
        if run.n_groups < MIN_GROUPS:
            raise ValueError(f"{run.n_groups} groups in {cfg.input.groups}: at least "
                             f"{MIN_GROUPS} independent groups are needed for resampling")
    if cfg.clusterability.enabled:
        c = cfg.clusterability
        run.clusterability = assess(dist, c.alpha, c.n_neighbors, c.max_k)

    s, st = cfg.selection, cfg.stability
    run.selection = select_partition(
        dist, blocks, st.high, cfg.selection_fraction, s.min_cluster_size, s.min_samples,
        seed=cfg.seed, n_jobs=cfg.n_jobs, resampling_risk=top_level_risk(s.resampling_risk),
        min_resamples=s.min_resamples, max_resamples=s.max_resamples,
        absent_zero=cfg.by_unit,
    )
    if not run.selection.found_clusters:
        run.labels = np.full(len(dist), -1)
        logger.info("Conclusion: %s", CONCLUSIONS["no_clusters"])
        return
    params = run.selection.params
    run.labels = relabel_by_size(hdbscan_labels(dist, *params))
    logger.info(
        "Final clustering: %d clusters, %.0f%% noise", run.n_clusters, 100 * np.mean(run.labels < 0)
    )
    if run.n_clusters == 0:
        logger.warning("HDBSCAN assigned every structure to noise.")
        return

    run.stability = cluster_stability(
        dist, run.labels, blocks, params, st.n_resamples, cfg.stability_fraction,
        st.high, st.moderate, seed=cfg.seed, n_jobs=cfg.n_jobs, absent_zero=cfg.by_unit,
    )
    table = run.stability.clusters
    # what the selection decided, kept apart from the independent estimate above
    table["selection_decision"] = "established" if run.selection.stable else "not_established"
    table["selection_estimate"] = run.selection.cluster_estimates
    table["jaccard_tier"] = table["tier"]
    verdict = run.clusterability.verdict if run.clusterability else None
    table["tier"] = cap_tiers(table["jaccard_tier"].tolist(), verdict)
    if verdict and (table["tier"] != table["jaccard_tier"]).any():
        logger.warning(
            "Cluster tiers capped at %s (clusterability: %s)", TIER_CAP[verdict], verdict
        )
    # effect size at the top level: clusters closer than the state scale are one state
    run.labels, run.stability, run.top_level_merges, top_sub = merge_states(
        dist, run.labels, run.stability, cfg, blocks)
    run.substructure = top_sub
    run.groups = run.labels.copy()
    run.paths = np.array([str(c + 1) if c >= 0 else "" for c in run.labels], dtype=object)
    run.stability.clusters["path"] = [str(c) for c in run.stability.clusters["cluster"]]
    if s.refine and run.selection.stable:
        ref = refine(dist, run.labels, run.stability, blocks, verdict, cfg)
        run.labels, run.stability, run.groups, run.paths, run.refinement = (
            ref.labels, ref.stability, ref.groups, ref.paths, ref.log)
        if ref.substructure is not None:
            run.substructure = (ref.substructure if run.substructure is None else
                                np.where(ref.substructure != "", ref.substructure,
                                         run.substructure))
    run.medoids = medoids(dist, run.labels)
    if run.ensemble.replica:
        run.membership_tau = membership_autocorrelation(run.labels, run.ensemble.replica)
        pop_blocks = population_blocks(run.sampling, run.membership_tau)
        run.populations = population_uncertainty(
            run.labels, resampling_blocks(run.ensemble, pop_blocks),
            run.ensemble.replica, cfg.seed)
        run.populations["blocks"] = pop_blocks
        for text in population_notes(run.populations):
            logger.warning(text)
        replica = np.asarray(run.ensemble.replica)
        run.sampling_check = sampling_check(
            run.sampling, run.membership_tau,
            cluster_transitions(run.labels, run.ensemble.replica),
            {c + 1: list(dict.fromkeys(replica[run.labels == c]))
             for c in range(run.n_clusters)})
        for text in run.sampling_check["reasons"]:
            logger.warning("Insufficient sampling: %s", text)
        if (cfg.input.membership_blocks and cfg.automatic_blocks and run.first_pass is None
                and second_pass(run)):
            return
    # effect size of every pair of final clusters, reported whether or not a rule used it
    run.separation = separation_pairs(dist, run.labels, blocks, cfg.stability_fraction,
                                      cfg.seed + 2)
    if run.n_clusters >= 2 and run.ensemble.atoms is not None:
        run.medoid_difference = medoid_differences(
            run.ensemble, run.labels, run.medoids, superposition(cfg, run.ensemble))
    final, before = run_conclusion(run)
    if final != before:
        logger.info("Before the effect-size rule (diagnostic): %s", before)
    logger.info("Conclusion: %s", CONCLUSIONS[final])

    if cfg.biology.enabled:
        b = cfg.biology
        run.biology = characterise(
            run.ensemble, run.labels, run.medoids, b.contact_cutoff, b.contact_min_separation,
            b.bond_max_distance)
