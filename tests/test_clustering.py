"""Clustering core: controls with known answers, plus the rules the method depends on."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from struct2clust.clusterability import assess
from struct2clust.clustering import (
    candidate_grid,
    hdbscan_labels,
    medoids,
    relabel_by_size,
    subsample,
    trajectory_blocks,
)
from struct2clust.config import Config
from struct2clust.hierarchy import _merge_groups, cap_tiers, merge_close, refine, stricter
from struct2clust.selection import BettingTest, evaluate, same_partition, select_partition
from struct2clust.stability import cluster_stability


def _run(points, seed=0):
    dist = cdist(points, points)
    blocks = np.arange(len(dist))
    verdict = assess(dist).verdict
    sel = select_partition(dist, blocks, seed=seed, n_jobs=1, max_resamples=200)
    labels = relabel_by_size(hdbscan_labels(dist, *sel.params))
    st = cluster_stability(dist, labels, blocks, sel.params, n_resamples=40, seed=seed, n_jobs=1)
    tiers = cap_tiers(st.clusters["tier"].tolist(), verdict) if len(st.clusters) else []
    return verdict, sel, labels, st, tiers


def test_positive_control_three_separated_groups():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(c, 1.0, (40, 5)) for c in (0, 5, 10)])
    verdict, sel, labels, st, tiers = _run(points)
    assert verdict == "CLUSTERABLE" and sel.stable
    assert adjusted_rand_score(np.repeat([0, 1, 2], 40), labels) > 0.95
    assert tiers == ["HIGH", "HIGH", "HIGH"]
    assigned = labels >= 0
    assert st.membership[assigned, labels[assigned]].mean() > 0.95


def test_nested_groups_are_not_under_clustered():
    """Two super-groups of three sub-groups each: the six sub-groups must be found."""
    rng = np.random.default_rng(1)
    centres = [np.r_[a, s, 0, 0, 0] for a in (0, 40) for s in (0, 8, 16)]
    points = np.vstack([rng.normal(c, 1.0, (30, 5)) for c in centres])
    _, sel, labels, _, tiers = _run(points)
    assert sel.stable and sel.n_clusters == 6
    assert adjusted_rand_score(np.repeat(range(6), 30), labels) > 0.9
    assert set(tiers) == {"HIGH"}


def _refined(points, seed=0, **selection):
    verdict, sel, labels, st, tiers = _run(points, seed)
    st.clusters["jaccard_tier"] = st.clusters["tier"]
    st.clusters["tier"] = tiers
    cfg = Config.model_validate({"seed": seed, "n_jobs": 1,
                                 "selection": {"max_resamples": 200, **selection},
                                 "stability": {"n_resamples": 40}})
    dist = cdist(points, points)
    return sel, refine(dist, labels, st, np.arange(len(dist)), verdict, cfg)


def test_refinement_resolves_states_hidden_by_a_dominant_split():
    """One far-away group dominates: EOM returns 2 clusters, refinement finds all 4."""
    rng = np.random.default_rng(3)
    inner = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 1.0, (60, 5)) for s in (0, 6, 12)])
    points = np.vstack([inner, rng.normal(np.r_[60, 0, 0, 0, 0], 1.0, (150, 5))])
    sel, ref = _refined(points)
    assert sel.n_clusters == 2
    truth = np.r_[np.repeat([0, 1, 2], 60), np.full(150, 3)]
    assert ref.labels.max() + 1 == 4
    assert adjusted_rand_score(truth, ref.labels) > 0.95
    tab = ref.stability.clusters
    assert sorted(tab["path"]) == ["1.1", "1.2", "1.3", "2"]
    assert set(ref.groups[ref.labels >= 0]) == {0, 1}
    assert ref.stability.membership.shape == (len(points), 4)
    assert all(e["split"] == (e["cluster"] == "1") for e in ref.log)


def test_refinement_reports_each_nodes_share_of_the_risk():
    """Every refinement attempt that runs a selection records node_risk(R, level, n, N)."""
    from struct2clust.hierarchy import node_risk

    rng = np.random.default_rng(3)
    inner = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 1.0, (60, 5)) for s in (0, 6, 12)])
    points = np.vstack([inner, rng.normal(np.r_[60, 0, 0, 0, 0], 1.0, (150, 5))])
    _, ref = _refined(points)
    risk = Config().selection.resampling_risk
    entries = [e for e in ref.log if "resampling_risk" in e]
    assert entries
    for e in entries:
        expected = node_risk(risk, e["cluster"].count("."), e["size"], len(points))
        assert e["resampling_risk"] == float(f"{expected:.3g}")
    assert sum(e["resampling_risk"] for e in entries) <= risk / 2


def test_refinement_keeps_homogeneous_clusters():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(c, 1.0, (40, 5)) for c in (0, 5, 10)])
    _, ref = _refined(points)
    assert ref.labels.max() + 1 == 3
    assert not any(e["split"] for e in ref.log)


def _tight_pair(seed=4):
    """Two tight, well-separated sub-groups whose medoids are much closer than the
    scatter of the other state (like one fold modelled by two protocols)."""
    rng = np.random.default_rng(seed)
    pair = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 0.15, (60, 5)) for s in (0, 1.2)])
    other = rng.normal(np.r_[40, 0, 0, 0, 0], 1.0, (150, 5))
    return np.vstack([pair, other])


def _pipeline(points, **selection):
    from struct2clust.pipeline import Run, cluster
    from struct2clust.structures import Ensemble

    n = len(points)
    cfg = Config.model_validate({"seed": 0, "n_jobs": 1, "biology": {"enabled": False},
                                 "selection": {"max_resamples": 200, **selection},
                                 "stability": {"n_resamples": 40}})
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)))
    run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(points, points),
              metric_label="", distance_unit="", n_input=n)
    cluster(run)
    return run


def test_groups_below_the_state_scale_are_one_state():
    """With the optional rule on, a tight pair much closer than the scatter of the other
    state is reported as one cluster with two labelled sub-groups."""
    run = _pipeline(_tight_pair(), refine_min_separation_ratio=1)
    assert run.n_clusters == 2
    pair = run.labels[:120]
    assert len(set(pair)) == 1 and pair[0] >= 0
    assert run.top_level_merges and run.top_level_merges[0]["pairs"][0]["ratio_to_state_scale"] < 1
    label = str(pair[0] + 1)
    assert set(run.substructure[:120]) == {label + "a", label + "b"}
    assert set(run.substructure[120:]) == {""}


def test_two_final_clusters_keep_the_conclusion(tmp_path):
    """The tight pair is merged, but two clusters remain: the conclusion is unchanged."""
    from struct2clust.output import run_summary
    from struct2clust.structures import Atoms

    run = _pipeline(_tight_pair(), refine_min_separation_ratio=1)
    assert run.n_clusters == 2 and run.top_level_merges
    run.out_dir = tmp_path
    run.ensemble.atoms = Atoms(*(np.array([v]) for v in ("A", 1, "G", "C3'", "C", "nucleic", 0)))
    res = run_summary(run, {})["result"]
    assert res["conclusion"] == res["conclusion_before_effect_size"] != "insufficient_clusters"


def test_effect_size_is_reported_not_applied_by_default(tmp_path):
    """Default: the tight pair stays two clusters; its ratio < 1 is reported for every
    pair (separation.csv, run_summary), described in the report, and merges nothing."""
    from struct2clust.output import cluster_summary, report, run_summary, write_separation
    from struct2clust.structures import Atoms

    run = _pipeline(_tight_pair())
    assert run.n_clusters == 3
    assert run.substructure is None and not run.top_level_merges
    assert len(run.separation) == 3                                  # 3 clusters, 3 pairs
    tight = min(run.separation, key=lambda p: p["ratio_to_state_scale"])
    assert tight["ratio_to_state_scale"] < 1 < max(
        p["ratio_to_state_scale"] for p in run.separation)
    run.out_dir = tmp_path
    run.ensemble.atoms = Atoms(*(np.array([v]) for v in ("A", 1, "G", "C3'", "C", "nucleic", 0)))
    summary = run_summary(run, {})
    assert summary["effect_size"]["pairs"] == run.separation
    assert summary["effect_size"]["rule"].startswith("off")
    text = report(run, summary, cluster_summary(run))
    assert "closer than the ensemble's states are wide" in text
    assert "reported as one state" not in text
    write_separation(run, tmp_path)
    table = pd.read_csv(tmp_path / "separation.csv")
    assert len(table) == 3 and {"ratio_q025", "ratio_q975", "state_scale"} <= set(table)


def test_separation_pairs():
    """Ratio = medoid distance / state scale, with its spread over subsamples; with two
    clusters there is no state scale."""
    from struct2clust.hierarchy import separation_pairs

    rng = np.random.default_rng(2)
    points = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 1.0, (50, 5)) for s in (0, 3, 30)])
    labels = np.repeat([0, 1, 2], 50)
    dist = cdist(points, points)
    pairs = separation_pairs(dist, labels, np.arange(150), 0.8, seed=0, n=100)
    assert [(p["a"], p["b"]) for p in pairs] == [(1, 2), (1, 3), (2, 3)]
    for p in pairs:
        assert p["ratio_to_state_scale"] == pytest.approx(
            p["medoid_distance"] / p["state_scale"], rel=1e-2)
    low, high = pairs[0]["ratio_subsampling_range"]
    assert low <= pairs[0]["ratio_to_state_scale"] <= high
    two = separation_pairs(dist[:100, :100], labels[:100], np.arange(100), 0.8, seed=0)
    assert two[0]["state_scale"] is None and two[0]["ratio_subsampling_range"] is None
    assert separation_pairs(dist[:50, :50], labels[:50], np.arange(50), 0.8, seed=0) == []


def test_effect_size_rule_can_be_switched_off():
    run = _pipeline(_tight_pair(), refine_min_separation_ratio=0)
    assert run.n_clusters == 3
    assert run.substructure is None and not run.top_level_merges


def test_refinement_applies_the_rule_inside_a_cluster():
    """The pair hidden inside a cluster (found by refinement) is kept as sub-structure."""
    rng = np.random.default_rng(4)
    pair = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 0.15, (60, 5)) for s in (0, 1.2)])
    other = rng.normal(np.r_[40, 0, 0, 0, 0], 1.0, (150, 5))
    import pandas as pd

    from struct2clust.stability import Stability
    points = np.vstack([pair, other])
    labels = np.r_[np.zeros(120, int), np.ones(150, int)]          # pair given as one cluster
    st = Stability(pd.DataFrame({"cluster": [1, 2], "size": [120, 150], "jaccard_mean": [1.0, 1.0],
                                 "tier": ["HIGH", "HIGH"]}), np.eye(2)[labels], 40)
    cfg = Config.model_validate({"seed": 0, "n_jobs": 1,
                                 "selection": {"max_resamples": 200,
                                               "refine_min_separation_ratio": 1},
                                 "stability": {"n_resamples": 40}})
    ref = refine(cdist(points, points), labels, st, np.arange(270), "CLUSTERABLE", cfg)
    assert ref.labels.max() + 1 == 2
    entry = next(e for e in ref.log if e["cluster"] == "1")
    assert entry.get("substructure") and not entry["split"]
    assert set(ref.substructure[:120]) == {"1a", "1b"}


def test_absolute_separation_floor():
    """Three separated groups (5 apart): a floor above their separation merges nothing
    at the top level but blocks no split either; checked on merge_close directly."""
    md = np.array([[0, 5, 9], [5, 0, 6], [9, 6, 0]], float)
    assert list(merge_close(md, 4)) == [0, 1, 2]
    assert list(merge_close(md, 5.5)) == [0, 0, 1]
    assert list(merge_close(md, 10)) == [0, 0, 0]


def test_merge_groups_keeps_weakest_statistics_and_sums_membership():
    import pandas as pd
    tab = pd.DataFrame({"cluster": [1, 2, 3], "size": [5, 3, 2],
                        "jaccard_mean": [0.9, 0.8, 0.95], "tier": ["HIGH"] * 3})
    child = np.array([0] * 5 + [1] * 3 + [2] * 2 + [-1])
    membership = np.tile([0.6, 0.3, 0.1], (11, 1))
    new_child, new_tab, new_m = _merge_groups(tab, membership, child, np.array([0, 1, 1]))
    assert list(new_tab["size"]) == [5, 5] and list(new_tab["jaccard_mean"]) == [0.9, 0.8]
    assert list(new_child) == [0] * 5 + [1] * 5 + [-1]
    assert np.allclose(new_m[:, 1], 0.4)


def test_background_noise_does_not_create_clusters():
    """Three groups on a structureless background: exactly the three groups."""
    rng = np.random.default_rng(2)
    groups = [rng.normal(c, 0.6, (40, 5)) for c in (0, 5, 10)]
    points = np.vstack(groups + [rng.uniform(-3, 13, (120, 5))])
    sel, ref = _refined(points)
    labels = ref.labels
    assert sel.stable and sel.n_clusters == 3 and labels.max() + 1 == 3
    truth = np.r_[np.repeat([0, 1, 2], 40), -np.ones(120, int)]
    core = truth >= 0
    assert adjusted_rand_score(truth[core], labels[core]) > 0.9


@pytest.mark.parametrize("seed", [0, 1])
def test_negative_control_gaussian_noise(seed):
    points = np.random.default_rng(seed).normal(size=(150, 6))
    verdict, sel, _, _, tiers = _run(points, seed=seed)
    assert verdict == "NOT_CLUSTERABLE"
    assert not sel.stable
    assert all(t == "LOW" for t in tiers)


def test_negative_control_uniform_noise():
    points = np.random.default_rng(3).uniform(size=(200, 5))
    verdict, sel, _, _, tiers = _run(points)
    assert verdict == "NOT_CLUSTERABLE"
    assert not sel.stable
    assert all(t == "LOW" for t in tiers)


def test_cap_tiers():
    assert cap_tiers(["HIGH", "MODERATE", "LOW"], "CLUSTERABLE") == ["HIGH", "MODERATE", "LOW"]
    assert cap_tiers(["HIGH", "MODERATE", "LOW"], "WEAK") == ["MODERATE", "MODERATE", "LOW"]
    assert cap_tiers(["HIGH", "MODERATE"], "NOT_CLUSTERABLE") == ["LOW", "LOW"]
    assert cap_tiers(["HIGH"], None) == ["HIGH"]
    assert stricter("CLUSTERABLE", "WEAK") == "WEAK"
    assert stricter(None, "WEAK") == "WEAK" and stricter("CLUSTERABLE", None) == "CLUSTERABLE"


def test_candidate_grid():
    grid = candidate_grid(300)
    assert all(1 <= ms <= mcs for mcs, ms, _ in grid)
    mcs_values = sorted({m for m, _, _ in grid})
    assert mcs_values[:16] == list(range(5, 21)) and mcs_values[-1] == 30
    assert {meth for _, _, meth in grid} == {"eom"}
    assert candidate_grid(100, [4, 8], [1, 8]) == [(4, 1, "eom"), (8, 1, "eom"), (8, 8, "eom")]


def test_trajectory_blocks_are_contiguous_and_resampled_whole():
    blocks = trajectory_blocks(103, 10)
    assert len(np.unique(blocks)) == 10
    assert np.all(np.diff(blocks) >= 0)
    idx = subsample(np.random.default_rng(0), blocks, 0.8)
    chosen = np.unique(blocks[idx])
    assert len(chosen) == 8
    assert np.array_equal(idx, np.flatnonzero(np.isin(blocks, chosen)))


def test_relabel_by_size_and_medoids():
    labels = relabel_by_size(np.array([5, 5, -1, 2, 2, 2]))
    assert labels.tolist() == [1, 1, -1, 0, 0, 0]
    x = np.array([[0.0], [1.0], [9.0], [10.0], [11.0], [12.0]])
    d = cdist(x, x)
    assert medoids(d, np.array([1, 1, -1, 0, 0, 0])) == {0: 4, 1: 0}


def test_resampling_blocks_never_span_replicas():
    from struct2clust.pipeline import resampling_blocks
    from struct2clust.structures import Ensemble

    ens = Ensemble(names=[str(i) for i in range(30)], sources=[None] * 30, atoms=None,
                   coords=np.zeros((30, 1, 3)), replica=["A"] * 12 + ["B"] * 18,
                   frame=list(range(12)) + list(range(18)))
    blocks = resampling_blocks(ens, n_blocks=5)
    assert set(blocks[:12]).isdisjoint(blocks[12:])
    assert len(set(blocks[:12])) == 5 and len(set(blocks[12:])) == 5
    plain = Ensemble(names=["a", "b"], sources=[None] * 2, atoms=None, coords=np.zeros((2, 1, 3)))
    assert resampling_blocks(plain, 5).tolist() == [0, 1]


def test_conclusion_is_graded_by_evidence():
    from struct2clust.pipeline import conclusion

    assert conclusion(True, "CLUSTERABLE") == "distinct_states"
    assert conclusion(True, "WEAK") == "partition_not_confirmed"
    assert conclusion(True, None) == "partition_not_confirmed"
    assert conclusion(True, "NOT_CLUSTERABLE") == "no_group_structure"
    assert conclusion(False, "CLUSTERABLE") == "no_reproducible_partition"


@pytest.mark.parametrize("verdict", ["CLUSTERABLE", "WEAK", None, "NOT_CLUSTERABLE"])
def test_one_final_cluster_gives_insufficient_clusters(verdict):
    """A conclusion about several clusters needs at least two in the final partition
    (METHODS.md, section 7); with two or more, the rule changes nothing."""
    from struct2clust.pipeline import NEEDS_TWO_CLUSTERS, conclusion

    before = conclusion(True, verdict)
    assert before in NEEDS_TWO_CLUSTERS
    assert conclusion(True, verdict, n_final=1) == "insufficient_clusters"
    assert conclusion(True, verdict, n_final=2) == before
    assert conclusion(True, verdict, n_final=5) == before


def test_insufficient_clusters_leaves_the_other_conclusions_alone():
    from struct2clust.pipeline import conclusion

    for n in (0, 1, 3):
        assert conclusion(False, "CLUSTERABLE", n_final=n) == "no_reproducible_partition"
        assert conclusion(True, "CLUSTERABLE", found=False, n_final=n) == "no_clusters"


def test_validation_scripts_take_the_conclusion_from_run_conclusion():
    """The output module and the validation scripts must not derive the conclusion
    themselves, or they would miss the final-cluster check."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = [root / "src" / "struct2clust" / "output.py",
             *sorted((root / "validation").glob("*.py"))]
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\w.])conclusion\(", text), f.name
    users = [f.name for f in files if "run_conclusion(run)" in f.read_text(encoding="utf-8")]
    assert {"output.py", "run_study.py", "seed_reproducibility.py", "block_sensitivity.py",
            "symmetry_check.py"} <= set(users)


def test_spectral_eigengap_counts_disconnected_groups():
    from struct2clust.clusterability import spectral_eigengap

    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(c, 1.0, (60, 5)) for c in (0, 40, 80, 120)])
    assert spectral_eigengap(cdist(points, points))[1] == 4


def test_betting_test_errs_at_most_alpha_however_often_it_looks():
    """Values whose mean is exactly the threshold: neither side may be established
    in more than alpha of the runs, although the test looks after every value."""
    rng = np.random.default_rng(0)
    alpha, runs, wrong = 0.05, 400, 0
    for _ in range(runs):
        test = BettingTest(0.75)
        # two-peaked like per-subsample Jaccard values: mostly ~0.9, sometimes dissolved
        x = np.where(rng.random(500) < 0.8125, rng.uniform(0.85, 0.95, 500), 0.15)
        for v in x:
            test.update(v)
            if test.above(alpha) or test.below(alpha):
                wrong += 1
                break
    assert wrong / runs <= alpha + 0.02


def test_betting_test_decides_clear_cases():
    rng = np.random.default_rng(1)
    hi, lo = BettingTest(0.75), BettingTest(0.75)
    for _ in range(200):
        hi.update(rng.uniform(0.9, 1.0))
        lo.update(rng.uniform(0.3, 0.5))
    assert hi.above(1e-5) and not hi.below(0.5)
    assert lo.below(1e-5) and not lo.above(0.5)


def _two_groups(sep, n=40, seed=0):
    rng = np.random.default_rng(seed)
    points = np.vstack([rng.normal(0, 1, (n, 4)), rng.normal(np.r_[sep, 0, 0, 0], 1, (n, 4))])
    dist = cdist(points, points)
    labels = np.repeat([0, 1], n)
    return dist, labels


def test_sequential_evaluation_stops_early_when_the_decision_is_clear():
    dist, labels = _two_groups(20)
    ev = evaluate(dist, labels, np.arange(len(dist)), (10, 5, "eom"), 0.8, 0, 0.75,
                  1e-5, min_resamples=20, max_resamples=500)
    assert ev.qualifies and not ev.borderline
    assert ev.n_subsamples <= 60          # perfectly reproducible: decided early


def test_sequential_evaluation_marks_undecided_candidates_borderline():
    """A threshold at the true mean Jaccard cannot be resolved with few subsamples:
    the candidate is flagged and, its stability not being established, does not
    qualify."""
    rng = np.random.default_rng(2)
    points = rng.normal(0, 1, (80, 4))
    dist = cdist(points, points)
    labels = (points[:, 0] > 0).astype(int)          # one group cut in half
    blocks, params, alpha = np.arange(80), (5, 1, "eom"), 0.001
    truth = evaluate(dist, labels, blocks, params, 0.8, 1, 0.999, alpha, 300, 300).weakest
    ev = evaluate(dist, labels, blocks, params, 0.8, 0, truth, alpha, 20, 40)
    assert ev.borderline and ev.n_subsamples == 40
    assert not ev.qualifies


def test_same_partition_ignores_cluster_numbering_only():
    a = np.array([0, 0, 1, 1, -1])
    assert same_partition(a, np.array([1, 1, 0, 0, -1]))
    assert not same_partition(a, np.array([0, 0, 1, 1, 1]))      # noise differs
    assert not same_partition(a, np.array([0, 1, 1, 1, -1]))     # members move
    assert not same_partition(a, np.array([0, 0, 0, 0, -1]))     # two clusters joined


def test_tied_settings_with_different_partitions_are_flagged(monkeypatch):
    """Two qualifying settings with the same clusters and noise: flagged only if their
    partitions differ beyond the cluster numbering."""
    import struct2clust.selection as selection
    from struct2clust.selection import Evaluation

    a = np.repeat([0, 1], 10)
    moved = a.copy()
    moved[[9, 10]] = [1, 0]                                 # one member of each swaps

    def run(second):
        monkeypatch.setattr(selection, "hdbscan_labels",
                            lambda d, mcs, ms, m: a if mcs == 5 else second)
        monkeypatch.setattr(selection, "evaluate", lambda *args: Evaluation(
            0.9, 0.9, 0.01, 20, [0.9, 0.9], True, False))
        return select_partition(np.zeros((20, 20)), np.arange(20), mcs_values=[5, 6],
                                ms_values=[1], n_jobs=1)

    assert run(moved).tie_partitions_differ
    assert not run(1 - a).tie_partitions_differ             # same partition, renumbered


def test_selection_decision_does_not_depend_on_the_seed():
    """Two groups whose partition is clearly reproducible: every seed selects it."""
    rng = np.random.default_rng(5)
    points = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(np.r_[7, 0, 0, 0], 1, (30, 4))])
    dist = cdist(points, points)
    picks = {select_partition(dist, np.arange(60), seed=s, n_jobs=1, max_resamples=200).n_clusters
             for s in range(4)}
    assert picks == {2}


def test_separation_notes():
    """Two clusters: no state scale, so the pair is reported with its distances; and a
    medoid distance below the coordinate error of experimental structures is flagged."""
    from struct2clust.output import separation_notes

    rng = np.random.default_rng(1)
    points = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 0.01, (60, 5)) for s in (0, 0.3)])
    run = _pipeline(points, refine_min_separation_ratio=1)
    notes = separation_notes(run)
    assert [t["clusters"] for t in notes["no_reference"]["top_level_clusters"]] == [[1], [2]]
    assert notes["no_reference"]["medoid_distance"] == pytest.approx(0.3, abs=0.02)
    assert [(p["a"], p["b"]) for p in notes["below_coordinate_error"]] == [(1, 2)]


def test_partial_merge_labels_only_the_merged_parts():
    """Inside one cluster, two close sub-groups form one sub-cluster and a distant one
    stays separate: only the merged parts carry a substructure label."""
    from struct2clust.hierarchy import refine
    from struct2clust.stability import Stability

    rng = np.random.default_rng(0)
    pair = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 0.15, (60, 5)) for s in (0, 1.2)])
    third = rng.normal(np.r_[8, 0, 0, 0, 0], 0.6, (80, 5))
    other = rng.normal(np.r_[200, 0, 0, 0, 0], 1.0, (150, 5))
    points = np.vstack([pair, third, other])
    labels = np.r_[np.zeros(200, int), np.ones(150, int)]   # pair + third hidden as one
    table = pd.DataFrame({"cluster": [1, 2], "size": [200, 150], "jaccard_mean": [1.0, 1.0],
                          "tier": ["HIGH", "HIGH"]})
    stab = Stability(table, np.eye(2)[labels], 40)
    cfg = Config.model_validate({"seed": 0, "n_jobs": 1,
                                 "selection": {"max_resamples": 200,
                                               "refine_min_separation_ratio": 1},
                                 "stability": {"n_resamples": 40}})
    ref = refine(cdist(points, points), labels, stab, np.arange(len(points)), "CLUSTERABLE", cfg)
    entry = next(e for e in ref.log if e["cluster"] == "1")
    # candidates by size: 1 = the distant group (80), 2 and 3 = the pair (60 each)
    assert entry["split"] and entry["merged_subclusters"] == [{"into": "1.1", "parts": [2, 3]}]
    assert set(ref.paths[:120]) == {"1.1"} and set(ref.paths[120:200]) == {"1.2"}
    assert set(ref.substructure[:120]) == {"1.1a", "1.1b"}
    assert set(ref.substructure[120:]) == {""}


def test_absolute_floor_merges_two_clusters_and_report_is_written(tmp_path):
    """Two clusters closer than the absolute floor become one state; with a single
    top-level cluster there is no state scale, and the report must still be written."""
    from struct2clust.output import cluster_summary, clusters_table, report, run_summary
    from struct2clust.structures import Atoms

    rng = np.random.default_rng(1)
    points = np.vstack([rng.normal(np.r_[s, 0, 0, 0, 0], 0.01, (60, 5)) for s in (0, 0.3)])
    run = _pipeline(points, refine_min_separation=0.5)
    assert run.n_clusters == 1 and set(run.substructure) == {"1a", "1b"}
    run.out_dir = tmp_path
    run.ensemble.atoms = Atoms(*(np.array([v]) for v in ("A", 1, "G", "C3'", "C", "nucleic", 0)))
    summary = run_summary(run, {})
    text = report(run, summary, cluster_summary(run))
    assert "one state" in text
    # the merge is due to the floor, so the report must not name the state scale alone
    assert "separation threshold" in text and "closer than the state scale" not in text
    # one final cluster: no claim about several clusters; the earlier result is kept apart
    res = summary["result"]
    assert res["n_clusters"] == 1 and res["conclusion"] == "insufficient_clusters"
    assert res["conclusion_before_effect_size"] == "distinct_states"
    lines = text.splitlines()
    assert lines[2].startswith("**Insufficient clusters")
    diag = next(i for i, x in enumerate(lines) if x.startswith("Before the effect-size rule"))
    assert diag > 2 and "`distinct_states`" in lines[diag] and "not the conclusion" in lines[diag]
    table = clusters_table(run)       # one number per cluster; sub-groups numbered 1, 2
    assert "path" not in table and set(table["top_level"]) == {1}
    assert set(table["substructure"]) == {1, 2}


def test_autocorrelation_time_of_a_slow_walk():
    """An AR(1) series with phi = 0.9 has tau = (1 + phi) / (1 - phi) = 19; white noise 1."""
    from struct2clust.sampling import integrated_autocorrelation_time

    rng = np.random.default_rng(0)
    x = np.zeros(20000)
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    tau, err = integrated_autocorrelation_time(x)
    assert tau == pytest.approx(19, rel=0.2) and err < 0.25
    tau, err = integrated_autocorrelation_time(rng.normal(size=5000))
    assert tau == pytest.approx(1, abs=0.2) and err < 0.15
    # a short series with a long correlation time: the estimate is uncertain
    assert integrated_autocorrelation_time(np.cumsum(rng.normal(size=60)))[1] > 0.5


def test_trajectory_sampling_flags_short_blocks():
    from struct2clust.sampling import trajectory_sampling

    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=(400, 3)), axis=0)       # strongly correlated
    noise = rng.normal(size=(400, 3))                          # independent frames
    x = np.vstack([walk, noise])
    out = trajectory_sampling(["walk"] * 400 + ["noise"] * 400, cdist(x, x), 50)
    by = {t["name"]: t for t in out}
    assert by["walk"]["block_shorter_than_autocorrelation_time"]
    assert not by["noise"]["block_shorter_than_autocorrelation_time"]
    assert by["walk"]["effective_frames"] < 40 < by["noise"]["effective_frames"]


def test_trajectory_sampling_reports_the_error_of_the_autocorrelation_time():
    """The relative error is that of the estimator; a short, slow trajectory gets a note."""
    from struct2clust.pipeline import sampling_notes
    from struct2clust.sampling import (
        integrated_autocorrelation_time,
        principal_coordinate,
        trajectory_sampling,
    )

    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=(60, 3)), axis=0)        # short and slow
    noise = rng.normal(size=(400, 3))
    x = np.vstack([walk, noise])
    out = trajectory_sampling(["walk"] * 60 + ["noise"] * 400, cdist(x, x), 10)
    by = {t["name"]: t for t in out}
    for name, part in (("walk", walk), ("noise", noise)):
        _, err = integrated_autocorrelation_time(principal_coordinate(cdist(part, part)))
        assert by[name]["autocorrelation_time_rel_error"] == round(err, 2)
    assert by["walk"]["autocorrelation_time_rel_error"] > 0.5 > by["noise"][
        "autocorrelation_time_rel_error"]
    assert any("uncertain" in n for n in sampling_notes(by["walk"]))
    assert not any("uncertain" in n for n in sampling_notes(by["noise"]))


def test_automatic_blocks_hold_about_one_autocorrelation_time():
    """Default trajectory blocks: n / tau of them, between 8 and 50, never more than frames;
    a correlated walk gets fewer, longer blocks than independent frames."""
    from struct2clust.sampling import MAX_BLOCKS, MIN_BLOCKS, automatic_blocks, trajectory_sampling

    assert automatic_blocks(200, 22.0) == 9
    assert automatic_blocks(200, 60.0) == MIN_BLOCKS
    assert automatic_blocks(1000, 1.0) == MAX_BLOCKS
    assert automatic_blocks(6, 1.0) == 6
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=(400, 3)), axis=0)
    noise = rng.normal(size=(400, 3))
    x = np.vstack([walk, noise])
    by = {t["name"]: t for t in trajectory_sampling(["walk"] * 400 + ["noise"] * 400,
                                                     cdist(x, x))}
    assert by["noise"]["n_blocks"] == MAX_BLOCKS and by["walk"]["n_blocks"] < MAX_BLOCKS
    assert by["walk"]["blocks_rule"] == "automatic"
    for t in by.values():
        assert t["block_shorter_than_autocorrelation_time"] == (t["n_blocks"] == MIN_BLOCKS
                                                                and t["effective_frames"] < 8)


def test_automatic_blocks_are_resampled_like_groups():
    """Automatic trajectory blocks are independent units: half of them per subsample, an
    absent cluster counts as not reproduced; a fixed number keeps the frame fractions."""
    from struct2clust.config import Config

    auto = Config.model_validate({"input": {"type": "trajectory", "topology": "t.pdb",
                                            "trajectory": "x.dcd"}})
    assert auto.by_unit and auto.selection_fraction == auto.stability_fraction == 0.5
    fixed = auto.model_copy(deep=True)
    fixed.input.trajectory_blocks = 50
    assert not fixed.by_unit and fixed.selection_fraction == 0.8
    assert not Config().by_unit


def test_sampling_check_separates_insufficient_sampling():
    """Blocks shorter than the geometric autocorrelation time, or a cluster whose membership
    stays correlated for longer than a block, fail the check; a fast-switching trajectory
    with long enough blocks passes."""
    from struct2clust.sampling import membership_autocorrelation, sampling_check

    rng = np.random.default_rng(0)
    fast = rng.integers(0, 2, 400)                         # switches every few frames
    slow = np.repeat([0, 1, 0, 1], 100)                     # three transitions
    tau = membership_autocorrelation(np.r_[fast, slow], ["fast"] * 400 + ["slow"] * 400)
    assert tau["fast"][1] < 3 < 25 < tau["slow"][1]
    ok = {"name": "fast", "block_length": 25.0, "autocorrelation_time": 1.0,
          "effective_frames": 400.0, "block_shorter_than_autocorrelation_time": False}
    assert sampling_check([ok], {"fast": tau["fast"]})["sufficient"]
    slow_t = dict(ok, name="slow")
    check = sampling_check([slow_t], {"slow": tau["slow"]})
    assert not check["sufficient"] and check["clusters_longer_than_block"] == [1, 2]
    short = dict(ok, block_shorter_than_autocorrelation_time=True)
    assert not sampling_check([short], {})["sufficient"]


def test_clusters_without_transitions_fail_the_sampling_check():
    """Two trajectories that each stay in their own cluster (no transition, noise frames
    aside) fail the check; a cluster that is entered and left passes this part."""
    from struct2clust.sampling import cluster_transitions, sampling_check

    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    labels[[10, 50]] = -1                                    # noise frames inside a run
    replica = ["A"] * 100 + ["B"] * 100
    moves = cluster_transitions(labels, replica)
    assert moves == {1: 0, 2: 0}
    t = {"block_length": 25.0, "autocorrelation_time": 1.0, "effective_frames": 100.0,
         "block_shorter_than_autocorrelation_time": False}
    check = sampling_check([dict(t, name="A"), dict(t, name="B")], {}, moves,
                           {1: ["A"], 2: ["B"]})
    assert not check["sufficient"] and check["clusters_without_transitions"] == [1, 2]
    assert "trajectory A" in check["reasons"][0]
    mixed = np.tile(np.repeat([0, 1], 5), 20)
    assert cluster_transitions(mixed, ["A"] * 200) == {1: 39, 2: 39}
    assert sampling_check([dict(t, name="A")], {},
                          cluster_transitions(mixed, ["A"] * 200))["sufficient"]


def test_membership_blocks_only_when_membership_alone_fails():
    """The second pass (input.membership_blocks) uses blocks of about one membership
    autocorrelation time, and only when the check failed for that reason alone and the
    trajectory is at least 8 such times long."""
    from struct2clust.sampling import membership_sampling

    t = {"name": "A", "n_frames": 1000, "n_blocks": 40, "block_length": 25.0,
         "autocorrelation_time": 20.0, "effective_frames": 50.0, "blocks_rule": "automatic",
         "block_shorter_than_autocorrelation_time": False}
    failed = {"sufficient": False, "clusters_longer_than_block": [1],
              "clusters_without_transitions": []}
    out = membership_sampling([t], {"A": {1: 60.0, 2: 40.0}}, failed)
    assert out[0]["n_blocks"] == 16 and out[0]["blocks_rule"] == "membership"
    assert out[0]["block_length"] >= 60 and out[0]["membership_autocorrelation_time"] == 60
    assert membership_sampling([t], {"A": {1: 60.0}}, {**failed, "sufficient": True}) is None
    assert membership_sampling([t], {"A": {1: 200.0}}, failed) is None      # 5 x tau only
    assert membership_sampling([t], {"A": {1: 60.0}},
                               {**failed, "clusters_without_transitions": [2]}) is None
    short = dict(t, block_shorter_than_autocorrelation_time=True)
    assert membership_sampling([short], {"A": {1: 60.0}}, failed) is None


def test_second_pass_on_a_two_state_trajectory():
    """A slowly switching two-state trajectory with input.membership_blocks: the run either
    passes the check at once or ends with membership blocks and its first pass recorded."""
    from struct2clust.pipeline import Run, cluster, run_conclusion
    from struct2clust.structures import Ensemble

    rng = np.random.default_rng(3)
    n = 600
    s = np.repeat(np.tile([0, 1], 10), 30)[:n]                 # 19 transitions, dwell 30
    x = rng.normal(size=(n, 3)) * 0.5
    x[:, 0] += 6 * s
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=["A"] * n, frame=list(range(n)))
    cfg = Config.model_validate({
        "seed": 0, "n_jobs": 1, "biology": {"enabled": False},
        "input": {"type": "trajectory", "topology": "t.pdb", "trajectory": "t.xtc",
                  "membership_blocks": True}})
    run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
              metric_label="", distance_unit="", n_input=n)
    cluster(run)
    assert run.n_clusters == 2
    if run.first_pass is not None:
        assert run.sampling[0]["blocks_rule"] == "membership"
        assert run.first_pass["sampling_check"]["clusters_longer_than_block"]
        assert run.sampling[0]["n_blocks"] < run.first_pass["blocks"]["A"]
    else:
        assert run.sampling_check["sufficient"]
    assert run_conclusion(run)[0] in ("distinct_states", "insufficient_sampling")


def test_trajectory_frames_are_not_removed_as_near_duplicates():
    """A state revisited with the same coordinates: copies in an ensemble, frames of a
    trajectory."""
    from struct2clust.pipeline import near_duplicates

    x = np.array([[0.0], [5.0], [0.0], [5.0], [0.05]])
    dist = cdist(x, x)
    names = [str(i) for i in range(len(x))]
    ensemble = Config.model_validate({})
    kept, removed, _ = near_duplicates(ensemble, dist, names)
    assert len(kept) == 2 and len(removed) == 3
    trajectory = Config.model_validate({"input": {"type": "trajectory", "topology": "t.pdb",
                                                  "trajectory": "t.xtc"}})
    kept, removed, _ = near_duplicates(trajectory, dist, names)
    assert list(kept) == list(range(len(x))) and removed == []


def test_insufficient_sampling_overrides_the_conclusion():
    from types import SimpleNamespace

    from struct2clust.pipeline import conclusion_if_sampling_sufficient, run_conclusion

    run = SimpleNamespace(
        selection=SimpleNamespace(stable=True, found_clusters=True),
        clusterability=SimpleNamespace(verdict="CLUSTERABLE"), n_clusters=3,
        sampling_check={"sufficient": False})
    assert run_conclusion(run) == ("insufficient_sampling", "insufficient_sampling")
    assert conclusion_if_sampling_sufficient(run) == "distinct_states"
    run.sampling_check = {"sufficient": True}
    assert run_conclusion(run)[0] == "distinct_states"
    assert conclusion_if_sampling_sufficient(run) is None


def test_resampling_blocks_per_replica():
    from struct2clust.pipeline import resampling_blocks
    from struct2clust.structures import Ensemble

    ens = Ensemble(names=[str(i) for i in range(40)], sources=[None] * 40, atoms=None,
                   coords=np.zeros((40, 1, 3)), replica=["A"] * 20 + ["B"] * 20,
                   frame=list(range(20)) * 2)
    blocks = resampling_blocks(ens, {"A": 4, "B": 10})
    assert len(set(blocks[:20])) == 4 and len(set(blocks[20:])) == 10


def test_population_intervals_and_part_agreement():
    """Block-bootstrap intervals contain the observed populations; parts with the same
    populations agree, parts in different clusters are flagged; one trajectory is split
    into halves."""
    from struct2clust.populations import population_uncertainty

    rng = np.random.default_rng(0)
    n = 400
    same = rng.integers(0, 2, 2 * n)
    replica = ["A"] * n + ["B"] * n
    blocks = np.repeat(np.arange(40), 20)
    out = population_uncertainty(same, blocks, replica, seed=0, n_bootstrap=500)
    for c in range(2):
        p = np.mean(same == c)
        assert out["fraction_low"][c] <= p <= out["fraction_high"][c]
        assert out["fraction_high"][c] - out["fraction_low"][c] < 0.15
    assert set(out["parts"]) == {"A", "B"} and not any(c["differs"] for c in out["comparisons"])

    split = np.r_[np.zeros(n, int), np.ones(n, int)]          # A in cluster 1, B in cluster 2
    out = population_uncertainty(split, blocks, replica, seed=0, n_bootstrap=500)
    assert all(c["differs"] for c in out["comparisons"])

    one = population_uncertainty(split, blocks, ["A"] * (2 * n), seed=0, n_bootstrap=500)
    assert set(one["parts"]) == {"first_half", "second_half"}
    assert all(c["differs"] for c in one["comparisons"])


def test_separation_range_brackets_the_ratio():
    from struct2clust.hierarchy import separation_range

    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (60, 3))
    b = rng.normal([3, 0, 0], 1, (60, 3))
    ref = rng.normal([20, 0, 0], 1, (60, 3))
    x = np.vstack([a, b, ref])
    d = cdist(x, x)
    idx = np.arange(180)
    lo, hi = separation_range(d, idx[:60], idx[60:120], [idx[120:]], idx, 0.8, 0)
    ma, mb, mr = (idx[s][np.argmin(d[np.ix_(idx[s], idx[s])].sum(1))]
                  for s in (slice(0, 60), slice(60, 120), slice(120, 180)))
    ratio = d[ma, mb] / np.median(d[idx[120:], mr])
    assert lo < hi and lo <= ratio * 1.05 and ratio * 0.95 <= hi


def test_node_risks_add_up_to_at_most_half_the_run_risk():
    """Refinement budgets: nodes of one level are disjoint, levels halve, so any tree
    of nodes adds up to at most R/2 (the top-level selection takes the other half)."""
    from struct2clust.hierarchy import node_risk

    rng = np.random.default_rng(0)
    for _ in range(200):
        n_total, risk, total = 500, 0.001, 0.0
        level_sets = [[n_total]]
        for level in range(6):              # a random tree: every node splits into parts
            parts = []
            for n in level_sets[-1]:
                cuts = np.sort(rng.integers(0, n + 1, rng.integers(1, 4)))
                parts += [p for p in np.diff(np.r_[0, cuts, n]) if p > 1]
            for n in level_sets[-1]:
                total += node_risk(risk, level, n, n_total)
            level_sets.append(parts)
        assert total <= risk / 2 + 1e-15


def test_population_blocks_are_three_autocorrelation_times_long():
    """Population intervals use blocks of about three times the longer of the geometric
    and the longest membership autocorrelation time (8-50), the fixed blocks of a
    trajectory with fixed blocks."""
    from struct2clust.sampling import population_blocks

    auto = {"name": "A", "n_frames": 1000, "autocorrelation_time": 20.0, "n_blocks": 50,
            "blocks_rule": "automatic"}
    short = dict(auto, name="B", autocorrelation_time=100.0)
    fixed = dict(auto, name="C", n_blocks=12, blocks_rule="fixed")
    assert population_blocks([auto, short, fixed]) == {"A": 16, "B": 8, "C": 12}
    member = {"A": {1: 30.0, 2: 10.0}, "B": {1: 5.0}, "C": {1: 500.0}}
    assert population_blocks([auto, short, fixed], member) == {"A": 11, "B": 8, "C": 12}
