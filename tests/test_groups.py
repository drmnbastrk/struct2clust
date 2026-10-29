"""Resampling by group (input.groups): structures of one source are not independent."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from struct2clust.clustering import hdbscan_labels, relabel_by_size, subsample
from struct2clust.config import Config
from struct2clust.hierarchy import MIN_GROUPS, refine
from struct2clust.pipeline import group_blocks, read_groups
from struct2clust.selection import select_partition
from struct2clust.stability import cluster_stability, one_resample


def _copies(rng, centres, n_entries, n_copies, sd_entry=1.0, sd_copy=0.05):
    """Entries drawn around each centre, every entry repeated as near-identical copies."""
    points, entry = [], []
    for c in centres:
        for _ in range(n_entries):
            base = rng.normal(c, sd_entry, 6)
            points.append(base + rng.normal(0, sd_copy, (n_copies, 6)))
            entry += [len(set(entry))] * n_copies
    return np.vstack(points), np.array(entry)


def _select(points, blocks, grouped, seed=0):
    dist = cdist(points, points)
    return select_partition(dist, blocks, fraction=0.5 if grouped else 0.8, seed=seed,
                            n_jobs=1, max_resamples=200, absent_zero=grouped)


def test_subsamples_keep_or_drop_whole_groups():
    """Groups of different sizes, in mixed order: every subsample holds half of the groups,
    each with all of its structures."""
    rng = np.random.default_rng(6)
    blocks = rng.permutation(np.repeat(np.arange(20), rng.integers(1, 8, 20)))
    for s in range(50):
        idx = subsample(np.random.default_rng(s), blocks, 0.5)
        drawn = np.unique(blocks[idx])
        assert len(drawn) == 10
        assert np.array_equal(idx, np.flatnonzero(np.isin(blocks, drawn)))


def _grouped_run(tmp_path, n_groups):
    from struct2clust.pipeline import Run
    from struct2clust.structures import Ensemble

    rng = np.random.default_rng(7)
    points = np.vstack([rng.normal(c, 1.0, (40, 4)) for c in (0, 8)])
    n = len(points)
    path = tmp_path / "g.csv"
    path.write_text("structure,group\n" + "".join(f"{i},{i % n_groups}\n" for i in range(n)))
    cfg = Config.model_validate({"n_jobs": 1, "biology": {"enabled": False},
                                 "input": {"groups": str(path)},
                                 "selection": {"max_resamples": 100},
                                 "stability": {"n_resamples": 20}})
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=None, frame=None)
    return Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(points, points),
               metric_label="", distance_unit="", n_input=n)


def test_a_run_needs_at_least_8_groups(tmp_path):
    from struct2clust.pipeline import cluster

    assert MIN_GROUPS == 8
    with pytest.raises(ValueError, match="at least 8 independent groups"):
        cluster(_grouped_run(tmp_path, 7))
    run = _grouped_run(tmp_path, 8)
    cluster(run)
    assert run.n_groups == 8


def test_copies_of_structureless_entries_are_not_reproducible_by_group():
    """The 'copies of entries' scenario (METHODS section 12): resampled by structure,
    every set of copies is a reproducible cluster; resampled by entry, none is."""
    rng = np.random.default_rng(0)
    points, entry = _copies(rng, [np.zeros(6)], n_entries=30, n_copies=6)
    by_structure = _select(points, np.arange(len(points)), grouped=False)
    by_group = _select(points, entry, grouped=True)
    assert by_structure.stable and by_structure.n_clusters >= 2
    assert not by_group.stable


def test_states_made_of_many_entries_are_found_by_group():
    """Two states 8 sd apart, each of 12 entries with 6 copies: the two states, not
    one cluster per entry."""
    rng = np.random.default_rng(1)
    points, entry = _copies(rng, [np.zeros(6), np.r_[8.0, np.zeros(5)]], 12, 6)
    sel = _select(points, entry, grouped=True)
    assert sel.stable and sel.n_clusters == 2
    labels = relabel_by_size(hdbscan_labels(cdist(points, points), *sel.params))
    assert adjusted_rand_score(np.repeat([0, 1], 72), labels) > 0.9


def test_groups_do_not_hide_states_of_independent_structures():
    """Three separated groups whose structures come from 30 random sources."""
    rng = np.random.default_rng(2)
    points = np.vstack([rng.normal(c, 1.0, (40, 5)) for c in (0, 5, 10)])
    source = rng.integers(0, 30, len(points))
    sel = _select(points, source, grouped=True)
    assert sel.stable and sel.n_clusters == 3


def test_absent_cluster_counts_as_not_reproduced_only_by_group():
    rng = np.random.default_rng(3)
    points = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(20, 1, (20, 3))])
    dist = cdist(points, points)
    labels = np.repeat([0, 1], 20)
    blocks = np.r_[np.zeros(20, int), np.arange(1, 21)]  # cluster 0 is a single group
    params = (5, 5, "eom")
    for s in range(20):
        idx, jac, _ = one_resample(dist, labels, blocks, params, 0.5, s, absent_zero=True)
        if not np.isin(0, blocks[idx]):
            assert jac[0] == 0.0
            assert np.isnan(one_resample(dist, labels, blocks, params, 0.5, s)[1][0])
            break
    else:
        pytest.fail("group 0 was drawn in every subsample")


def test_without_groups_nothing_changes():
    """absent_zero=False is the default everywhere: resampling without groups."""
    rng = np.random.default_rng(4)
    points = np.vstack([rng.normal(c, 1.0, (30, 4)) for c in (0, 6)])
    dist = cdist(points, points)
    blocks = np.arange(len(dist))
    a = select_partition(dist, blocks, seed=3, n_jobs=1, max_resamples=100)
    b = select_partition(dist, blocks, seed=3, n_jobs=1, max_resamples=100, absent_zero=False)
    pd.testing.assert_frame_equal(a.candidates, b.candidates)
    cfg = Config()
    assert not cfg.grouped
    assert cfg.selection_fraction == cfg.selection.subsample_fraction == 0.8
    assert cfg.stability_fraction == cfg.stability.subsample_fraction == 0.8


def test_group_config(tmp_path):
    cfg = Config.model_validate({"input": {"groups": str(tmp_path / "g.csv")}})
    assert cfg.grouped and cfg.selection_fraction == cfg.stability_fraction == 0.5
    with pytest.raises(ValueError, match="trajectories"):
        Config.model_validate({"input": {"type": "trajectory", "topology": "t.psf",
                                         "trajectory": "t.dcd", "groups": "g.csv"}})


def test_reading_groups(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("structure,group\na,1\nb,1\nc,2\n")
    groups = read_groups(path)
    assert list(group_blocks(["c", "a", "b"], groups)) == [1, 0, 0]
    with pytest.raises(ValueError, match="without a group"):
        group_blocks(["a", "d"], groups)
    path.write_text("structure,source\na,1\n")
    with pytest.raises(ValueError, match="group"):
        read_groups(path)
    path.write_text("structure,group\na,1\na,2\n")
    with pytest.raises(ValueError, match="twice"):
        read_groups(path)


def test_refinement_does_not_split_a_node_with_few_groups(tmp_path):
    """Two sub-groups inside one cluster, but the cluster comes from 4 sources only."""
    rng = np.random.default_rng(5)
    inner = np.vstack([rng.normal(0, 0.5, (30, 4)), rng.normal(4, 0.5, (30, 4))])
    far = rng.normal(40, 1.0, (60, 4))
    points = np.vstack([inner, far])
    dist = cdist(points, points)
    blocks = np.r_[np.arange(60) % 4, 4 + np.arange(60)]   # 4 groups inside, 60 outside
    labels = np.r_[np.zeros(60, int), np.ones(60, int)]
    st = cluster_stability(dist, labels, blocks, (10, 5, "eom"), n_resamples=20,
                           fraction=0.5, seed=0, n_jobs=1, absent_zero=True)
    cfg = Config.model_validate({"n_jobs": 1, "input": {"groups": str(tmp_path / "g.csv")},
                                 "selection": {"max_resamples": 100},
                                 "stability": {"n_resamples": 20}})
    ref = refine(dist, labels, st, blocks, "CLUSTERABLE", cfg)
    first = next(e for e in ref.log if e["cluster"] == "1")
    assert first["n_groups"] == 4 < MIN_GROUPS and not first["split"]
    assert "too few independent groups" in first["reason"]
