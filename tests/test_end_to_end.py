"""The command line from PDB files to report, on a synthetic three-state RNA ensemble."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from helpers import rna_pdb

from struct2clust.cli import main

SEQ = "GACUGACU"
N_ATOMS = len(SEQ) * 8


@pytest.fixture(scope="module")
def ensemble(tmp_path_factory):
    """Three conformational states (20 models each) of an 8-nt chain."""
    root = tmp_path_factory.mktemp("ensemble")
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(size=(N_ATOMS, 3)) * 1.5, axis=0)
    states = []
    for s in range(3):
        xyz = base.copy()
        xyz[N_ATOMS // 2:] += rng.normal(size=3) * 8 * (s > 0) + s * np.array([0, 6.0, 0])
        states.append(xyz)
    k = 0
    for xyz in states:
        for _ in range(20):
            rna_pdb(root / f"model_{k}.pdb", xyz + rng.normal(scale=0.4, size=xyz.shape), seq=SEQ)
            k += 1
    truth = np.repeat([0, 1, 2], 20)
    return root, truth


@pytest.fixture(scope="module")
def fast_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("cfg") / "fast.yaml"
    path.write_text(yaml.safe_dump({
        "n_jobs": 1,
        "selection": {"max_resamples": 100},
        "stability": {"n_resamples": 30},
        "figures": {"formats": ["png"], "dpi": 100},
    }))
    return path


@pytest.fixture(scope="module")
def run_dir(ensemble, fast_config, tmp_path_factory):
    root, _ = ensemble
    out = tmp_path_factory.mktemp("results")
    result = CliRunner().invoke(main, ["run", str(root), "-o", str(out), "-n", "demo",
                                       "-c", str(fast_config)])
    assert result.exit_code == 0, result.output
    return out / "demo"


def test_outputs_exist(run_dir):
    for name in ["clusters.csv", "cluster_summary.csv", "selection.csv", "run_summary.json",
                 "report.md", "distance_matrix.npy", "distance_matrix_labels.txt",
                 "medoid_difference.csv",
                 "medoids/C1.pdb", "medoids/view_medoids.pml", "medoids/view_medoids.tcl",
                 "biology/rmsf.csv", "biology/contacts.csv", "biology/torsions.csv",
                 "figures/overview.png", "figures/distances.png", "figures/rmsf.png",
                 "figures/contacts.png", "figures/torsions.png"]:
        assert (run_dir / name).is_file(), name


def test_rerun_into_results_is_refused(run_dir, ensemble, fast_config):
    """A second run into a directory with results stops before computing anything."""
    root, _ = ensemble
    result = CliRunner().invoke(main, ["run", str(root), "-o", str(run_dir.parent), "-n",
                                       run_dir.name, "-c", str(fast_config)])
    assert result.exit_code != 0 and "--overwrite" in result.output


def test_overwrite_removes_the_earlier_results(tmp_path):
    """--overwrite removes what a run writes (also a stale medoid of a cluster the new run
    may not find), keeps the distance cache and leaves other files in place."""
    from struct2clust.pipeline import prepare_out_dir

    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "distances_x.npy").write_text("x")
    (tmp_path / "medoids").mkdir()
    (tmp_path / "medoids" / "C3.pdb").write_text("x")
    (tmp_path / "cluster_summary.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        prepare_out_dir(tmp_path, overwrite=False)
    prepare_out_dir(tmp_path, overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".cache", "notes.txt"]
    assert (tmp_path / ".cache" / "distances_x.npy").is_file()
    prepare_out_dir(tmp_path / "new", overwrite=False)      # absent directory: nothing to do


def test_states_are_recovered(run_dir, ensemble):
    from sklearn.metrics import adjusted_rand_score

    _, truth = ensemble
    clusters = pd.read_csv(run_dir / "clusters.csv")
    order = clusters["structure"].str.replace("model_", "").astype(int).to_numpy()
    assert adjusted_rand_score(truth[order], clusters["cluster"]) > 0.9
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["result"]["n_clusters"] == 3
    assert summary["selection"]["reproducible_partition_found"]
    assert summary["result"]["conclusion"] in ("distinct_states", "partition_not_confirmed")
    assert summary["distance"]["compared_atoms"]["compared"]["nucleic"]["n_atoms"] == N_ATOMS
    table = pd.read_csv(run_dir / "cluster_summary.csv")
    assert set(table["stability_tier"]) == {"HIGH"}
    # the selection's decision and estimate are reported apart from the independent one
    assert set(table["selection_decision"]) == {"established"}
    assert (table["selection_estimate"] >= 0.75).all()
    assert {"independent_stability_estimate", "independent_stability_low",
            "independent_stability_high"} <= set(table.columns)
    assert clusters["is_medoid"].sum() == 3


def test_cluster_views(run_dir):
    """One multi-model PDB and two viewer scripts per cluster; models run from the
    medoid to the farthest member, with falling opacity."""
    views = run_dir / "cluster_views"
    clusters = pd.read_csv(run_dir / "clusters.csv")
    members = pd.read_csv(views / "members.csv")
    for c in sorted(clusters.loc[clusters["cluster"] > 0, "cluster"].unique()):
        for suffix in ("_members.pdb", ".tcl", ".py"):
            assert (views / f"C{c}{suffix}").is_file()
        rows = members[members["cluster"] == c]
        in_cluster = clusters[clusters["cluster"] == c]
        assert len(rows) == len(in_cluster)
        medoid = in_cluster.loc[in_cluster["is_medoid"], "structure"].iloc[0]
        assert rows["structure"].iloc[0] == medoid
        assert rows["distance_to_medoid"].is_monotonic_increasing
        assert rows["opacity"].is_monotonic_decreasing and rows["opacity"].iloc[0] == 1.0
        pdb = (views / f"C{c}_members.pdb").read_text()
        assert pdb.count("ENDMDL") == len(rows)
        tcl = (views / f"C{c}.tcl").read_text()
        assert tcl.count("mol drawframes") == len(rows)
    compile((views / "C1.py").read_text(), "C1.py", "exec")   # valid Python


def test_duplicates_and_atom_pairing_are_reported(ensemble, tmp_path):
    """Exact copies are listed in duplicates.csv against the kept centre; the summary
    and the axis label state whether equivalent atoms were matched."""
    import shutil

    root, _ = ensemble
    data = tmp_path / "with_copies"
    shutil.copytree(root, data)
    shutil.copy(root / "model_0.pdb", data / "copy_a.pdb")
    shutil.copy(root / "model_25.pdb", data / "copy_b.pdb")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "n_jobs": 1, "metric": {"dedup_threshold": 0.01, "equivalent_atoms": False},
        "selection": {"max_resamples": 100}, "stability": {"n_resamples": 30},
        "biology": {"enabled": False}, "figures": {"enabled": False}}))
    result = CliRunner().invoke(main, ["run", str(data), "-o", str(tmp_path), "-n", "copies",
                                       "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    out = tmp_path / "copies"
    dup = pd.read_csv(out / "duplicates.csv")
    assert list(dup.columns) == ["structure", "duplicate_of", "distance"]
    assert {frozenset(p) for p in zip(dup["structure"], dup["duplicate_of"])} == {
        frozenset({"model_0", "copy_a"}), frozenset({"model_25", "copy_b"})}
    assert (dup["distance"] == 0).all()
    clustered = set(pd.read_csv(out / "clusters.csv")["structure"])
    assert set(dup["duplicate_of"]) <= clustered and not set(dup["structure"]) & clustered
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["distance"]["equivalent_atoms_matched"] is False
    assert summary["distance"]["label"].endswith("(atoms paired by name)")
    assert summary["input"]["n_used"] == 60


def test_equivalent_atoms_matched_by_default_is_reported(run_dir):
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["distance"]["equivalent_atoms_matched"] is True
    assert summary["distance"]["label"].endswith("(equivalent atoms matched)")


def _documented(section):
    """Names in the table under '## `section`' of docs/OUTPUTS.md: those in the first
    column, and all names written anywhere in the table."""
    import re
    from pathlib import Path

    text = (Path(__file__).parents[1] / "docs" / "OUTPUTS.md").read_text(encoding="utf-8")
    body = text.split(f"## `{section}`", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in body.splitlines() if r.startswith("|") and not r.startswith("|---")][1:]
    first = {n for r in rows for n in re.findall(r"`([^`]+)`", r.split("|")[1])}
    anywhere = {n for r in rows for n in re.findall(r"`([^`]+)`", r)}
    return first, anywhere


# written only for trajectories, or only when the case arises
CONDITIONAL = {"replica", "frame", "substructure", "fraction_low", "fraction_high",
               "membership_autocorrelation_time"}


def test_output_columns_match_the_documentation(run_dir):
    for name in ("clusters.csv", "cluster_summary.csv", "selection.csv",
                 "medoid_difference.csv"):
        first, anywhere = _documented(name)
        columns = set(pd.read_csv(run_dir / name, nrows=1).columns)
        assert first - CONDITIONAL - {"population_<replica>"} <= columns, name
        undocumented = {c for c in columns - anywhere if not c.startswith("population_")}
        assert not undocumented, (name, undocumented)
    first, _ = _documented("run_summary.json")
    summary = json.loads((run_dir / "run_summary.json").read_text())
    documented = {n.split(".")[0] for n in first}
    assert documented == set(summary), documented ^ set(summary)


def test_member_opacity():
    from struct2clust.output import member_opacity

    assert list(member_opacity(1)) == [1.0]
    op = member_opacity(25)
    assert op[0] == 1.0 and op[-1] == pytest.approx(0.1)
    assert np.all(np.diff(op) <= 0) and len(set(op)) <= 10


def test_composition_by_source(run_dir, tmp_path):
    """Sources equal to the true states: every cluster holds one source (purity 1)."""
    import pandas as pd

    clusters = pd.read_csv(run_dir / "clusters.csv")
    sources = tmp_path / "sources.csv"
    pd.DataFrame({"structure": clusters["structure"],
                  "source": [f"state {int(n.split('_')[-1]) // 20}"     # model_0 ... model_59
                             for n in clusters["structure"]]}).to_csv(sources, index=False)
    result = CliRunner().invoke(main, ["composition", str(run_dir), str(sources),
                                       "-o", str(tmp_path / "comp"), "--label", "State"])
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "comp" / "composition.json").read_text())
    assert all(c["purity"] == 1.0 for c in summary["clusters_table"] if c["cluster"] != "noise")
    for name in ["counts.csv", "fractions.csv", "by_source.csv", "composition.png"]:
        assert (tmp_path / "comp" / name).is_file(), name


def test_composition_with_incomplete_sources(run_dir, tmp_path):
    """Structures without a source are left out of the agreement; a file that matches no
    structure is refused with a message."""
    import pandas as pd

    names = pd.read_csv(run_dir / "clusters.csv")["structure"]
    partial = tmp_path / "partial.csv"
    pd.DataFrame({"structure": names[:30], "source": "A"}).to_csv(partial, index=False)
    result = CliRunner().invoke(main, ["composition", str(run_dir), str(partial),
                                       "-o", str(tmp_path / "partial")])
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "partial" / "composition.json").read_text())
    assert summary["n_without_source"] == 30
    assert summary["clusters"]["cramers_v"] is None     # one known source only

    unmatched = tmp_path / "unmatched.csv"
    pd.DataFrame({"structure": ["x", "y"], "source": ["A", "B"]}).to_csv(unmatched, index=False)
    result = CliRunner().invoke(main, ["composition", str(run_dir), str(unmatched)])
    assert result.exit_code != 0 and "no structure name matches" in result.output


def test_compare_run_with_itself(run_dir, tmp_path):
    result = CliRunner().invoke(main, ["compare", str(run_dir), str(run_dir), "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["adjusted_rand_index"] == pytest.approx(1.0)
    assert (tmp_path / "contingency.png").is_file()


def test_two_replicas_clustered_together(tmp_path, fast_config):
    """Replica SET1 samples states A/B, SET2 samples B/C: states are shared, populations differ."""
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(size=(N_ATOMS, 3)) * 1.5, axis=0)
    states = []
    for s in range(3):
        xyz = base.copy()
        xyz[N_ATOMS // 2:] += s * np.array([0, 7.0, 0])
        states.append(xyz)
    top = tmp_path / "top.pdb"
    rna_pdb(top, base, seq=SEQ, extra=False)
    paths = []
    for rep, seq_states in (("SET1", [0] * 20 + [1] * 20), ("SET2", [1] * 20 + [2] * 20)):
        (tmp_path / rep).mkdir()
        models = []
        for k, s in enumerate(seq_states):
            f = tmp_path / "tmp.pdb"
            rna_pdb(f, states[s] + rng.normal(scale=0.3, size=base.shape), seq=SEQ, extra=False)
            models.append(f"MODEL {k}\n" + f.read_text().replace("END\n", "") + "ENDMDL\n")
        p = tmp_path / rep / "traj.pdb"
        p.write_text("".join(models) + "END\n")
        paths.append(p)
    out = tmp_path / "out"
    args = ["run", "--topology", str(top), "-o", str(out), "-n", "reps", "-c", str(fast_config)]
    for p in paths:
        args += ["--trajectory", str(p)]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    clusters = pd.read_csv(out / "reps" / "clusters.csv")
    assert set(clusters["replica"]) == {"SET1", "SET2"}
    summary = pd.read_csv(out / "reps" / "cluster_summary.csv")
    assert len(summary) == 3
    assert {"population_SET1", "population_SET2"} <= set(summary.columns)
    # the shared state is populated in both replicas, the others in one each
    shared = summary.loc[(summary["population_SET1"] > 0.3) & (summary["population_SET2"] > 0.3)]
    assert len(shared) == 1
    assert (out / "reps" / "figures" / "timeline.png").is_file()
    assert (out / "reps" / "figures" / "populations.png").is_file()
    assert "Cluster populations by trajectory" in (out / "reps" / "report.md").read_text(
        encoding="utf-8")
    # populations with intervals; the replicas sample different states, so they differ
    assert {"fraction_low", "fraction_high"} <= set(summary.columns)
    assert (out / "reps" / "populations.csv").is_file()
    run_summary = json.loads((out / "reps" / "run_summary.json").read_text())
    assert run_summary["result"]["populations"]["parts_agree"] is False


def test_no_clusters_writes_a_report(tmp_path):
    """One tight group in which no setting finds two clusters: the run still writes its
    tables and a report that says so."""
    rng = np.random.default_rng(5)
    base = np.cumsum(rng.normal(size=(N_ATOMS, 3)) * 1.5, axis=0)
    root = tmp_path / "one_group"
    root.mkdir()
    for k in range(40):
        rna_pdb(root / f"m{k}.pdb", base + rng.normal(scale=0.4, size=base.shape), seq=SEQ)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"n_jobs": 1, "selection": {"min_cluster_size": [20]},
                                   "figures": {"enabled": False}}))
    result = CliRunner().invoke(main, ["run", str(root), "-o", str(tmp_path), "-n", "one",
                                       "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    out = tmp_path / "one"
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["result"]["conclusion"] == "no_clusters"
    assert "No clusters" in (out / "report.md").read_text(encoding="utf-8")
    assert (pd.read_csv(out / "clusters.csv")["cluster"] == 0).all()


def test_bias_corrected_cramers_v():
    """Independent labels: the classical V is clearly positive in a small table with many
    columns, the corrected one is close to 0; identical labels give 1 for both."""
    import pandas as pd

    from struct2clust.composition import cramers_v, cramers_v_corrected

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 3, 60), rng.integers(0, 8, 60)
    t = pd.crosstab(a, b)
    assert cramers_v(t) > 0.2 and cramers_v_corrected(t) < cramers_v(t) - 0.1
    same = pd.crosstab(np.repeat([0, 1, 2], 20), np.repeat([0, 1, 2], 20))
    assert cramers_v(same) == pytest.approx(1) and cramers_v_corrected(same) == pytest.approx(1)
