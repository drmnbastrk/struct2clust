import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from struct2clust.cli import build_config, main
from struct2clust.config import Config, load_config


def test_defaults_are_valid():
    cfg = Config()
    assert cfg.metric.name == "rmsd"
    assert cfg.input.type == "ensemble"
    assert cfg.metric.atoms == "heavy-atoms"


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("selection:\n  n_bootstrap: 10\n")  # not a setting
    with pytest.raises(ValidationError):
        load_config(path)


def test_default_config_roundtrip(tmp_path):
    result = CliRunner().invoke(main, ["config"])
    assert result.exit_code == 0
    path = tmp_path / "cfg.yaml"
    path.write_text(result.output)
    assert load_config(path) == Config()


def test_cli_overrides_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"metric": {"name": "tm"}, "seed": 1}))
    cfg = build_config(path, metric="rmsd", seed=7)
    assert cfg.metric.name == "rmsd"
    assert cfg.seed == 7


def test_trajectory_requires_files():
    with pytest.raises(ValidationError):
        build_config(None, input_type="trajectory")


def test_stability_thresholds_ordered():
    with pytest.raises(ValidationError):
        Config.model_validate({"stability": {"high": 0.5, "moderate": 0.6}})


def test_dry_run(tmp_path):
    result = CliRunner().invoke(main, ["run", str(tmp_path), "--metric", "tm", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.output)["metric"]["name"] == "tm"


def test_run_without_input_is_usage_error():
    result = CliRunner().invoke(main, ["run", "--dry-run"])
    assert result.exit_code == 2


def test_trajectory_accepts_one_or_many(tmp_path):
    one = Config.model_validate({"input": {"type": "trajectory", "topology": "t.psf",
                                           "trajectory": "a.dcd"}})
    assert [p.name for p in one.input.trajectory] == ["a.dcd"]
    many = build_config(None, topology=tmp_path / "t.psf",
                        trajectory=[tmp_path / "a.dcd", tmp_path / "b.dcd"])
    assert many.input.type == "trajectory" and len(many.input.trajectory) == 2


def test_topology_count_must_match():
    with pytest.raises(ValidationError):
        Config.model_validate({"input": {"type": "trajectory", "topology": ["a", "b"],
                                         "trajectory": ["x", "y", "z"]}})
    ok = Config.model_validate({"input": {"type": "trajectory", "topology": ["a", "b"],
                                          "trajectory": ["x", "y"]}})
    assert len(ok.input.topology) == 2


def test_resample_range_ordered():
    with pytest.raises(ValidationError):
        Config.model_validate({"selection": {"min_resamples": 100, "max_resamples": 50}})


def test_too_few_structures_is_a_clear_error(tmp_path):
    from helpers import rna_pdb

    rng = np.random.default_rng(0)
    for k in range(5):
        rna_pdb(tmp_path / f"m{k}.pdb", rng.normal(size=(24, 3)) * 5)
    result = CliRunner().invoke(main, ["run", str(tmp_path), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "at least 10 are needed" in result.output and "Traceback" not in result.output


def test_run_name_keeps_dots_of_a_directory(tmp_path):
    from helpers import rna_pdb

    src = tmp_path / "models.v2"
    src.mkdir()
    rng = np.random.default_rng(0)
    for k in range(3):
        rna_pdb(src / f"m{k}.pdb", rng.normal(size=(24, 3)) * 5)
    result = CliRunner().invoke(main, ["run", str(src), "-o", str(tmp_path / "out")])
    assert "at least 10 are needed" in result.output      # stops early, after naming the run
    assert (tmp_path / "out" / "models.v2").is_dir()


def test_equivalent_atoms_are_matched_by_default():
    assert Config().metric.equivalent_atoms
    off = Config.model_validate({"metric": {"equivalent_atoms": False}})
    assert not off.metric.equivalent_atoms


def test_unreadable_coordinates_name_the_file_and_line(tmp_path):
    good = ("ATOM      1  P     G A   1       1.000   2.000   3.000  1.00  0.00           P\n"
            "END\n")
    (tmp_path / "a.pdb").write_text(good)
    (tmp_path / "b.pdb").write_text("REMARK\nATOM      1  P     G A   1     garbage\n")
    result = CliRunner().invoke(main, ["run", str(tmp_path), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "b.pdb, line 2" in result.output


def test_workers_fit_into_the_available_memory():
    """n_jobs None: no more workers than the memory holds beside the main process (at
    least one); an explicit number is kept (up to the cores)."""
    from struct2clust.resources import cpu_count, peak_memory, worker_count

    n, cpus = 1000, cpu_count()
    assert peak_memory(n, 2) == 12 * 8 * n * n
    assert worker_count(None, n, None) == cpus
    assert worker_count(None, n, peak_memory(n, 1)) == 1
    assert worker_count(None, n, peak_memory(n, 0)) == 1
    assert worker_count(None, n, 10 ** 15) == cpus
    assert worker_count(1, n, 0) == 1 and worker_count(-1, n, 0) == cpus
