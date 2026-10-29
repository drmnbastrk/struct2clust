"""Command-line interface: ``struct2clust run``, ``compare``, ``composition`` and ``config``."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import click
import yaml
from pydantic import ValidationError

from struct2clust import __version__
from struct2clust.config import Config, default_config_yaml, load_config


def _listify(value) -> list:
    return [value] if isinstance(value, (str, Path)) else list(value)


def _setup_logging(verbose: bool = False) -> None:
    """Struct2Clust messages at INFO (DEBUG with -v); other libraries only at WARNING."""
    import warnings

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("struct2clust").setLevel(logging.DEBUG if verbose else logging.INFO)
    if not verbose:
        warnings.filterwarnings("ignore", module="MDAnalysis")
        warnings.filterwarnings("ignore", module="umap")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_config(
    config_path: Optional[Path],
    *,
    input_type: Optional[str] = None,
    metric: Optional[str] = None,
    atoms: Optional[str] = None,
    topology: Sequence[Path] = (),
    trajectory: Sequence[Path] = (),
    stride: Optional[int] = None,
    seed: Optional[int] = None,
    n_jobs: Optional[int] = None,
    refine: Optional[bool] = None,
) -> Config:
    """YAML file (or defaults) with command-line options applied on top.

    Command-line values win over the file. The merged dict is validated once,
    so cross-field rules (e.g. trajectory needs both files) see the final values.
    """
    data = load_config(config_path).model_dump()
    inp, met = data["input"], data["metric"]
    if input_type is not None:
        inp["type"] = input_type
    if topology:
        inp["topology"] = _listify(topology)
    if trajectory:
        inp["trajectory"] = _listify(trajectory)
    if topology and trajectory and input_type is None:
        inp["type"] = "trajectory"
    if stride is not None:
        inp["stride"] = stride
    if metric is not None:
        met["name"] = metric
    if atoms is not None:
        met["atoms"] = atoms
    if seed is not None:
        data["seed"] = seed
    if n_jobs is not None:
        data["n_jobs"] = n_jobs
    if refine is not None:
        data["selection"]["refine"] = refine
    return Config.model_validate(data)


# -h as well as --help; wide enough that command summaries are not cut off
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name="struct2clust")
def main() -> None:
    """Reproducible clustering of biomolecular structural ensembles and MD trajectories.

    Struct2Clust clusters protein and nucleic-acid structures by pairwise RMSD
    (or TM-score) with HDBSCAN and reports only clusters that reappear when the
    data are resampled. Each run ends with a graded conclusion, e.g.
    distinct_states, partition_not_confirmed or insufficient_sampling.

    \b
    Typical use:
      struct2clust run models/ -o results                  # PDB directory or multi-model PDB
      struct2clust run --topology top.psf --trajectory md1.dcd --trajectory md2.dcd -o results
      struct2clust config > my.yaml                        # all settings, then: run -c my.yaml
      struct2clust composition results/models sources.csv  # where cluster members come from
      struct2clust compare results/run_a results/run_b     # agreement of two runs

    \b
    More detail on each command, with all its options and examples:
      struct2clust run --help
      struct2clust compare --help
      struct2clust composition --help
      struct2clust config --help

    Method: docs/METHODS.md; output files: docs/OUTPUTS.md.
    """


@main.command(short_help="Cluster an ensemble of PDB structures or MD trajectories.")
@click.argument("input_path", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), default=Path("results"),
              show_default=True, help="Parent directory for results.")
@click.option("-n", "--name", default=None,
              help="Run name (sub-directory of --output-dir). Default: input name.")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True, path_type=Path),
              help="YAML configuration file (see `struct2clust config`).")
@click.option("--type", "input_type", type=click.Choice(["ensemble", "trajectory"]),
              help="Input type. Default: ensemble (trajectory if --topology/--trajectory given).")
@click.option("--metric", type=click.Choice(["rmsd", "tm"]),
              help="Pairwise metric (default: rmsd; tm is slower and does not separate "
                   "noise well in short chains).")
@click.option("--atoms", type=click.Choice(["heavy-atoms", "backbone", "anchor"]),
              help="Atoms compared by rmsd. heavy-atoms (default): all non-hydrogen "
                   "atoms incl. bases/side chains; backbone: nucleic P,O5',C5',C4',O4',C3',"
                   "O3',C2',O2',C1' / protein N,CA,C,O; anchor: C3' / CA. See docs/ATOMS.md.")
@click.option("--topology", type=click.Path(exists=True, path_type=Path), multiple=True,
              help="Topology file: once (shared by all trajectories) or once per --trajectory, "
                   "in the same order.")
@click.option("--trajectory", type=click.Path(exists=True, path_type=Path), multiple=True,
              help="Trajectory file. Repeat to cluster several trajectories of the same "
                   "molecule together (replicas, runs from different starts).")
@click.option("--stride", type=int, help="Trajectory mode: keep every n-th frame (default: 1).")
@click.option("--refine/--no-refine", default=None,
              help="Split clusters that contain reproducible sub-states at least as distinct "
                   "as the ensemble's states (default: on).")
@click.option("--seed", type=int, help="Random seed (default: 42).")
@click.option("--n-jobs", type=int,
              help="Parallel workers (default: all cores that fit in memory; -1 = all cores). "
                   "Results do not depend on it.")
@click.option("--overwrite", is_flag=True,
              help="Replace the results of an earlier run in the same directory (default: "
                   "refuse, so that files of two runs are never mixed).")
@click.option("--dry-run", is_flag=True, help="Print the resolved configuration and exit.")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def run(input_path, output_dir, name, config_path, input_type, metric, atoms, topology,
        trajectory, stride, refine, seed, n_jobs, overwrite, dry_run, verbose):
    """Cluster the structures in INPUT_PATH, or the frames of MD trajectories.

    INPUT_PATH is a directory of PDB files or one multi-model PDB file (an
    ensemble of at least 10 structures). For trajectories leave it out and give --topology and
    --trajectory. Command-line options override the configuration file;
    `struct2clust config` prints every setting with its default.

    Results go to OUTPUT_DIR/NAME: report.md (conclusion and summary),
    clusters.csv, cluster_summary.csv, run_summary.json, medoids/, figures/ and
    cluster_views/. docs/OUTPUTS.md describes every file.

    \b
    Examples:
      struct2clust run models/ -o results
      struct2clust run ensemble.pdb --atoms backbone --n-jobs 4
      struct2clust run --topology top.psf --trajectory md1.dcd --trajectory md2.dcd --stride 10
      struct2clust run models/ -c my.yaml --dry-run
    """
    _setup_logging(verbose)
    try:
        cfg = build_config(
            config_path, input_type=input_type, metric=metric, atoms=atoms,
            topology=topology, trajectory=trajectory, stride=stride, refine=refine,
            seed=seed, n_jobs=n_jobs,
        )
    except ValidationError as exc:
        raise click.UsageError(str(exc)) from exc

    if cfg.input.type != "trajectory" and input_path is None:
        raise click.UsageError("INPUT_PATH is required unless --topology/--trajectory are given.")

    if dry_run:
        click.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
        return

    # a directory keeps its whole name (e.g. "models.v2"); a file loses its extension
    if input_path is None:
        default = cfg.input.topology[0].stem
    else:
        default = input_path.name if input_path.is_dir() else input_path.stem
    run_name = name or default
    from struct2clust.pipeline import run_pipeline

    try:
        run_pipeline(cfg, input_path, output_dir / run_name, overwrite=overwrite)
    # unusable input or an occupied output directory: a message, not a traceback
    except (ValueError, FileNotFoundError, FileExistsError) as err:
        if verbose:
            raise
        raise click.ClickException(str(err)) from None


@main.command(short_help="Compare the clusterings of two finished runs (ARI, AMI).")
@click.argument("run_a", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("run_b", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), default=None,
              help="Where to write the comparison. Default: RUN_A/compare_<RUN_B name>.")
def compare(run_a, run_b, output_dir):
    """Compare the clusterings of two finished runs on the same structures.

    RUN_A and RUN_B are run directories written by `struct2clust run`. The
    comparison uses the structures they share and reports the adjusted Rand
    index and adjusted mutual information.

    \b
    Example:
      struct2clust compare results/models_rmsd results/models_tm
    """
    _setup_logging()
    from struct2clust.compare import compare_runs

    out = output_dir or run_a / f"compare_{run_b.name}"
    result = compare_runs(run_a, run_b, out)
    click.echo(
        f"{result['n_shared_structures']} shared structures: "
        f"ARI {result['adjusted_rand_index']:.3f}, AMI {result['adjusted_mutual_information']:.3f} "
        f"-> {out}"
    )


@main.command("composition",
              short_help="Show where the members of each cluster come from (method, PDB entry).")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("sources", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(path_type=Path), default=None,
              help="Where to write the result. Default: RUN_DIR/composition_<SOURCES name>.")
@click.option("--label", default="Source", show_default=True,
              help='What the sources are, for the figure, e.g. "Prediction method".')
def composition_cmd(run_dir, sources, output_dir, label):
    """Cluster composition by source.

    RUN_DIR is a run directory written by `struct2clust run`. SOURCES is a CSV
    file with the columns structure and source (e.g. the prediction method or
    PDB entry of each structure). Writes a table, a figure and the agreement
    between clusters and sources (Cramer's V, ARI, AMI).

    \b
    Example:
      struct2clust composition results/models sources.csv --label "Prediction method"
    """
    _setup_logging()
    from struct2clust.composition import composition

    out = output_dir or run_dir / f"composition_{sources.stem}"
    try:
        result = composition(run_dir, sources, out, label)
    except ValueError as err:
        raise click.ClickException(str(err)) from None
    agree = result["clusters"]
    if agree["cramers_v"] is None:
        stats = "fewer than two clusters or sources, no agreement computed"
    else:
        stats = (f"Cramer's V {agree['cramers_v']:.3f} (bias-corrected "
                 f"{agree['cramers_v_corrected']:.3f}), ARI {agree['adjusted_rand_index']:.3f}, "
                 f"AMI {agree['adjusted_mutual_information']:.3f}")
    click.echo(f"{result['n_structures']} structures, {result['n_sources']} sources: "
               f"{stats} -> {out}")


@main.command("config", short_help="Print all settings with their defaults as YAML.")
def config_cmd() -> None:
    """Print the default configuration as YAML; docs/CONFIGURATION.md explains each setting.

    \b
    Example:
      struct2clust config > my.yaml        # edit, then:
      struct2clust run models/ -c my.yaml
    """
    click.echo(default_config_yaml())
