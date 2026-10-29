"""Run configuration.

Every configurable value lives here, with its default. Unknown keys are
rejected (``extra="forbid"``) so a typo in a YAML file fails loudly instead of
being silently ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

InputType = Literal["ensemble", "trajectory"]
MetricName = Literal["rmsd", "tm"]
AtomSelection = Literal["heavy-atoms", "backbone", "anchor"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InputConfig(_Strict):
    type: InputType = "ensemble"
    """ensemble: a directory of PDB files (or one multi-model PDB).
    trajectory: one topology plus one or more trajectory files."""

    topology: list[Path] = Field(default_factory=list)
    """One topology shared by all trajectories, or one per trajectory (same order).
    Topologies may differ (atom order, hydrogens, force-field names) as long as they
    describe the same molecule."""
    trajectory: list[Path] = Field(default_factory=list)
    """Trajectory files of the same molecule (e.g. replicas, or runs from different
    starting structures). They are clustered together, so every trajectory is
    described by the same set of states."""
    stride: int = Field(default=1, ge=1)
    """Trajectory mode: keep every n-th frame."""
    max_frames: Optional[int] = Field(default=None, ge=4)
    """Trajectory mode: use at most this many frames per trajectory, spread evenly over
    the whole trajectory. Useful for long runs and for trajectories saved at different
    intervals (each contributes the same number of frames)."""
    selection: str = "all"
    """Trajectory mode: MDAnalysis selection applied before Struct2Clust's own filtering
    (which already drops water, ions, ligands and hydrogens). Use it to restrict the
    analysis, e.g. "segid A". The MDAnalysis keywords "protein"/"nucleic" do not
    recognise force-field terminal residue names such as G5/C3, so they are not the default."""
    residues: Optional[str] = None
    """Ensembles and trajectories: compare only these residues, e.g. "1-70" or
    "A:1-70,B:5-30" (residue numbers of the input files, ranges inclusive, optionally per
    chain). Everything (distances, superposition, characterisation) then uses these
    residues only. Use it to leave out a flexible terminus or loop that is not of
    interest: RMSD squares deviations, so a few strongly displaced residues can decide
    the clusters (docs/METHODS.md, section 10). None = all residues."""
    trajectory_blocks: Optional[int] = Field(default=None, ge=5)
    """Trajectory mode: each trajectory is resampled in this many contiguous blocks,
    so time-correlated neighbouring frames are kept or dropped together. None = as many
    blocks as the trajectory holds independent frames (blocks of about one
    autocorrelation time), at least 8 and at most 50, resampled like input.groups;
    a number keeps subsample_fraction and ignores clusters absent from a subsample
    (docs/METHODS.md, section 9)."""
    membership_blocks: bool = True
    """Trajectory mode, automatic blocks: if the sampling check fails only because a
    cluster's membership stays correlated for longer than one block, and every trajectory
    is at least 8 such times long, resample once more in blocks of about one membership
    autocorrelation time and report that run (docs/METHODS.md, section 9). False = one
    pass with the geometric blocks."""
    groups: Optional[Path] = None
    """Ensemble mode: CSV file with the columns structure, group, naming the source of
    every structure (PDB entry, prediction run, simulation). Structures of one group
    are not independent; with groups, subsamples keep or drop whole groups
    (group_fraction of them), and a cluster none of whose members is in a subsample
    counts as not reproduced there (Jaccard 0), so a cluster must reappear from
    independent groups. None = every structure is its own group (no change)."""
    group_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    """With groups, and for trajectories with automatic blocks: fraction of the groups
    (blocks) in every subsample, for the selection, the stability and the separation
    ranges (half, as in Hennig's subsetting)."""

    @field_validator("residues")
    @classmethod
    def _residue_ranges(cls, v):
        if v is not None:
            from struct2clust.structures import parse_residue_ranges
            parse_residue_ranges(v)      # raises ValueError with the offending item
        return v

    @field_validator("topology", "trajectory", mode="before")
    @classmethod
    def _as_list(cls, v):
        if v is None:
            return []
        return [v] if isinstance(v, (str, Path)) else list(v)


class MetricConfig(_Strict):
    name: MetricName = "rmsd"
    atoms: AtomSelection = "heavy-atoms"
    """Atom subset for rmsd (tm always uses US-align's CA / C3').

    heavy-atoms: every non-hydrogen atom of every protein / nucleic-acid residue
        (nucleic: phosphate, sugar, 2'-OH, base; protein: backbone + side chains).
    backbone: nucleic P, O5', C5', C4', O4', C3', O3', C2', O2', C1';
        protein N, CA, C, O.
    anchor: C3' per nucleotide, CA per amino acid.
    Full per-residue lists: docs/ATOMS.md.
    """
    equivalent_atoms: bool = True
    """rmsd only (tm ignores it): pair chemically equivalent atoms (e.g. OD1/OD2 of
    Asp, CD1/CD2 of Leu, OP1/OP2 of nucleotides) in the order that gives the lower
    RMSD, so that their naming in the files does not count as a conformational
    difference. False = atoms paired by name. Groups: docs/METHODS.md, section 2."""
    usalign_exe: str = "USalign"
    dedup_threshold: Optional[float] = Field(default=None, ge=0.0)
    """Near-duplicates: structures within this distance of a kept structure (the centre
    of their group, Butina 1999) are removed before clustering (ensembles only;
    trajectory frames are all kept), in the units of the distance (rmsd: A; tm:
    1 - TM-score). None = metric default (rmsd: 0.1 A; tm: 0.001, i.e. TM-score >= 0.999)."""


class SelectionConfig(_Strict):
    """Resolution selection: finest partition whose clusters all reach stability.high.

    Every candidate partition is checked with subsamples added sequentially until
    its decision (all clusters >= stability.high or not) is established by a
    sequential test by betting at the given resampling risk, between min_resamples
    and max_resamples subsamples; the final clustering is then described with
    stability.n_resamples subsamples.
    """

    resampling_risk: float = Field(default=0.001, gt=0.0, lt=0.5)
    """Tolerated probability that any decision of the run differs from the one that
    infinitely many subsamples would give (Monte Carlo error only: the effect of the
    random seed on the same data). Half goes to the top-level selection, half is
    shared by the refinement selections (docs/METHODS.md, section 6)."""
    min_resamples: int = Field(default=20, ge=10)
    max_resamples: int = Field(default=1000, ge=20)
    """Candidates still undecided after this many subsamples are marked borderline and do
    not qualify (their stability is not established)."""
    subsample_fraction: float = Field(default=0.8, ge=0.5, lt=1.0)
    min_cluster_size: Optional[list[int]] = None
    """Candidate values. None = every integer 5-20, then ~10% steps up to max(10, N/10)."""
    min_samples: Optional[list[int]] = None
    """Candidate values. None = {1, 2, 3, 5, 8, min_cluster_size} (capped at mcs)."""
    refine: bool = True
    """Apply the selection again inside every cluster; split clusters that contain
    reproducible sub-states (recursively)."""
    refine_min_separation_ratio: float = Field(default=0.0, ge=0.0)
    """Optional effect-size rule, at the top level and in refinement: clusters whose
    medoids are closer than this multiple of the state scale (median distance of the
    members of the other top-level clusters to their own medoid) are reported as one
    state, with their parts labelled as sub-structure. 0 (default) = off: the ratio of
    every pair of clusters is reported (separation.csv) but no clusters are merged."""
    refine_min_separation: Optional[float] = Field(default=None, ge=0.0)
    """Optional absolute floor for the same rule, in distance units (A for rmsd), e.g.
    the coordinate error of experimental structures. None = no floor."""

    @model_validator(mode="after")
    def _resample_range(self) -> "SelectionConfig":
        if self.min_resamples > self.max_resamples:
            raise ValueError("selection.min_resamples must not exceed selection.max_resamples")
        return self


class StabilityConfig(_Strict):
    """Cluster-wise stability by subsampling without replacement (Hennig 2007)."""

    n_resamples: int = Field(default=200, ge=20)
    subsample_fraction: float = Field(default=0.8, ge=0.5, lt=1.0)
    high: float = Field(default=0.75, gt=0.0, le=1.0)
    moderate: float = Field(default=0.60, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "StabilityConfig":
        if self.moderate >= self.high:
            raise ValueError("stability.moderate must be smaller than stability.high")
        return self


class ClusterabilityConfig(_Strict):
    enabled: bool = True
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    n_neighbors: int = Field(default=15, ge=5)
    max_k: int = Field(default=10, ge=2)


class BiologyConfig(_Strict):
    enabled: bool = True
    contact_cutoff: float = Field(default=4.5, gt=0.0)
    """Angstrom. Two residues are in contact when any of their heavy atoms are this close."""
    contact_min_separation: int = Field(default=2, ge=1)
    """Ignore residue pairs closer than this in sequence (2 skips direct neighbours)."""
    bond_max_distance: float = Field(default=2.0, gt=0.0)
    """Angstrom. Neighbouring residues are covalently linked, and share the torsions
    alpha, epsilon, zeta, phi and psi, when C-N or O3'-P is shorter than this
    (C-N about 1.33 A, O3'-P about 1.61 A). Raise it for strained geometry."""


class FiguresConfig(_Strict):
    enabled: bool = True
    formats: list[Literal["pdf", "svg", "png"]] = Field(default_factory=lambda: ["pdf", "png"])
    dpi: int = Field(default=300, ge=72)
    umap_n_neighbors: int = Field(default=15, ge=2)
    umap_min_dist: float = Field(default=0.5, ge=0.0)
    cluster_views: bool = True
    """Write cluster_views/: per cluster, all members superposed on the medoid in one
    multi-model PDB, with VMD and PyMOL scripts that show them as cartoons in the
    cluster colour, fading from the medoid to the member farthest from it."""
    cluster_view_max_members: Optional[int] = Field(default=None, ge=1)
    """Show at most this many members per cluster (the ones closest to the medoid).
    None = all members."""


class Config(_Strict):
    seed: int = 42
    n_jobs: Optional[int] = None
    """Parallel workers. None = all CPU cores, but no more than the available memory holds
    (each worker needs about 16 N^2 bytes for N structures; docs/METHODS.md, section 13,
    "Memory"); -1 = all CPU cores. Results do not depend on it."""
    input: InputConfig = Field(default_factory=InputConfig)
    metric: MetricConfig = Field(default_factory=MetricConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    clusterability: ClusterabilityConfig = Field(default_factory=ClusterabilityConfig)
    biology: BiologyConfig = Field(default_factory=BiologyConfig)
    figures: FiguresConfig = Field(default_factory=FiguresConfig)

    @model_validator(mode="after")
    def _consistent(self) -> "Config":
        inp = self.input
        if inp.type == "trajectory":
            if not inp.topology or not inp.trajectory:
                raise ValueError("input.type 'trajectory' requires input.topology and at least "
                                 "one input.trajectory")
            if len(inp.topology) not in (1, len(inp.trajectory)):
                raise ValueError(f"{len(inp.topology)} topologies for {len(inp.trajectory)} "
                                 "trajectories: give one shared topology or one per trajectory")
            if inp.groups is not None:
                raise ValueError("input.groups applies to ensembles; trajectories are "
                                 "resampled in blocks (input.trajectory_blocks)")
        return self

    @property
    def grouped(self) -> bool:
        """Whether structures are resampled by group (input.groups)."""
        return self.input.groups is not None

    @property
    def automatic_blocks(self) -> bool:
        """Trajectories resampled in blocks of about one autocorrelation time."""
        return self.input.type == "trajectory" and self.input.trajectory_blocks is None

    @property
    def by_unit(self) -> bool:
        """Whether the resampling units are treated as independent groups: input.groups,
        or the automatic trajectory blocks. Subsamples then keep group_fraction of the
        units, a cluster absent from a subsample counts as not reproduced, and a cluster
        from fewer than 8 units is not refined (docs/METHODS.md, sections 6 and 9)."""
        return self.grouped or self.automatic_blocks

    @property
    def selection_fraction(self) -> float:
        return self.input.group_fraction if self.by_unit else self.selection.subsample_fraction

    @property
    def stability_fraction(self) -> float:
        return self.input.group_fraction if self.by_unit else self.stability.subsample_fraction


def load_config(path: Optional[Path]) -> Config:
    """Load a YAML config file; ``None`` returns the defaults."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


def default_config_yaml() -> str:
    """The full default configuration as YAML (for ``struct2clust config``)."""
    data = Config().model_dump(mode="json")
    return yaml.safe_dump(data, sort_keys=False)
