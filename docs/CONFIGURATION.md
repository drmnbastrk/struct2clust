# Configuration

Every setting has a default, so no configuration file is needed. To change
settings, write the defaults to a file, edit it, and pass it with `-c`:

```bash
struct2clust config > my_settings.yaml
struct2clust run structures/ -c my_settings.yaml
```

Only the settings you want to change need to stay in the file. Command-line
options override the file. Unknown keys are rejected, so a misspelled setting
raises an error instead of being ignored silently.

## Top Level

| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | Random seed for subsampling and UMAP. The same seed and input give the same result. |
| `n_jobs` | `null` | Parallel workers. `null`: all usable CPU cores (on Linux, those the job was given), but no more than the available memory holds (Linux, also the job's memory limit, and Windows; each worker needs about 16·N² bytes for N structures, METHODS.md, section 13, "Memory"); `-1`: all CPU cores. Results do not depend on it. CLI: `--n-jobs`. |

## `input`

| Key | Default | Meaning |
|---|---|---|
| `type` | `ensemble` | `ensemble` (PDB directory or multi-model PDB) or `trajectory`. CLI: `--type`. It is set automatically when `--topology` and `--trajectory` are given. |
| `topology` | `[]` | Topology file(s): one shared by all trajectories, or one per trajectory in the same order. CLI: `--topology` (repeatable). |
| `trajectory` | `[]` | Trajectory file(s) of the same molecule. They are clustered together. CLI: `--trajectory` (repeatable). |
| `stride` | `1` | Keep every n-th frame of each trajectory. CLI: `--stride`. |
| `max_frames` | `null` | Use at most this many frames per trajectory, spread evenly over the whole trajectory (after `stride`). Gives every trajectory the same weight when they were saved at different intervals. |
| `selection` | `all` | Trajectories only: MDAnalysis selection applied before Struct2Clust's own filtering, which already drops water, ions, ligands and hydrogens. Use it to restrict the analysis, e.g. `segid A`. |
| `residues` | `null` | Ensembles and trajectories: compare only these residues, by the residue numbers of the input files, e.g. `1-70` or `A:1-70,B:5-30` (ranges inclusive, optionally per chain). Distances, superposition and characterisation then use these residues only. Use it to leave out a flexible terminus or loop that is not of interest ([METHODS.md](METHODS.md), section 10). `null`: all residues. |
| `trajectory_blocks` | `null` | Each trajectory is resampled in this many contiguous blocks, so correlated neighbouring frames stay together. `null`: as many blocks as the trajectory holds independent frames (blocks of about one autocorrelation time), at least 8 and at most 50; these blocks are resampled like `groups` (`group_fraction` of them per subsample, an absent cluster counts as not reproduced). A number fixes the blocks of every trajectory; they are resampled like structures (`subsample_fraction` of them, absent clusters left out) ([METHODS.md](METHODS.md), section 9). |
| `membership_blocks` | `true` | Trajectories with automatic blocks: if the sampling check fails only because a cluster's membership stays correlated for longer than one block, and every trajectory is at least 8 such times long, resample once more in blocks of about one membership autocorrelation time and report that pass ([METHODS.md](METHODS.md), section 9). |
| `groups` | `null` | Ensembles only: CSV file with the columns `structure` (file name without extension, as in `clusters.csv`) and `group`, the source of every structure (PDB entry, prediction run, simulation). Subsamples then keep or drop whole groups, and a cluster none of whose members is in a subsample counts as not reproduced there, so a cluster must reappear from independent groups ([METHODS.md](METHODS.md), section 6, "Dependent structures"). At least 8 groups are needed, and a cluster from fewer than 8 groups is not split in refinement. This reduces spurious stability from the dependence the column describes; it does not remove dependence the groups do not capture (e.g. runs of one tool sharing a template; METHODS.md, section 13), and it costs power. Optional; `null`: every structure is its own group, and results are those of the method without groups. |
| `group_fraction` | `0.5` | With `groups`, and for trajectories with automatic blocks: fraction of the groups in every subsample (selection, stability and separation ranges); replaces `selection.subsample_fraction` and `stability.subsample_fraction`. 0.5 favours fewer false clusters over sensitivity (METHODS.md, section 12, "Resampling by group"); with 0.8 a cluster from a single group can still pass the stability threshold. |

## `metric`

| Key | Default | Meaning |
|---|---|---|
| `name` | `rmsd` | `rmsd` or `tm` (TM-score, used as the distance "1-TM-score"; needs US-align). CLI: `--metric`. |
| `atoms` | `heavy-atoms` | Atoms compared by `rmsd`: `heavy-atoms`, `backbone` or `anchor` (C3′/CA). See [ATOMS.md](ATOMS.md). CLI: `--atoms`. |
| `equivalent_atoms` | `true` | `rmsd` only (ignored for `tm`): chemically equivalent atoms (e.g. OD1/OD2 of Asp, CD1/CD2 of Leu, OP1/OP2 of nucleotides) are paired in the order that gives the lower RMSD, so that their naming in the files does not count as a difference. `false` pairs atoms by name. Groups and method: [METHODS.md](METHODS.md), section 2. |
| `usalign_exe` | `USalign` | US-align executable: a name on `PATH`, or a full path. |
| `dedup_threshold` | `null` | Near-duplicates: structures within this distance of a kept structure (the centre of their group; [METHODS.md](METHODS.md), section 3) are removed before clustering (ensembles only; trajectory frames are all kept). In the units of the distance: Å for RMSD, "1-TM-score" for `tm`. `null` = 0.1 Å for RMSD; 0.001 for `tm`, i.e. TM-score ≥ 0.999. |

## `selection`

Resolution selection ([METHODS.md](METHODS.md), section 6).

| Key | Default | Meaning |
|---|---|---|
| `resampling_risk` | `0.001` | Tolerated probability that any decision of the run (a candidate's clusters all reach `stability.high` or not) differs from the decision infinitely many subsamples would give. Half goes to the top-level selection, half is shared by the refinement selections. It concerns Monte Carlo error only, i.e. the effect of the random seed on the same data. Subsamples are added until a sequential test by betting establishes the decision ([METHODS.md](METHODS.md), section 6, "Monte Carlo precision"). |
| `min_resamples` | `20` | Subsamples before a candidate can be decided |
| `max_resamples` | `1000` | Candidates still undecided after this many subsamples are marked `borderline` and do not qualify (stability not established). Raise it to resolve partitions at the threshold. |
| `subsample_fraction` | `0.8` | Fraction of structures (or trajectory blocks) in each subsample, at least 0.5. With `input.groups`, `input.group_fraction` is used instead. |
| `min_cluster_size` | `null` | Candidate values. `null` = every integer 5-20, then steps of about 10% up to max(10, N/10). Give a single value, e.g. `[25]`, to fix the resolution by hand. |
| `min_samples` | `null` | Candidate values. `null` = {1, 2, 3, 5, 8, `min_cluster_size`}. |
| `refine` | `true` | Hierarchical refinement: apply the selection again inside every cluster and split a cluster when its members are CLUSTERABLE (dip test at the FWER-adjusted level), a sub-partition qualifies under the same rule and Monte Carlo precision as at the top level, and its sub-clusters are at least as distinct as the ensemble's states (effect size), recursively ([METHODS.md](METHODS.md), section 8). Command line: `--refine/--no-refine`. |
| `refine_min_separation_ratio` | `0` | Optional effect-size rule ([METHODS.md](METHODS.md), section 6, "Effect size"): clusters and sub-clusters whose medoids are closer than this multiple of the state scale (median distance of the members of the other clusters to their own medoid) are reported as one state, with their parts labelled in `clusters.csv` (`substructure`). `0` (default): off; the ratio of every pair of clusters is reported in `separation.csv` and nothing is merged. |
| `refine_min_separation` | `null` | Optional absolute floor for the same rule, in distance units (Å for RMSD), e.g. the coordinate error of experimental structures. With RMSD, the report lists clusters whose medoids are closer than 0.5 Å whether or not a floor is set. |

## `stability`

Cluster-wise stability and tiers ([METHODS.md](METHODS.md), section 7).

| Key | Default | Meaning |
|---|---|---|
| `n_resamples` | `200` | Subsamples for the final, descriptive stability estimate (tiers, Jaccard intervals, membership). The resolution decision itself uses `selection.*_resamples`. |
| `subsample_fraction` | `0.8` | Fraction kept per subsample, at least 0.5 (with `input.groups`: `input.group_fraction`) |
| `high` | `0.75` | Mean Jaccard for tier HIGH. It is also the threshold every cluster must reach in resolution selection. |
| `moderate` | `0.60` | Mean Jaccard for tier MODERATE |

The defaults are Hennig's (2007) published thresholds. Changing them changes the
meaning of the tiers, so report any change.

## `clusterability`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Run the tests. If `false`, no verdict is computed and tiers are not limited. |
| `alpha` | `0.05` | Significance level of the dip test |
| `n_neighbors` | `15` | Neighbours in the graph used for the spectral eigengap |
| `max_k` | `10` | Largest number of groups the eigengap can suggest |

## `biology`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Compute RMSF, contacts, torsions and radius of gyration |
| `contact_cutoff` | `4.5` | Heavy-atom distance (Å) that defines a residue-residue contact |
| `contact_min_separation` | `2` | Ignore residue pairs closer than this in sequence (2 skips direct neighbours) |
| `bond_max_distance` | `2.0` | Å. Neighbouring residues are covalently linked, and share the torsions α, ε, ζ, φ and ψ, when C-N or O3′-P is shorter than this (the bonds are about 1.33 and 1.61 Å). Raise it for strained geometry |

## `figures`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Write figures |
| `formats` | `[pdf, png]` | Any of `pdf`, `svg`, `png` |
| `dpi` | `300` | Resolution of PNG files |
| `umap_n_neighbors` | `15` | UMAP neighbourhood size |
| `umap_min_dist` | `0.5` | UMAP minimum distance between embedded points |
| `cluster_views` | `true` | Write `cluster_views/`: all members of each cluster with VMD / PyMOL scripts ([OUTPUTS.md](OUTPUTS.md)) |
| `cluster_view_max_members` | `null` | At most this many members per cluster in `cluster_views/` (closest to the medoid first); `null` = all |

## Example: many trajectories from one file

```yaml
input:
  type: trajectory
  topology: [md/system.psf]
  trajectory:
    - md/run1/traj.dcd
    - md/run2/traj.dcd
    - md/run3/traj.dcd
  stride: 2
metric:
  atoms: backbone
```

```bash
struct2clust run -c runs.yaml -n three_runs
```
