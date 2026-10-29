# Outputs

Every run writes to `<output-dir>/<name>/`. The name defaults to the input
directory, or to the topology file name for trajectories.

```
<name>/
  report.md                  overall conclusion and short summary
  clusters.csv               one row per structure / frame
  cluster_summary.csv        one row per cluster
  medoid_difference.csv      where the cluster medoids differ, per residue
  separation.csv             effect size of every pair of clusters
  populations.csv            trajectories: populations with intervals, overall and per part
  population_differences.csv trajectories: population differences between the parts
  duplicates.csv             near-duplicates removed before clustering (if any)
  selection.csv              every candidate resolution that was assessed
  run_summary.json           everything needed to report or reproduce the run
  distance_matrix.npy        distance matrix of the clustered structures
  distance_matrix_labels.txt row / column labels of the matrix
  medoids/                   one representative structure per cluster
  cluster_views/             every member of every cluster, with VMD / PyMOL scripts
  biology/                   descriptive tables (RMSF, contacts, torsions)
  figures/                   figures (PDF + PNG by default)
  .cache/                    cached distance matrix, reused by re-runs
```

A run does not write into a directory that already holds results: it stops with
a message, so that files of two runs are never mixed (a run writes only the files
of its own outcome, and files of an earlier run, such as the medoid of a cluster
the new run did not find, would remain). Choose another name, or pass
`--overwrite`, which first removes the files and directories listed above; the
distance cache `.cache/` is kept, and other files are left in place and named in
the log.

Clusters are numbered 1, 2, … and noise is cluster `0`. Top-level clusters are
ordered from largest to smallest; the sub-clusters of a split cluster follow one
another, again from largest to smallest. The column `top_level` gives the
top-level cluster each cluster comes from ([METHODS.md](METHODS.md), section 8):
clusters with the same `top_level` are sub-states of one top-level cluster,
separated by refinement. Without any split, `top_level` equals the cluster
number.

## `clusters.csv`

| Column | Meaning |
|---|---|
| `structure` | Member name: the file name, `<file>_m<k>` for model k of a multi-model file, or `[<trajectory>:]frame_<i>` for trajectories |
| `file` | Source file (ensembles only) |
| `replica` | Trajectory label (trajectories only; e.g. `SET1`) |
| `frame` | Frame index in its trajectory (trajectories only; counts the frames of the original trajectory, before `stride`) |
| `cluster` | Cluster number; `0` = noise |
| `top_level` | Top-level cluster the structure belongs to; `0` = noise at the top level. A structure that belongs to a split top-level cluster but to none of its sub-clusters has `cluster` `0` and its `top_level`. |
| `substructure` | Only if present (optional effect-size rule or absolute floor): the reproducible sub-group (1, 2, …) a structure belongs to when its cluster is made of sub-groups that are closer than the separation threshold and are therefore reported as one state ([METHODS.md](METHODS.md), section 6, "Effect size"); `0` otherwise. |
| `membership` | Fraction of subsamples containing this structure in which it stayed in its cluster. For a sub-cluster: the membership in its parent times the membership within the parent. For noise, the value for the closest cluster. |
| `closest_cluster` | Own cluster, or for noise the cluster it joins most often in subsamples (`0` if none) |
| `is_medoid` | `True` for the medoid of its cluster |
| `distance_to_medoid` | Distance to its cluster's medoid, in the metric's unit (empty for noise) |
| `radius_of_gyration` | Heavy-atom radius of gyration (Å) |

## `cluster_summary.csv`

| Column | Meaning |
|---|---|
| `cluster` | Cluster number |
| `top_level` | Top-level cluster the cluster comes from |
| `size`, `fraction` | Number and fraction of clustered structures |
| `membership_autocorrelation_time` | Trajectories only: integrated autocorrelation time (frames) of the cluster's membership, the largest over the trajectories ([METHODS.md](METHODS.md), section 9, "Sampling check") |
| `fraction_low`, `fraction_high` | Trajectories only: approximate 95% interval of `fraction` from block-bootstrap samples, the partition held fixed (the 0.05% and 99.95% quantiles: nominal 95% intervals were too narrow; these covered the true population in 92-99.5% of simulated samples; [METHODS.md](METHODS.md), section 9) |
| `selection_decision` | `established`: the resolution selection established, at the stated Monte Carlo risk, that the cluster's mean Jaccard is at least `stability.high`; `not_established`: no reproducible partition was found ([METHODS.md](METHODS.md), sections 6 and 7) |
| `selection_estimate` | Mean Jaccard from the selection's own subsamples; the sequential test stops once its decision is established, so this estimate can rest on few subsamples |
| `independent_stability_estimate` | Mean Jaccard from `stability.n_resamples` fresh subsamples (Hennig 2007). For a sub-cluster, measured within its parent |
| `independent_stability_low`, `independent_stability_high` | 2.5% and 97.5% quantiles of the single-subsample Jaccard values of that estimate (their spread, not a confidence interval of the mean) |
| `stability_tier` | Hennig's band of `independent_stability_estimate` (`HIGH`, `MODERATE`, `LOW`), capped by the clusterability verdict; it describes the estimate and is not a second decision |
| `dissolved_fraction` | Fraction of subsamples in which the cluster dissolved (Jaccard ≤ 0.5) |
| `medoid` | Name of the medoid structure |
| `median_intra_distance` | Median pairwise distance within the cluster |
| `rg_mean`, `rg_sd` | Radius of gyration (Å), mean and standard deviation |
| `rmsf_mean` | Mean per-residue RMSF within the cluster (Å) |
| `population_<replica>` | Several trajectories only: fraction of that trajectory's frames in the cluster |

## `populations.csv`

Trajectories only ([METHODS.md](METHODS.md), section 9).

| Column | Meaning |
|---|---|
| `part` | `all`, a trajectory, or `first_half` / `second_half` of a single trajectory |
| `cluster` | Cluster number |
| `n_frames` | Frames of the part |
| `population` | Fraction of the part's frames in the cluster |
| `low`, `high` | Approximate 95% interval of `population` from block-bootstrap samples of the part, the partition held fixed (as `fraction_low`, `fraction_high` in `cluster_summary.csv`; [METHODS.md](METHODS.md), section 9) |

## `population_differences.csv`

Trajectories only: every pair of parts and every cluster.

| Column | Meaning |
|---|---|
| `part_a`, `part_b` | The two parts compared |
| `cluster` | Cluster number |
| `population_a`, `population_b`, `difference` | Populations in the two parts and their difference (a − b) |
| `difference_low`, `difference_high` | Bootstrap interval of the difference, Bonferroni-adjusted over clusters and pairs to a nominal 95% family-wise level (approximate: halves of simulated stationary trajectories were flagged in 8-20% of samples; [METHODS.md](METHODS.md), section 9) |
| `differs` | The interval excludes zero: the populations differ by more than their approximate sampling uncertainty. A flag, not a test at the 5% level |

## `separation.csv`

One row per pair of final clusters ([METHODS.md](METHODS.md), section 6,
"Effect size"). Written when there are at least two clusters, whether or not
the optional effect-size rule is used; it describes the clusters and does not
change them.

| Column | Meaning |
|---|---|
| `a`, `b` | The two clusters |
| `medoid_distance` | Distance between their medoids |
| `scatter_a`, `scatter_b` | Median distance of each cluster's members to its own medoid |
| `state_scale` | Median distance of the members of all other clusters to their own medoid, pooled; empty with two clusters |
| `ratio_to_state_scale` | `medoid_distance` / `state_scale`; below 1, the medoids are closer than the ensemble's states are wide |
| `ratio_q025`, `ratio_q975` | 2.5% and 97.5% quantiles of the ratio over 200 subsamples (spread between subsamples of the same data, not a confidence interval) |

## `medoid_difference.csv`

One row per pair of clusters and residue ([METHODS.md](METHODS.md), section
10, "Where the clusters differ"). Written when there are at least two clusters.

| Column | Meaning |
|---|---|
| `cluster_a`, `cluster_b` | The pair of clusters |
| `residue` | Residue position (1-based); `chain`, `resid` and `resname` give the original numbering |
| `share` | The residue's part of the squared RMSD between the two medoids, superposed as for the clustering distance (least squares on the clustering atoms for RMSD, TM-score superposition on C3′/CA for TM; the shares of a pair add up to 1) |
| `rmsd` | RMSD of the residue's own atoms between the two superposed medoids (Å) |

## `duplicates.csv`

Written only when near-duplicates were removed ([METHODS.md](METHODS.md),
section 3; never for trajectories). These structures take no part in the clustering, the figures or any
statistic; the file lists them for completeness.

| Column | Meaning |
|---|---|
| `structure` | The removed structure |
| `duplicate_of` | The centre of its group of copies, which was kept and clustered; the removed structure is within the threshold of this centre (two copies of one centre can be up to twice the threshold apart) |
| `distance` | Their distance, in the metric's unit |

## `selection.csv`

This file has one row per candidate HDBSCAN setting ([METHODS.md](METHODS.md),
section 6).

| Column | Meaning |
|---|---|
| `min_cluster_size`, `min_samples`, `method` | HDBSCAN setting (`method` is always `eom`) |
| `n_clusters`, `noise` | Clusters and noise fraction on the full data |
| `min_jaccard` | Mean Jaccard of the least stable cluster |
| `evaluated` | Whether the candidate was assessed. Candidates are assessed from the most clusters to the fewest; once one qualifies, coarser candidates cannot change the result and are not assessed |
| `min_jaccard_se` | Standard error of `min_jaccard` (descriptive; the decision does not use it) |
| `mean_jaccard` | Mean Jaccard averaged over clusters |
| `n_subsamples` | Subsamples used: added until the sequential test established the decision (`selection.min_resamples` to `max_resamples`) |
| `qualifies` | Every cluster is established to reach the stability threshold (`stability.high`) at the resampling risk |
| `borderline` | No decision after `max_resamples`: the true value is too close to the threshold to be resolved; the candidate does not qualify |
| `selected` | The chosen setting |

Because subsampling stops as soon as a decision is clear, `min_jaccard` of
candidates decided early is less precise (see `min_jaccard_se`); the final,
precise estimate of the selected clusters is in `cluster_summary.csv`.

## `run_summary.json`

| Key | Content |
|---|---|
| `input` | Input type, numbers of structures read / duplicates removed / used, the duplicate pairs, molecule type, number of residues, `resampling_groups` (with `input.groups`: the file, number of groups, group fraction and the rule for absent clusters; otherwise `null`), and per trajectory (`replicas`): frames, `autocorrelation_time` (frames) and its `autocorrelation_time_rel_error`, `effective_frames` (independent frames), `n_blocks` and `blocks_rule` (`automatic` or `fixed`), `block_length` (frames) and `block_shorter_than_autocorrelation_time` ([METHODS.md](METHODS.md), section 9) |
| `distance` | Metric, axis label and unit, atom subset, whether chemically equivalent atoms were matched (`equivalent_atoms_matched`; RMSD only, [METHODS.md](METHODS.md), section 2), and `compared_atoms`: the atoms actually compared, per residue type |
| `clusterability` | Verdict, dip statistic and p-value, spectral k |
| `selection` | Selection rule, whether a fully reproducible partition was found (`reproducible_partition_found`), selected parameters, weakest-cluster Jaccard and its standard error, number of candidates, the resampling risk, whether a candidate with more clusters was left out because it was borderline (`borderline`), how many candidates were borderline, and whether qualifying settings tied with the selected one on clusters and noise give different partitions (`tie_partitions_differ`; the choice among them is not covered by the Monte Carlo guarantee, [METHODS.md](METHODS.md), section 6) |
| `effect_size` | The optional effect-size rule (`off` by default), the number of clusters the resolution selection gave, `pairs` (the rows of `separation.csv`, with `ratio_subsampling_range`), every top-level merge (the resulting cluster, how many selected clusters it joins, their medoid distances, state scale, ratio and `ratio_subsampling_range`, the 2.5% and 97.5% quantiles of the ratio over 200 subsamples), `no_reference` (with exactly two top-level clusters: the clusters in each, their medoid distance and the median distance of each one's members to its medoid, since the rule has no state scale for this pair) and `below_coordinate_error` (RMSD only: pairs of clusters whose medoids are closer than 0.5 Å) |
| `medoid_difference` | The fewest residues that carry at least half of the squared deviation between the cluster medoids, averaged over the pairs of clusters (`residues`, in the original numbering, their number and combined `share`), and the same for every pair (`pairs`) |
| `refinement` | Whether refinement was on, the number of top-level clusters, and one entry per refinement attempt: the group of structures examined, given by its top-level cluster (`top_level`), its depth below it (`level`, 0 = the top-level cluster itself) and the clusters it contains (`clusters`); its size, with `input.groups` the number of groups among its members (`n_groups`; below 8 the cluster is not split), its share of the Monte Carlo risk (`resampling_risk`; [METHODS.md](METHODS.md), section 6), the significance level of its dip test (`alpha`, FWER-adjusted), its clusterability verdict, whether it was split, into how many sub-clusters, the weakest sub-cluster Jaccard of the selection with its standard error, subsamples used, `borderline` flag (a borderline sub-partition is not split) and `tie_partitions_differ`, the candidate sub-clusters of the selected sub-partition (numbered 1, 2, … by size) with their sizes (`subcluster_sizes`) and mean Jaccard (`subcluster_jaccard`), the state scale, the separation threshold, the medoid distance of every pair of candidates, its ratio to the state scale and the ratio's `ratio_subsampling_range` (`medoid_separation`), whether all candidates form one state (`substructure`) or some of them one cluster (`merged_subclusters`: that cluster and its candidates), and, if not split, the reason |
| `result` | `conclusion` (`distinct_states`, `partition_not_confirmed`, `no_group_structure`, `no_reproducible_partition`, `no_clusters`, `insufficient_clusters` or `insufficient_sampling`; [METHODS.md](METHODS.md), section 7), `conclusion_before_effect_size` (the same rule applied to the partition of the resolution selection, before the effect-size rule and refinement; a diagnostic, not the conclusion), number of clusters, noise fraction, top-level noise fraction, count of each stability tier (`stability_tiers`), and for trajectories `populations`: the nominal level of the bootstrap quantiles (`interval_level`, 0.999; the intervals are approximate 95% intervals, above), the family-wise level of the comparison of the parts (`flag_level`, 0.95), the blocks per trajectory (`blocks`), the parts compared (`trajectories`, or `halves` of a single trajectory) with their frames and blocks, the largest population difference between parts, the clusters whose populations differ between parts by more than their sampling uncertainty (`clusters_differing`) and `parts_agree`; for trajectories `sampling_check` (`sufficient`, `reasons`, `clusters_longer_than_block`, `clusters_without_transitions`) and `conclusion_if_sampling_sufficient` (the conclusion with independent blocks when the check failed, a diagnostic; otherwise `null`); with `input.membership_blocks`, `first_pass` (the pass with geometric blocks that a second pass replaced: its conclusion, clusters, clusterability verdict, blocks and sampling check; otherwise `null`) |
| `figures.umap_trustworthiness` | Neighbourhood preservation of the UMAP map (0-1) |
| `config` | The complete configuration used |
| `software`, `seed`, `created` | Versions of Struct2Clust and its dependencies, random seed, timestamp. `software.struct2clust_source_sha256` is the SHA-256 of the package's Python files (names and contents, in name order); it identifies the code that wrote the file even when the version number is unchanged |
| `struct2clust_version` | Version of Struct2Clust that wrote the file (the same value as `software.struct2clust`), first key of the file |

## `medoids/`

- **Structures.** `C1.pdb`, `C2.pdb`, … are the medoid of each cluster.
  - They contain all heavy atoms of the molecule.
  - They are superposed onto `C1.pdb` as for the clustering distance (least
    squares on the clustering atoms for RMSD, TM-score superposition on C3′/CA
    for TM), so they can be loaded together directly.
- **Viewer scripts.**
  - `view_medoids.pml`: run `pymol view_medoids.pml` inside `medoids/`.
  - `view_medoids.tcl`: run `vmd -e view_medoids.tcl` inside `medoids/`.
  - Both load all medoids as cartoons, coloured like the figures.

## `cluster_views/`

All members of each cluster, to see how tight or spread out it is.

- **Structures.** `C1_members.pdb`, `C2_members.pdb`, … hold every member of the
  cluster as one multi-model PDB file. Model 1 is the medoid; the others follow
  in order of increasing distance to the medoid (a `REMARK` line before each
  model names the structure). Members are superposed on their medoid, and the
  medoids on the C1 medoid, in the same way as in `medoids/`, so all clusters
  share one frame.
- **Viewer scripts**, one per cluster; run them inside `cluster_views/`:
  - `C1.tcl`: `vmd -e C1.tcl`
  - `C1.py`: `pymol C1.py` (or `run C1.py` inside PyMOL)
  - Every member is drawn as a cartoon (NewCartoon in VMD) in the cluster's
    colour (the same as in the figures). The medoid is opaque; opacity falls
    linearly with rank to 0.1 for the member farthest from the medoid (in 10
    steps). A tight cluster looks like one solid structure, a spread-out one
    like a solid core inside a faint cloud.
  - Several clusters can be loaded into one session.
- **`members.csv`**: cluster, model number, structure, distance to the medoid
  and opacity of every model.
- Noise structures are not included. `figures.cluster_view_max_members` limits
  the members per cluster (the closest to the medoid are kept); viewing
  hundreds of transparent cartoons can be slow. `figures.cluster_views: false`
  turns the folder off.

## `biology/`

All values are descriptive ([METHODS.md](METHODS.md), section 10). Residues are
numbered by position (`residue`, 1-based). The original `chain`, `resid` and
`resname` are given alongside.

| File | Columns |
|---|---|
| `rmsf.csv` | `cluster`, `residue`, `rmsf` (Å) |
| `contacts.csv` | `cluster`, `residue_i`, `residue_j`, `frequency`: fraction of the cluster's structures with the contact; only pairs with a non-zero frequency |
| `torsions.csv` | `cluster`, `residue`, `torsion`, `circular_mean` (°), `circular_sd` (°), `n` |
| `torsion_separation.csv` | `residue`, `torsion`, `separation` (0-1, how strongly the torsion separates the clusters), `max_mean_difference` (°, largest difference between cluster means); sorted by `separation` |

## `figures/`

The default formats are PDF (vector, fonts embedded) and PNG (300 dpi); set
them with `figures.formats`. Widths are 89 mm for one
column, 183 mm for two.

| Figure | Content |
|---|---|
| `overview` | (a) distance matrix ordered by cluster; (b) UMAP map with numbered clusters and medoids; (c) resolution selection: weakest-cluster Jaccard of every candidate against its number of clusters, with the selected one circled; (d) stability of every cluster with its tier |
| `distances` | Distances within each cluster and between clusters |
| `rmsf` | Per-residue RMSF of every cluster (heatmap) |
| `contacts` | Contact-frequency map of every cluster |
| `torsions` | The six torsions that best separate the clusters, one panel each |
| `timeline` | Trajectories (up to four): the cluster of every frame over time |
| `populations` | Several trajectories: fraction of each trajectory's frames in each cluster |

## `struct2clust composition RUN SOURCES`

Where the members of each cluster come from ([METHODS.md](METHODS.md), section
10, "Composition by source"). `SOURCES` is a CSV file with the columns
`structure` (names as in `clusters.csv`) and `source` (any label: prediction
method, PDB entry, ligand, simulation, …). The file is optional input for this
command only; `struct2clust run` does not use it. Not every structure needs a
source: structures without one are shown as `unknown` and left out of the
agreement values. Near-duplicates removed before clustering (`duplicates.csv`)
are not counted. By default the output goes to
`RUN/composition_<SOURCES name>/`; `--label` names the sources in the figure.

| File | Content |
|---|---|
| `counts.csv` | Structures per cluster (rows; noise last) and source (columns) |
| `fractions.csv` | The same as fractions of each cluster |
| `by_source.csv` | Fraction of each source's structures in each cluster |
| `composition.json` | Per cluster: size, top-level cluster, largest source and its share (`purity`), number of sources; Cramér's V and its bias-corrected form (`cramers_v_corrected`, Bergsma 2013), the share of table cells with an expected count below 5 (`sparse_cells`), adjusted Rand index and adjusted mutual information between clusters and sources (noise and `unknown` excluded; empty when fewer than two clusters or sources remain), and the same for the top-level clusters when refinement split any |
| `composition.pdf/.png` | (a) The members of each cluster by source; (b) the clusters of each source's structures, in the cluster colours of the other figures |

## `struct2clust compare A B`

This command compares two finished runs over the structures they share. By
default it writes to `A/compare_<B>/`.

| File | Content |
|---|---|
| `comparison.json` | Adjusted Rand index and adjusted mutual information over all shared structures (noise as a group of its own), and the ARI over structures clustered in both runs |
| `contingency.csv` | Number of structures per pair of clusters (A rows, B columns) |
| `contingency.pdf/.png` | The same table as a heatmap, coloured by the fraction of each row |
