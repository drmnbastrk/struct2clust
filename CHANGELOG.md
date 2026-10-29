# Changelog

## 1.0.0

Initial public release. Contents are:

- **Input and atom matching:** PDB directories, multi-model PDB files and MD
  trajectories (several, also with different topologies) of protein, RNA, DNA
  and their complexes; atoms matched across structures, chemically equivalent
  atoms matched by default.
- **Distances:** heavy-atom RMSD (default) or "1-TM-score" with US-align;
  near-duplicate structures of ensembles removed before clustering.
- **Clusterability test** before clustering.
- **Resolution selection:** HDBSCAN at the finest resolution whose clusters are
  all reproducible under subsampling, each candidate decided by a sequential
  test with enough subsamples that the result does not depend on the seed.
- **Cluster-wise stability** and tiers; trajectories resampled in blocks.
- **Trajectories:** sampling check, cluster populations with approximate 95%
  block-bootstrap intervals, and agreement of the populations between
  trajectories.
- **Hierarchical refinement** inside every cluster.
- **Effect size** reported for every pair of clusters, with an optional rule
  that reports groups closer than the state scale as one state.
- **Graded conclusion:** `distinct_states`, `partition_not_confirmed`,
  `no_group_structure`, `no_reproducible_partition`, `no_clusters` or
  `insufficient_clusters`.
- **Resampling by group (optional, `input.groups`):** subsamples keep or drop
  whole sources (PDB entry, prediction run, …); at least 8 groups are required.
  Without groups, results are unchanged.
- **Output:** report, tables, `run_summary.json` with all settings, the
  package versions and the SHA-256 of the package source; representative
  structures, per-cluster VMD/PyMOL views, figures, and descriptive
  characterisation (RMSF, contacts, torsions, radius of gyration).
- **Commands:** `run`, `composition`, `compare`, `config`.
- **Documentation:** method, rationale, validation and limitations
  (docs/METHODS.md), outputs, configuration and atom selections; simulation
  study and other checks in `validation/`; example scripts in `example/`.
