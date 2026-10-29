# Tests

```bash
python -m pytest -q tests
```

Four kinds of test:

- **Unit:** one function, known input and output.
- **Scientific regression:** a property the method promises, checked on data
  with a known answer.
- **Integration:** several modules together, e.g. loading and atom matching.
- **End-to-end:** a full `struct2clust run` and its output files.

The table lists, per claim of [docs/METHODS.md](../docs/METHODS.md), the tests
that check it.

| Claim (METHODS section) | Kind | Tests |
|---|---|---|
| Inputs must be one molecule; residue-name variants are unified (1) | integration | `test_load_ensemble_rejects_different_sequence`, `test_construct_with_an_extra_residue_is_rejected`, `test_canonical_names`, `test_residue_kind`, `test_terminal_nucleotides_of_every_base`, `test_unknown_terminal_names_are_not_nucleotides`, `test_trajectories_of_different_molecules_are_rejected` |
| Atoms are matched by residue and name; an atom missing anywhere is compared nowhere (1) | integration | `test_load_ensemble_filters_and_matches_atoms`, `test_atom_missing_in_one_structure_is_left_out_of_all`, `test_atoms_are_matched_by_name_not_by_file_order`, `test_trajectories_with_different_topologies`, `test_describe_selection_lists_atoms_per_residue` |
| Alternate locations are never mixed; highest mean occupancy wins (1) | scientific regression | `test_alternate_conformers_are_never_mixed`, `test_conformer_with_the_highest_occupancy_is_kept` |
| Ligands, free residues with standard names and unrecognised residues are left out; gaps inside chains are reported (1) | scientific regression | `test_free_ligand_with_a_standard_residue_name_is_ignored`, `test_unlinked_residues`, `test_modified_residue_inside_a_chain_is_reported` |
| Chains keep distinct identities (1) | integration | `test_pdb_chain_ids_stay_distinct`, `test_multimodel_file_and_write_roundtrip` |
| Kabsch RMSD equals the explicit superposition; reflections are not rotations (2) | unit | `test_rmsd_matrix_matches_explicit_kabsch` |
| Naming of chemically equivalent atoms is not a difference (2) | scientific regression | `test_equivalent_atoms_naming_is_not_a_difference`, `test_equivalent_atoms_match_exhaustive_search_on_noisy_copies`, `test_equivalent_atoms_are_matched_by_default`, `test_equivalent_atoms_matched_by_default_is_reported` |
| TM-score: superposition on the core, d0, US-align parsing and errors (2) | unit | `test_tm_superposition_fits_the_core_not_the_moved_tail`, `test_tm_d0`, `test_parse_usalign_table`, `test_missing_usalign_is_clear_error` |
| TM-score distances with the real US-align program: symmetric, zero diagonal, identical structures at distance 0 (2); skipped unless US-align is installed (`STRUCT2CLUST_USALIGN` or `USalign` on PATH); it also needs the endothiapepsin example data (`example/prepare_crystal_ensembles.py`) | integration | `test_tm_distances_with_usalign` |
| Near-duplicate removal does not depend on input order; ties go to the more central structure (3) | unit | `test_deduplicate_keeps_first`, `test_deduplicate_does_not_depend_on_the_input_order`, `test_deduplicate_ties_go_to_the_more_central_structure`, `test_duplicates_and_atom_pairing_are_reported` |
| Clusterability: eigengap and graded conclusion (4) | unit | `test_spectral_eigengap_counts_disconnected_groups`, `test_conclusion_is_graded_by_evidence` |
| Conclusion needs two final clusters (7) | unit, scientific regression | `test_one_final_cluster_gives_insufficient_clusters`, `test_insufficient_clusters_leaves_the_other_conclusions_alone`, `test_two_final_clusters_keep_the_conclusion`, `test_validation_scripts_take_the_conclusion_from_run_conclusion`; the report check is in `test_absolute_floor_merges_two_clusters_and_report_is_written` |
| Finest reproducible partition: positive and negative controls (6) | scientific regression | `test_positive_control_three_separated_groups`, `test_nested_groups_are_not_under_clustered`, `test_background_noise_does_not_create_clusters`, `test_negative_control_gaussian_noise`, `test_negative_control_uniform_noise`, `test_candidate_grid` |
| Sequential test: error at most α at any number of looks; clear cases decided early; undecided ones borderline (6) | scientific regression | `test_betting_test_errs_at_most_alpha_however_often_it_looks`, `test_betting_test_decides_clear_cases`, `test_sequential_evaluation_stops_early_when_the_decision_is_clear`, `test_sequential_evaluation_marks_undecided_candidates_borderline`, `test_selection_decision_does_not_depend_on_the_seed` |
| Run-level risk: the shares of all selections add up to at most R (6) | unit | `test_node_risks_add_up_to_at_most_half_the_run_risk`, `test_refinement_reports_each_nodes_share_of_the_risk` |
| Tied settings: partitions compared up to cluster numbering (6) | unit | `test_same_partition_ignores_cluster_numbering_only`, `test_tied_settings_with_different_partitions_are_flagged` |
| Effect size: reported for every pair by default; with the optional rule, groups closer than the state scale are one state (6, 8) | scientific regression | `test_effect_size_is_reported_not_applied_by_default`, `test_separation_pairs`, `test_groups_below_the_state_scale_are_one_state`, `test_effect_size_rule_can_be_switched_off`, `test_absolute_separation_floor`, `test_absolute_floor_merges_two_clusters_and_report_is_written`, `test_separation_notes`, `test_separation_range_brackets_the_ratio` |
| Tiers are capped by clusterability; merged groups keep the weakest statistics (7) | unit | `test_cap_tiers`, `test_merge_groups_keeps_weakest_statistics_and_sums_membership`, `test_relabel_by_size_and_medoids` |
| Resampling by group (`input.groups`, 6): subsamples keep or drop whole groups; an absent cluster counts as not reproduced; copies of entries are not reproducible by group, states of many entries are; at least 8 groups per run and per refined cluster; configuration and group file checks | unit, scientific regression | all tests in `test_groups.py` |
| Without `input.groups`, a full clustering (selection, stability, refinement, conclusion) equals that of the version before resampling by group | regression | `test_without_groups_results_equal_the_version_before_groups` |
| Refinement finds hidden states and keeps homogeneous clusters (8) | scientific regression | `test_refinement_resolves_states_hidden_by_a_dominant_split`, `test_refinement_keeps_homogeneous_clusters`, `test_refinement_applies_the_rule_inside_a_cluster`, `test_partial_merge_labels_only_the_merged_parts` |
| Trajectories: contiguous blocks, never across replicas; automatic blocks of about one autocorrelation time; short blocks flagged; population intervals and agreement between parts; automatic blocks resampled like groups; sampling check (also clusters without transitions) and `insufficient_sampling` (16) | unit | `test_trajectory_blocks_are_contiguous_and_resampled_whole`, `test_automatic_blocks_are_resampled_like_groups`, `test_sampling_check_separates_insufficient_sampling`, `test_clusters_without_transitions_fail_the_sampling_check`, `test_insufficient_sampling_overrides_the_conclusion`, `test_resampling_blocks_never_span_replicas`, `test_resampling_blocks_per_replica`, `test_automatic_blocks_hold_about_one_autocorrelation_time`, `test_population_intervals_and_part_agreement`, `test_autocorrelation_time_of_a_slow_walk`, `test_trajectory_sampling_flags_short_blocks`, `test_trajectory_sampling_reports_the_error_of_the_autocorrelation_time`, `test_select_frames_spreads_over_whole_trajectory`, `test_trajectory_loader`, `test_replica_labels`, `test_two_trajectories_become_one_ensemble` |
| Torsions, RMSF, contacts, medoid differences (10) | unit | `test_dihedral_known_values`, `test_circular_statistics_handle_wraparound`, `test_circular_separation`, `test_rmsf_zero_for_rigid_copies_and_superposition`, `test_contact_frequencies`, `test_no_torsion_across_missing_residues`, `test_medoid_differences_locate_the_differing_residue`, `test_characterise_rna_ensemble` |
| Composition by source, bias-corrected Cramér's V (10) | end-to-end | `test_composition_by_source`, `test_composition_with_incomplete_sources`, `test_bias_corrected_cramers_v` |
| Full run: outputs, states recovered, replicas, no-cluster case, comparison (all) | end-to-end | `test_outputs_exist`, `test_states_are_recovered`, `test_two_replicas_clustered_together`, `test_no_clusters_writes_a_report`, `test_compare_run_with_itself`, `test_cluster_views`, `test_member_opacity` |
| Output files and `run_summary.json` keys are the documented ones ([OUTPUTS.md](../docs/OUTPUTS.md)) | end-to-end | `test_output_columns_match_the_documentation` |
| `run_summary.json` names the code that wrote it (SHA-256 of the package source) | unit | `test_source_hash_covers_every_python_file_of_the_package`, `test_versions_report_the_source_hash` |
| Configuration and command line: defaults, unknown keys, overrides, clear errors | unit | all tests in `test_config_cli.py` |

The reference of `test_regression_ungrouped.py` (`tests/data/regression_ungrouped.json`)
was written on Windows 11 with Python 3.12 and the package versions in
`requirements/tested-windows-py312.txt`. The test compares exactly, so other
package versions or platforms can give small numerical differences that fail it.
The CI therefore runs it in a separate job with these pinned versions and leaves
it out of the general operating-system and Python matrix
(`.github/workflows/ci.yml`).

The scientific regression tests check behaviour on small synthetic data. They do
not replace the simulation study ([docs/METHODS.md](../docs/METHODS.md),
section 12).
