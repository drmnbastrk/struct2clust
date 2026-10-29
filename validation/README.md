# Validation study

This folder holds the scripts of the simulation study behind the error and
detection rates in [docs/METHODS.md](../docs/METHODS.md#12-validation), and of
the other checks cited there. The result tables behind the numbers of
METHODS.md are in `results/`; the per-run records (`runs*/`) are not part of
the repository and are written when a study is run (see "Re-running").

## What is tested

Synthetic scenarios with a known answer (6 dimensions, Euclidean distances),
run through the same clustering step as `struct2clust run` with default
settings.

| Group | Scenarios | Question |
|---|---|---|
| `top_null_*` | one structureless group: Gaussian, uniform, elongated, curved, density gradient, drifting MD trajectory | Is anything reported as distinct states? |
| `top_null_copies` | 40 structureless points, each repeated as 6 near-identical copies | Are copies of entries reported as distinct states? |
| `top_pos_*` | 2-3 groups, 4-6σ apart, round or elongated; two groups 6σ apart split 80:20 or 90:10; 4 or 6 groups with every pair 6σ apart | Are the groups found? |
| `traj_heldout_*` | trajectories not used while the second pass was designed: faster and slower drift, anisotropic drift, a slow walk along a curve; three states, two states with transition paths, two states 4σ apart | Does the second pass hold on new kinds of drift and states? |
| `traj_*` | trajectories: two overlapping replicas of a structureless drift; two states switching slowly (about 5 transitions in 1000 frames), at a medium rate (about 20), fast (about 100), with unequal populations (80:20), or in two replicas | Is anything reported for the drift, and are the states found, or reported as `insufficient_sampling`? |
| `refine_null_*` | two groups far apart, without structure inside | Does refinement split them? |
| `refine_pos_*` | three sub-groups 4-8σ apart, hidden by a far-away group | Does refinement recover them? |
| `refine_scale_*` | two tight sub-groups 0.5, 1.5 or 3 state scales apart, next to one ordinary group; run with the optional effect-size rule at ratio 1 | Are they reported as one state below the state scale and as two above it? |

The generators are in [`scenarios.py`](scenarios.py). Scenarios with replica
labels (the drifting MD scenarios and `traj_*`) are run as trajectories, as
`struct2clust run` runs them: automatic blocks resampled like groups, and the
sampling check, with the default second pass in blocks of one membership
autocorrelation time (`input.membership_blocks`, METHODS.md, section 9). They are
run a second time without it (`--single-pass`), written to `runs_single_pass/`.

### Dependent structures and resampling by group (`input.groups`)

Two further studies, 20 samples per scenario (METHODS.md, section 12, "Resampling
by group"):

| Script | Scenarios | Question | Runs |
|---|---|---|---|
| [`group_study.py`](group_study.py) `grouped` | copies of entries and run effects (structureless or 3 groups), with the source as group; scenarios of `scenarios.py` with 30 random sources | Does resampling by group remove clusters made of dependent structures, and what does it cost when sources carry no structure? | `runs_grouped/` |
| [`group_study.py`](group_study.py) `frac05` | scenarios of `scenarios.py` without groups, subsample fraction 0.5 | Is the cost of grouping due to the groups or to the smaller subsample? | `runs_frac05/` |
| [`min_groups_study.py`](min_groups_study.py) `run`, `table` | a cluster from 6, 9 or 12 runs next to a far-away group: structureless, structureless with strong run effects, or 3 sub-groups 6σ apart | How does the minimum number of groups for refinement (4, 6, 8, 10) change false and missed splits? | `runs_min_groups/`, `results/min_groups.csv` |

These are synthetic; they show the behaviour of the option in these settings,
not that a given group column captures the dependence of real data.

## Output files

| File (written by) | Content |
|---|---|
| `results/runs.csv` (`summarise.py`) | One row per run (43 scenarios × 50 samples: the 36 of METHODS.md section 12 and the 7 `traj_heldout_*`): clusterability, conclusion, clusters selected, at the top level and final, effect-size merges, sub-structure, splits, tiers, ARI, noise; for trajectories the sampling check, the conclusion with independent blocks and the blocks |
| `results/single_pass/` (`summarise.py --single-pass`) | The same for the trajectory scenarios with `input.membership_blocks` off (one pass with the geometric blocks) |
| `results/summary.csv` (`summarise.py`) | One row per scenario: rates and means |
| `results/without_gate/` (`summarise.py --without-gate`) | The same for `refine_null_*` with the clusterability test switched off: 10 samples, run with the optional effect-size rule at ratio 1 (METHODS.md, section 8) |
| `results/effect_size_sensitivity.csv` (`effect_size_sensitivity.py`) | The effect-size rule at ratios 0.5, 1 and 2 (METHODS.md, section 8) |
| `results/seed_reproducibility.csv`, `_summary.csv` (`seed_reproducibility.py`) | The same data clustered with different random seeds: how many different partitions the seeds give (METHODS.md, section 12) |
| `results/random_halves.csv` (`random_halves.py`) | The effect-size ratio of random halves of the clusters of the examples (METHODS.md, section 6) |
| `results/real_alanine.csv`, `results/real_adk.csv` (`real_benchmark.py`) | Real molecules with a known answer: alanine dipeptide trajectories (states from the backbone dihedrals) and adenylate kinase ensembles (open 4AKE and closed 1AKE with network-model fluctuations): conclusion, clusters and ARI to the true states |
| `results/population_coverage.csv` (`population_coverage.py`) | Coverage of the nominal 95% population intervals and rate of the halves flag for several block lengths (METHODS.md, section 9) |
| `results/block_sensitivity_<run>_*.csv` (`block_sensitivity.py`) | A finished trajectory run clustered again with other block numbers, and its blocking analysis (METHODS.md, section 9) |
| `results/symmetry/` (`equivalent_atoms.py`, `symmetry_check.py`) | The table of chemically equivalent atoms checked against the Chemical Component Dictionary, and examples clustered with and without matching them (METHODS.md, section 2) |
| `results/population_calibration.csv` (`population_calibration.py`) | Coverage of the population intervals at nominal levels 95-99.9% on two-state Markov chains with known populations (METHODS.md, section 9) |
| `results/threshold_sensitivity_<run>.csv` (`threshold_sensitivity.py`) | A finished run repeated with Jaccard thresholds 0.70/0.75/0.80 and subsample fractions 0.5/0.8: conclusion, clusters and agreement with the run (METHODS.md, section 13) |
| `results/block_length_politis_white_*.csv` (`block_length_politis_white.py`) | The automatic block length of Politis and White for AR(1) series and for the trajectories of the 1RKJ example, compared with the block length used (METHODS.md, section 13) |
| `results/min_groups.csv` (`min_groups_study.py table`) | False and missed splits for minimum group numbers 4, 6, 8 and 10 (METHODS.md, section 12) |

The tables in METHODS.md section 12 are taken from `runs.csv` and
`summary.csv`; METHODS.md gives every rate as a count with its 95%
Clopper-Pearson interval. The runs used Python 3.12, numpy 2.5, scipy 1.18 and
scikit-learn 1.9. The main study needs about 85 CPU-hours with 50 samples, about 65 of them
for the trajectory scenarios (5-15 CPU-minutes per run); their single-pass runs
need about 45 more, and the group and minimum-group studies about 18.

## Re-running

```bash
python validation/run_study.py --seeds 50 --jobs 2     # all scenarios, 2 parallel runs
python validation/run_study.py --scenarios top_null_banana --seeds 50
python validation/summarise.py                         # writes results/runs.csv, summary.csv

python validation/run_study.py --single-pass --seeds 50 --jobs 2 --scenarios \
    top_null_md_drift refine_null_md_drift traj_null_replicas_meet traj_pos_markov_slow \
    traj_pos_markov_medium traj_pos_markov_fast traj_pos_markov_unequal traj_pos_markov_replicas
python validation/summarise.py --single-pass           # writes results/single_pass/

python validation/run_study.py --without-gate --seeds 10 --jobs 2 --scenarios \
    refine_null_gauss refine_null_uniform refine_null_dominant \
    refine_null_elongated refine_null_banana refine_null_gradient
python validation/summarise.py --without-gate          # writes results/without_gate/

python validation/effect_size_sensitivity.py
python validation/seed_reproducibility.py --data-seeds 3 --mc-seeds 8 --jobs 2
python validation/random_halves.py example/results/*     # after running the examples
python validation/threshold_sensitivity.py example/results/tar --jobs 2
python validation/real_benchmark.py fetch --data validation/real_data   # about 135 MB
python validation/real_benchmark.py alanine --data validation/real_data --jobs 4
python validation/real_benchmark.py adk --data validation/real_data --jobs 4
python validation/real_benchmark.py table
python validation/population_calibration.py validation/results/population_calibration.csv 200
python validation/block_length_politis_white.py --self-check
python validation/block_length_politis_white.py example/results/1RKJ_rna

python validation/alternative_rules.py --samples 5 --jobs 2   # resolution rule vs alternatives
python validation/bet_limit_sensitivity.py synthetic --reps 500 --jobs 2
python validation/bet_limit_sensitivity.py examples example/results/endothiapepsin example/results/eralpha
python validation/mc_error_example.py example/results/endothiapepsin   # Monte Carlo error
python validation/equivalent_atoms.py --cache ../ccd --out validation/results/symmetry
python validation/symmetry_check.py example/data/eralpha --out validation/results/symmetry
```

`equivalent_atoms.py` checks the table of chemically equivalent atoms against the
PDB Chemical Component Dictionary; `symmetry_check.py` clusters an ensemble with
atoms paired by name and with equivalent atoms matched, side by side.

Every finished run is saved at once to `validation/runs/`, and existing runs are
skipped, so an interrupted study can be restarted with the same command, and
scenarios can be run in any order (`--scenarios`): every run depends only on its
scenario and sample number. Fewer samples (`--seeds 10`) give a quick check with
wider confidence bounds; on a Slurm cluster, adapt
[`slurm_example.sh`](slurm_example.sh).

The study keeps all the cores it is given busy for hours. On a laptop, use few
parallel runs (`--jobs`) and make sure the machine is cooled well.
