# Methods

This document describes what Struct2Clust computes, why each choice was made,
and what the choices were tested against. The file names and column names used
here are listed in [OUTPUTS.md](OUTPUTS.md). Every setting mentioned here is
listed in [CONFIGURATION.md](CONFIGURATION.md).

## Overview

![Struct2Clust workflow](images/workflow.png)

```
structures / trajectories
  -> common atoms (same molecule required)
  -> pairwise distance matrix (RMSD, or "1-TM-score")
  -> removal of near-duplicates
  -> clusterability (is there cluster structure at all?)
  -> resolution selection (finest partition whose clusters are all reproducible)
  -> HDBSCAN with the selected setting
  -> cluster-wise stability (tiers, membership probabilities)
  -> effect size of every pair of clusters (reported; an optional rule merges groups
     closer than the ensemble's states are wide)
  -> hierarchical refinement (the same rules inside every cluster) and medoids
  -> overall conclusion, graded by the strength of the evidence
  -> descriptive characterisation (RMSF, contacts, torsions, radius of gyration)
  -> tables, figures, report
```

The method has one statistical idea at its centre: **a cluster is only reported
as reliable if it is reproducible**. That is, the same cluster must reappear when
the data are subsampled (Hennig 2007). The same criterion chooses the resolution,
assigns confidence tiers and gives every structure a membership probability.

Reproducibility is necessary but not sufficient. A continuum without groups,
such as a curved or elongated cloud of structures, can also be cut in the same
place every time (Ben-David et al. 2006; von Luxburg 2010). Struct2Clust
therefore reports **distinct states** only when the data also show group
structure (section 4). Otherwise the clusters are reported as reproducible
regions of the ensemble (section 7, "Overall conclusion"). Finally, groups
that exist need not differ as much as the ensemble's states do. For every pair
of clusters the report therefore gives the distance between their medoids
relative to the typical width of a state; an optional rule reports groups
closer than that width as one state (section 6, "Effect size"). The error
rates of these statements were measured in a simulation study (section 12).

## 1. Input and atom matching

**Ensembles.** An ensemble is a directory of PDB files, or a single multi-model
PDB file, where every model is one member. Examples are NMR models, structure
predictions or refinement outputs.

**Trajectories.** One or more MD trajectories are read with MDAnalysis
(Michaud-Agrawal et al. 2011; Gowers et al. 2016). Every frame becomes one
member. The trajectories can come from several sources:
- one trajectory,
- replicas of the same simulation,
- runs started from different structures.

They share one topology, or each has its own. They are always clustered
together, so every trajectory is described by the same set of states.

**Same molecule.** All inputs must describe the same molecule, because every
distance compares the members atom by atom.
- Force-field variants of residue names are unified before this check.
  - Terminal nucleotides of every base count as that base: a nucleotide name
    followed by 5 (5′ end), 3 (3′ end) or N (both ends), e.g. `G5`, `U3`, `DT3`,
    `AN`.
  - `HIE`, `HID` and `HSD` count as `HIS`; `RA` counts as `A`; `MSE`
    (selenomethionine in crystal structures) counts as `MET`.
- Old atom names are converted to current PDB names (`O1P` becomes `OP1`, `C4*`
  becomes `C4'`).
- Inputs with different residue sequences are rejected with an error. Different
  constructs of one molecule (e.g. with extra terminal residues) must be cut to
  their common region before clustering.
- Residues whose names are not recognised as protein or nucleic-acid residues
  (modified residues such as pseudouridine or phosphoserine, ligands, water,
  ions) are left out. A left-out residue that lies inside a polymer chain leaves
  a gap there; the log lists such residues with a warning.

**Atom matching.** Atoms are matched by residue position and atom name. The
following are never used:
- hydrogens,
- water, ions and ligands,
- any atom missing from at least one member.

**Alternate locations.** A residue modelled in several conformers contributes
one of them, never a mixture: atoms without an alternate-location label belong
to every conformer and are kept; of the labelled atoms, only those of the label
with the highest mean occupancy (the mean over the atoms that carry the label)
are kept, ties going to the earlier label (A before B). An atom present only in
another conformer is missing from that residue, like any other missing atom.
The number of residues with alternate locations is logged. Atoms that are
dropped are listed by name in the log. The atom subsets are defined in
[ATOMS.md](ATOMS.md).

## 2. Distances

| Metric | Definition |
|---|---|
| `rmsd` (default) | RMSD after optimal rigid-body superposition (Kabsch 1976) over the chosen atom subset. The default subset is `heavy-atoms`, with chemically equivalent atoms matched (below). |
| `tm` | TM-score (Zhang & Skolnick 2004), computed by US-align (Zhang et al. 2022), and used as the distance **"1-TM-score"**. |

**How RMSD is computed.** RMSD is taken from the singular values of the
covariance matrix, with the smallest one negated when the optimal transform
would be a reflection (Kabsch 1978). No rotation matrix has to be built, and all
pairs are computed in vectorised form.

**Chemically equivalent atoms.** RMSD pairs atoms by name. Some atoms of a
residue cannot be told apart by its heavy-atom structure, so their naming is a
convention, and two structures with the same heavy-atom positions can name them
the other way round. These are the automorphisms of the residue's heavy-atom bond
graph with elements as labels (hydrogens and stereochemistry ignored, as in
heavy-atom RMSD):

| Residue | Atoms that change places together |
|---|---|
| Arg | NH1 ↔ NH2 |
| Asp | OD1 ↔ OD2 |
| Glu | OE1 ↔ OE2 |
| Leu | CD1 ↔ CD2 |
| Val | CG1 ↔ CG2 |
| Phe, Tyr | CD1 ↔ CD2 with CE1 ↔ CE2 |
| C-terminal amino acid | O ↔ OXT |
| Nucleotide | OP1 ↔ OP2; at a 5'-terminal phosphate, any order of OP1, OP2, OP3 |

The table was checked against the bond graphs of the PDB Chemical Component
Dictionary for the 20 amino acids and 8 nucleotides, inside a chain and at its
end (`validation/equivalent_atoms.py`). The two methyl groups of Leu (CD1, CD2) and of
Val (CG1, CG2) are chemically identical; which one is called 1 is a naming
convention for prochiral groups (Markley et al. 1998), so they are matched like
the other groups. A change of the side-chain rotamer still counts: when χ2 of
Leu turns by 120°, one methyl group moves to a new position, and that
displacement enters the RMSD; pairing by name would count it twice. lDDT treats
Val in the same way (Mariani et al. 2013). By default (`metric.equivalent_atoms:
true`) the RMSD of a pair is minimised over these renamings in the spirit of
symmetry-corrected RMSD (Meli & Biggin 2020), by alternating two steps that each
never increase the deviation: superposition, then, for every group, the order
of the atoms closest to the other structure. A renaming is made only if it
lowers the deviation, so the rounds end at a naming that no single group can
improve; on test cases with three groups this was the minimum over all
namings. The setting applies to `rmsd` only; `false` pairs atoms by name. The
report and `run_summary.json` state which RMSD was used. Cluster views and
medoid differences use the names as given in the files.

`validation/symmetry_check.py` clusters an ensemble with atoms paired by name
and with equivalent atoms matched, side by side, so the effect of the setting
can be checked on any data set. Matching makes the distance matrix slower to
compute; the factor depends on the ensemble.

**TM-score and "1-TM-score".** TM-score is a *similarity*: 1 for identical
structures, about 0.17-0.3 for unrelated ones. Values above 0.5 (proteins; Xu &
Zhang 2010) or 0.45 (RNA; Gong et al. 2019) indicate the same fold. Clustering
needs a *distance*, which is 0 for identical structures, so Struct2Clust uses
"1-TM-score". Every distance, threshold and axis for `--metric tm` refers to
"1-TM-score": a distance of 0.001 means a TM-score of 0.999.

**How TM-score is computed.** US-align is run in sequence-dependent
superposition mode: `-TMscore 1`, or `-TMscore 2` for multi-chain molecules.
Residues with the same index are treated as equivalent, which is correct for
conformations of one molecule. The alignment-search modes are meant for
comparing different molecules. The TM-score reported is the mean of the two
normalisations. If US-align fails on any pair, the run stops with an error; no
fallback value is used.

**Choice of metric.** On very diverse ensembles, RMSD saturates and compresses
large differences. TM-score weights the well-superposed core more strongly. The
two metrics can therefore give different partitions, most of all on diverse
ensembles. The metric is a scientific choice and should be reported. Two runs can be compared with `struct2clust compare`, which gives
the adjusted Rand index (Hubert & Arabie 1985) and the adjusted mutual
information (Vinh et al. 2010) between their partitions.

## 3. Near-duplicates

After the distance matrix is computed, near-duplicates (structures closer than
`dedup_threshold`) are removed; exact or near-exact copies would otherwise
create artificial density for HDBSCAN. Which copy is kept does not depend on the
order of the input. Following Butina (1999), every structure is counted with its
neighbours (structures within the threshold). In order of decreasing count, a
structure not yet assigned is kept as a centre and its unassigned neighbours are
removed as its copies; ties go to the structure with the smaller summed distance
to its neighbours (the more central one), then to the name. Every removed
structure is within the threshold of its centre; two copies of one centre can be
up to twice the threshold apart. Removed structures, their centres and the distance
are listed in `duplicates.csv` and `run_summary.json`. They take no part in the
clustering, the figures or any statistic; cluster sizes count the clustered
structures only. For deposited structures this means that a conformation
deposited many times (for example in a series of soaks of one crystal form)
counts once per distinct structure: how often a conformation was deposited
reflects which experiments were done, not a population (section 13).

Trajectory frames are not removed. They are samples in time: a frame that comes
back close to an earlier one is a revisit of that conformation, and it counts
towards the populations and the autocorrelation times (section 9). Removing such
frames would shorten the time series unevenly and bias both. For a small or rigid
molecule this matters: the five heavy atoms of the alanine in alanine dipeptide
stay within 0.65 Å of each other over 250 ns, and a 0.1 Å rule kept 38 of 6000
frames. `dedup_threshold` therefore applies to ensembles only.

| Metric | Default threshold | Meaning |
|---|---|---|
| `rmsd` | 0.1 Å | Below the coordinate error of most experimental structures (Cruickshank 1999) and far below thermal fluctuations |
| `tm` | "1-TM-score" ≤ 0.001 | TM-score ≥ 0.999: the structures agree to three decimal places of TM-score |

Published TM-score thresholds (0.5 for proteins, 0.45 for RNA) decide whether
two structures share a fold. They are not meant for detecting copies, and there
is no published TM-score threshold for near-identity. The threshold is therefore
set by TM-score's own scale.

For a uniform deviation d at every residue, "1-TM" ≈ (d / d0)², where d0 grows
with chain length (proteins: Zhang & Skolnick 2004; RNA: Gong et al. 2019). A TM-score of 0.999 therefore corresponds to:

| Molecule | d0 | d at "1-TM" = 0.001 |
|---|---|---|
| RNA, 20 nt | 0.60 Å | 0.02 Å |
| RNA, 42 nt | 1.37 Å | 0.04 Å |
| RNA, 100 nt | 3.48 Å | 0.11 Å |
| Protein, 300 residues | 6.36 Å | 0.20 Å |

For RNAs up to about 100 nt this is at or below the RMSD threshold of 0.1 Å.
For large proteins it is up to about twice as large, which is still far below
thermal fluctuations.

TM-score compares one atom per residue (C3′ or CA). Structures with identical
backbones but different base or side-chain positions can therefore count as
copies. Use `--metric rmsd` when such differences matter.

TM-score compares the overall fold; for local and side-chain differences, use
RMSD. On the 52 example and benchmark data sets, clustered with both metrics,
`--metric tm` removed most structures of several protein ensembles as copies
(e.g. 399 → 25 and 340 → 25), whose RMSD clusters differ in side chains and
loops. Of the 49 data sets that could be clustered with both, TM-score gave the
same conclusion as RMSD in 27, a weaker one in 18, a stronger one in 3 and
`no_group_structure` instead of `no_reproducible_partition` in 1, and fewer
clusters in 20. In three large molecules (two RNA assemblies of more than
2,600 residues and a protein-DNA complex) fewer than 10 structures remained, and
the run stopped, as for any input with fewer than 10 distinct structures.

For short chains d0 is at its floor (RNA below 30 nt: 0.3-0.7 Å; proteins up
to about 20 residues: 0.5 Å), smaller than thermal fluctuations of about 1 Å.
Every deviation then lowers TM-score almost as much as a conformational change,
and "1-TM-score" no longer separates states from noise: the synthetic example
(8 nt, three states 6 Å apart) gives three distinct states with RMSD and no
reproducible partition with TM-score. Use `--metric rmsd` for short chains.

## 4. Clusterability

Before clustering, two signals computed from the same distances ask whether the
data contain any group structure: a test and a heuristic. The result is reported as a verdict.

- **Dip-dist.** This is Hartigan's dip test of unimodality (Hartigan & Hartigan
  1985) applied to the pairwise distances. Adolfsson et al. (2019) found it one
  of the most reliable clusterability tests.
  - At most 72,000 randomly sampled pairs are used. This is the range covered by
    the test's p-value tables.
  - Pairwise distances are not independent of each other, so the p-value is
    indicative, not exact.
- **Spectral eigengap.** A k-nearest-neighbour graph (k = 15) is built with
  self-tuning Gaussian affinities (Zelnik-Manor & Perona 2004). The suggested
  number of groups is the position of the largest gap among the smallest
  eigenvalues of its normalised Laplacian (von Luxburg 2007).
  - Because the affinity scale adapts to local density, the result does not
    depend on distance units.
  - k = 1 is possible and means that there is no gap.

| Verdict | Dip test (p < 0.05) | Eigengap (k ≥ 2) | Highest tier allowed |
|---|---|---|---|
| CLUSTERABLE | yes | yes | HIGH |
| WEAK | only one of the two | | MODERATE |
| NOT_CLUSTERABLE | no | no | LOW |

**Why the verdict limits the tiers.** Resampling stability tests whether a
partition can be reproduced, not whether it is real: a curved or elongated
continuum can be cut in the same place in every subsample (Ben-David et al.
2006; von Luxburg 2010). In the simulation study (section 12), 13 of 50 samples
of a structureless curved cloud gave a reproducible partition; the verdict
(WEAK) kept the clusters of 11 of them below HIGH. Clustering always proceeds,
whatever the verdict.

**What the two tests measure.** The simulation study (section 12) shows that
the two tests answer different questions:

| Data (50 samples per scenario) | Dip test p < 0.05 | Eigengap k ≥ 2 |
|---|---|---|
| One Gaussian or uniform group | 0/100 (0-4%) | 0/100 (0-4%) |
| One elongated, curved or density-gradient group, or a drifting MD trajectory, without groups | 6/200 (1-6%) | 200/200 (98-100%) |
| 2 or 3 round groups, 5-6σ apart | 148/150 (95-100%) | 150/150 (98-100%) |
| 3 elongated groups, 6σ apart | 39/50 (64-88%) | 50/50 (93-100%) |
| 3 round groups, 4σ apart | 1/50 (0-11%) | 50/50 (93-100%) |
| 40 structureless points, 6 near-identical copies of each | 50/50 (93-100%) | 47/50 (83-99%) |

Counts are samples out of samples; the 95% Clopper-Pearson interval is given in
parentheses. The last row is discussed in section 12 ("Copies of entries").

- The **dip test** is the evidence for groups: it is rarely significant without
  groups and almost always significant with well-separated ones.
- The **eigengap** also responds to anisotropy. For a continuum the eigenvalues
  grow steadily and the largest gap falls at the end of the search window; the
  heuristic is reliable only for pronounced clusters (von Luxburg 2007).

A **WEAK** verdict (eigengap only) therefore means that weakly separated groups
and a continuum cannot be told apart. The same holds for groups of very unequal
size and for many groups (below, "Power of the dip test"). Both occur in the study: 3 groups 4σ apart
and structureless elongated data are both WEAK. This is why WEAK caps the tiers
at MODERATE and why the result is reported as "group structure not confirmed".
In single MD trajectories, neighbouring states are connected by transition
frames, so the dip test is often non-significant (p ≈ 1) even when substates
are reproducible.

**Power of the dip test.** The dip test detects groups when the distances within
groups and between groups form two modes of comparable mass. With one small
group, or with many groups, the distances between groups dominate, and the test
loses power. Groups given as the partition, 100 samples each
([`validation/clusterability_power.py`](../validation/clusterability_power.py)):

| Groups (6-D, unit variance) | Dip test p < 0.05 |
|---|---|
| 2 groups 6σ apart, 50% / 50% | 100/100 |
| 2 groups 6σ apart, 80% / 20% | 42/100 |
| 2 groups 6σ apart, 90% / 10% | 1/100 |
| 2 groups 6σ apart, 95% / 5% | 0/100 |
| 2 groups 8σ apart, 90% / 10% and 95% / 5% | 100/100 and 94/100 |
| 4 groups, every pair 6σ apart | 20/100 |
| 6 groups, every pair 6σ apart | 0/100 |

Such data are reported as `partition_not_confirmed` with tiers at most MODERATE,
although the groups are found and reproducible, and a cluster that holds them is
not split in refinement (section 8). Two tests that do not depend on the balance
were compared on the same samples and on structureless data cut into parts (by
k-means, or into an outer 10-20% and the rest): the dip test on the union of
every cluster and its nearest one, the larger subsampled to the size of the
smaller, and the dip test on the projection onto the line through the two
medoids (with or without sample splitting). Each detected the unequal or many
groups more often (up to 63/100 for 90% / 10%, 100/100 for six groups), but in
its worst case reported group structure in a curved structureless cloud or a
drifting MD trajectory in 91, 99 and 76 of 100 samples respectively, against at
most 8 of 100 for the dip test on all distances. No assumption-free test separates a small group from the tail
of a curved continuum, so the dip test on all distances is kept: it rarely
confirms groups that are not there, at the cost of missing unequal or many
groups. In that case, `separation.csv` (medoid distances well above the scatter
of the clusters) and the map are the evidence to read.

## 5. Clustering

HDBSCAN (Campello et al. 2013; scikit-learn implementation, Pedregosa et al.
2011) is run on the precomputed distance matrix. Excess-of-mass (EOM) cluster
extraction is used. Structures in no dense region are labelled noise. The
medoid of a cluster is its member with the smallest summed distance to the
other members.

## 6. Resolution selection

HDBSCAN needs two parameters: `min_cluster_size` and `min_samples`. Different
values give different resolutions, and data with nested structure are
legitimately clusterable at several of them. Struct2Clust does not report
alternative resolutions. It chooses one, with an explicit rule:

> **Select the finest partition in which every cluster is reproducible**, i.e.
> has a mean Jaccard similarity of at least 0.75 over subsamples (Hennig's
> threshold for a stable cluster).

- A finer partition whose parts are all reproducible always wins over a coarser
  one. This prevents **under-clustering**.
- A cluster that does not reproduce disqualifies its partition. Examples are a
  density fluctuation split off a larger group, or a group made of noise. This
  prevents **over-clustering**.
- **Ties** (same number of clusters) are broken by the lower noise fraction,
  i.e. the setting that assigns the most structures to its reproducible
  clusters, then by the higher mean Jaccard, then by the larger
  `min_cluster_size`. Noise is thus a secondary criterion: it ranks only
  settings that are equally fine and equally reproducible.
- **No reproducible partition.** If no partition qualifies, the one whose
  weakest cluster is most stable is returned for inspection. It is flagged as
  "no reproducible cluster structure" (`reproducible_partition_found: false`).
- **No clusters.** If no setting gives two or more clusters, every structure is
  reported as noise (conclusion `no_clusters`, section 7).

### Candidates

| Parameter | Candidate values |
|---|---|
| `min_cluster_size` | every integer from 5 to 20, then steps of about 10% up to max(10, N/10) |
| `min_samples` | 1, 2, 3, 5, 8 and `min_cluster_size` (never above `min_cluster_size`) |

Every setting is assessed separately, even when two settings give the same
partition on the full data, because they behave differently on subsamples.

### Monte Carlo precision of the decision

Whether a candidate qualifies is decided from a mean Jaccard that is
*estimated* from random subsamples. With a fixed, small number of subsamples,
the decision for a candidate whose true value lies near 0.75 changes with the
random seed. This is Monte Carlo error, not a property of the data, and it can
go either way: a spurious split in one run, a missed one in the next.
Struct2Clust therefore decides every candidate with a sequential test whose
error is bounded however long it runs, the approach of Monte Carlo testing with
bounded resampling risk (Gandy 2009; Ding, Gandy & Hahn 2020):

1. Subsamples are added in batches of 10, at least `selection.min_resamples`
   (20).
2. For every cluster, two wealth processes bet on its per-subsample Jaccard
   values, which lie between 0 and 1: one that the mean Jaccard is above 0.75,
   one that it is below (testing by betting; Waudby-Smith & Ramdas 2024). Each
   starts at 1 and is multiplied, with every subsample, by 1 + λ(J − 0.75)
   (respectively 1 − λ(J − 0.75)); the stake λ is set from the earlier values
   only, by approximate growth-rate adaptive betting (aGRAPA: λ = (μ̂ − 0.75) /
   (σ̂² + (μ̂ − 0.75)²), with μ̂ and σ̂² the mean and variance of the earlier
   values and one pseudo-value of mean 1/2 and variance 1/4), and truncated to
   c/0.75 upwards and c/0.25 downwards with c = 0.75, one of the authors'
   suggested defaults (1/2 or 3/4). Every c < 1 keeps the wealth positive, so
   the guarantee below holds for any c; c changes only how fast a decision is
   reached. With c = 0.5, 0.75 and 0.9, the wrong-decision rate on simulated
   values stayed below α for all three (`validation/bet_limit_sensitivity.py
   synthetic`, which writes `validation/results/bet_limit_synthetic.csv`). If a bet is wrong, its wealth is a nonnegative supermartingale, and by
   Ville's inequality (Ville 1939) it ever reaches 1/α with probability at most
   α. A wealth that reaches 1/α therefore establishes its side at level α,
   after any number of looks and without a normal approximation. This matters
   because the Jaccard values of single subsamples are often two-peaked: a
   cluster is found almost unchanged, or it dissolves.
3. A candidate **qualifies** when every cluster is established above 0.75. This
   is an intersection-union test (Berger 1982): testing each cluster at level α
   keeps the candidate at level α. It **does not qualify** as soon as some
   cluster is established below 0.75 at level α/k (k clusters).
4. α is the selection's share of the **resampling risk**
   (`selection.resampling_risk`, R = 0.001), divided by the number of its
   candidates (Bonferroni). The top-level selection receives R/2. A refinement
   selection (section 8) at depth ℓ (0 = inside a top-level cluster) on n_v of
   the N structures receives (R/2) · 2^−(ℓ+1) · n_v / N.
5. A candidate still undecided after `selection.max_resamples` (1000)
   subsamples is marked **borderline** (`selection.csv`) and **does not
   qualify**. A mean exactly at the threshold cannot be decided by any
   sequential procedure (Gandy 2009); such a candidate's stability has not been
   established. If a borderline candidate has more clusters than the selected
   one, `run_summary.json` and `report.md` say so.

Every subsample can add only a bounded amount of evidence, so even a perfectly
reproducible candidate needs about 50 subsamples; borderline ones use many.
Candidates are assessed from the most clusters to the fewest: once one
qualifies, coarser candidates cannot change the result and are not assessed
(`selection.csv`, column `evaluated`). Where the evidence stays insufficient,
the finer partition is not claimed, following the convention of Monte Carlo
testing, where a null hypothesis is not rejected unless the Monte Carlo
evidence suffices (Gandy 2009), and of hierarchical testing, where a finer
partition is a claim that needs evidence (Meinshausen 2008; Kimes et al. 2017).

**Guarantee for a whole run.** The *exact* mean Jaccard of a cluster is the mean
over all possible subsamples in which the cluster is present, i.e. the value that
infinitely many subsamples would give. A candidate should qualify if every
cluster's exact mean is at least 0.75. Subsamples are treated as independent
random draws; every candidate has its own stream of the random generator.

1. *One candidate.* A wrong qualification needs a cluster whose exact mean is
   below 0.75 and whose upward wealth nevertheless reaches 1/α: by Ville's
   inequality this has probability at most α, at whatever subsample the test
   stops. A wrong rejection needs one of the k clusters whose exact mean is at
   least 0.75 to have a downward wealth reaching k/α: at most α/k each, at most α
   together. For a given candidate only one of the two errors is possible, so it
   is decided wrongly with probability at most α. The clusters of a candidate
   share subsamples; the bounds do not require them to be independent.
2. *Selections chosen during the run.* Which selections take place depends on
   earlier results: a top-level cluster is refined only if the top-level
   selection found a reproducible partition, a node is split only if its own
   selection did. The share of a selection depends only on its depth and size,
   and its candidates are tested with subsamples drawn independently of
   everything that led to the selection. The probability that some tested
   candidate is decided wrongly is therefore at most the expected sum of the α of
   all tested candidates (union bound over the tests that may take place). A
   selection's candidates hold at most its share in total, because α divides it
   by the number of all its candidates with at least two clusters, and the
   finest-first evaluation tests a subset of them. Nodes of one depth hold
   disjoint sets of structures, so their n_v add up to at most N; on every
   possible course of the run, the shares add up to at most
   R/2 + (R/2) Σ_ℓ 2^−(ℓ+1) ≤ R. Hence, **with probability at least 1 − R, every
   candidate that is decided is decided as its exact means dictate**.
3. *From decisions to the result.* The candidate partitions (HDBSCAN on all
   structures of a node), the clusterability tests (fixed random sample of pairs
   for large N), the effect-size merges and the refinement nodes are functions of
   the data and of earlier decisions only. If, in addition, no candidate is left
   undecided, the run takes the course of the *ideal* run, in which every
   candidate is decided from its exact means: same conclusion and same partition
   at every level.

The guarantee does not cover:
- *Borderline candidates.* They are undecided and do not qualify, so the course
  of the run can differ from the ideal one. The report lists them.
- *Ties.* Among qualifying settings with the same number of clusters and the same
  noise fraction, the higher estimated mean Jaccard decides. If such settings give
  different partitions, the choice can change with the seed; the log and the
  report then say so (`run_summary.json`: `tie_partitions_differ`). If they give
  the same partition, only the setting used for the stability estimates changes.
- *No reproducible partition.* The conclusion is covered; the partition shown for
  inspection (highest estimated weakest-cluster Jaccard) is not.
- *Estimates.* Stability estimates and tiers (section 7) are estimates, not
  decisions.

**What the risk refers to.** The resampling risk concerns Monte Carlo error
only: the probability that a different random seed would change the result on
the same data. It is not an error rate of the clusters with respect to the
population the structures come from. That question is addressed by the
simulation study (section 12), with the limitations of section 13.

**Size of the Monte Carlo error.** With a fixed number of subsamples, a
candidate whose weakest-cluster Jaccard lies near 0.75 can fall on either side
of the threshold depending on the random seed, so a rule with a fixed number
of subsamples can accept or reject the same candidate. The sequential test
leaves such a candidate undecided (borderline) rather than claiming it.
`validation/mc_error_example.py` compares the estimates from 50 and from 1000
subsamples for every candidate of a finished run. Section 12 ("Monte Carlo
reproducibility") shows the result of the same data under different random
seeds.

### Why this rule

A resolution rule for conformational ensembles has four jobs:

1. **Resolve nested states.** When sub-states sit inside larger families, the
   sub-states should be reported, not only the families.
2. **Find nothing in structureless data.** Noise must not be reported as
   clusters.
3. **Ignore diffuse background.** Scattered structures around real states must
   not form clusters of their own.
4. **Keep structures assigned.** Noise must not become a dumping ground that
   makes the remaining clusters look cleaner.

"Every cluster reproducible" addresses jobs 2 and 3: a cluster made of noise or
background is not expected to reappear in subsamples. "Finest such partition" addresses
job 1. Excess-of-mass extraction (section 5) addresses job 4: it assigns
structures to the most persistent clusters instead of trimming them to their
cores.

The table compares the rule with common alternatives on three synthetic controls
with a known answer, five samples each (`validation/alternative_rules.py`, which writes
`validation/results/alternative_rules.csv`). All rules choose among the same
candidate settings. Entries give the number of clusters chosen in each sample.

| Rule | Nested states (6) | Pure noise (0) | Groups + background (3) |
|---|---|---|---|
| **Finest partition, every cluster Jaccard ≥ 0.75 (used)** | **6/6/6/6/6** | **0/0/0/0/0** | **3/3/3/3/3** |
| Highest cross-subsample ARI (Ben-Hur et al. 2002) | 6/6/6/2/6 | 2/2/2/3/2 | 3/3/3/3/3 |
| Highest DBCV (Moulavi et al. 2014) | 2/2/2/2/2 | 2/9/2/13/2 | 3/3/3/3/3 |
| Most stable clusters, unstable ones ignored | 6/6/6/6/6 | 2/0/0/2/2 | 3/3/3/3/3 |
| Every cluster Jaccard ≥ 0.6 | 6/6/6/6/6 | 0/2/0/2/2 | 3/3/3/3/3 |

Nested: two groups 40σ apart, each of three sub-groups 8σ apart (50 points
each). Pure noise: 300 points from one Gaussian. Background: three groups 8σ
apart (80 points each) and 60 points spread uniformly over their bounding box.

Of the rules compared, only the one used reported no clusters in every pure-noise
sample (job 2), and it resolved the nested states in every sample (job 1). DBCV
chose the coarse partition in every nested sample. The background control did not
separate the rules: all of them found the three groups. The comparison covers
these designs only; it does not show that the alternatives fail in general. The
table evaluates the top-level rule alone; hierarchical refinement (section 8)
recovers sub-groups hidden inside dominant groups.

Other resampling criteria were designed for algorithms with a fixed number of
clusters k: the normalised stability of Lange et al. (2004) scores a whole
partition, consensus clustering (Monti et al. 2003) summarises how often pairs
of points are clustered together, and prediction strength (Tibshirani & Walther
2005) transfers labels between halves of the data with a nearest-centroid
classifier and takes the weakest cluster. HDBSCAN has no k and labels noise;
cluster-wise Jaccard similarity (Hennig 2007) handles both directly, which is
why it is used here.

### Effect size: one state or several?

Reproducibility (above) and group structure (section 4) establish that groups
*exist*. They do not say whether two groups are as different from each other as
the states of the ensemble are. Tight groups can be reproducible and cleanly
separated although they differ far less than the members of a single state
differ from each other. For example, two prediction tools starting from the
same template need not produce equivalent models; whether their outputs form
distinct groups, and how the separation between those groups compares with
the spread of states within the ensemble, must be established by evaluating
the resulting structures. Whether such a difference is "meaningful"
cannot be decided by statistics alone; in equivalence testing the corresponding margin is set by
the field, not by the data (Schuirmann 1987). Struct2Clust therefore takes the
margin from the ensemble itself: two groups count as different states only if
they are at least as far apart as the states of the ensemble are wide.

**Measure (always reported).** For every pair of final clusters, the distance
between their medoids is compared with the **state scale**: the median distance
of the members of all *other* clusters to their own medoid, pooled over those
clusters. `separation.csv` and `run_summary.json` (`effect_size.pairs`) give,
per pair, the medoid distance, each cluster's own scatter (median distance of
its members to its medoid), the state scale, their ratio, and the ratio's
spread over 200 subsamples of the structures (the 2.5% and 97.5% quantiles;
medoids and state scale are found again in each subsample). The spread is the
variation between subsamples of the same data, not a confidence interval,
because subsamples of 80% of the data vary less than new data would. With two
clusters there is no other cluster to define the state scale; the medoid
distance and both scatters are given. The report names the closest pair and
every pair with a ratio below 1, i.e. medoids closer than the ensemble's
states are wide. The ratio describes the clusters; it does not change them.

**Optional rule.** With `selection.refine_min_separation_ratio` set above 0
(default 0, off), two clusters (at the top level, or sub-clusters in
refinement) are merged into one state when their medoids are closer than that
multiple of the state scale (in refinement: the state scale of the top-level
clusters other than the parent). Merging is by single linkage. The merged
groups are numbered within their cluster in `clusters.csv` (column
`substructure`: 1, 2, …), and the report names them with their medoid
distances and ratios, so finer structure is reported, not discarded.

**Why the ratio is reported rather than applied by default.** The margin that
makes two groups "different states" is a convention, not an estimate (below),
and the ratio is noisy: random halves of one cluster also lie apart (next
point). Turning a noisy effect size into a yes/no
decision at a conventional threshold hides how close a pair lies to it; reporting
the effect size with its variation lets the reader judge (Wasserstein & Lazar
2016; Cumming 2014). Where it acts, it changes the reported partition at a margin chosen by convention. It is therefore
available, with its effect characterised (section 8), but off by default.

**Why these definitions.**
- Separation is measured between **medoids**, scatter as **distance to the
  medoid**. Both are the quantities of the Davies-Bouldin index (Davies &
  Bouldin 1979), which compares cluster scatter with the distance between
  cluster centres. The median distance *between members* of two groups is not
  used, because it contains the groups' own scatter, which the medoid distance
  does not. The medoid distance is not free of sampling noise either: the
  medoids of two random halves of one cluster also lie apart
  ([`validation/random_halves.py`](../validation/random_halves.py) measures
  this ratio for the clusters of finished runs). The ratio is therefore not
  used to tell random halves from real sub-groups: halves of one group
  are neither clusterable nor reproducible, and both are required first (next
  point; section 8). The ratio only decides whether groups that exist are as
  distinct as the ensemble's states.
- The reference excludes the clusters being compared, so a wide or
  heterogeneous cluster cannot set its own threshold, and it pools the
  member-to-medoid distances of all other clusters (hundreds of values)
  instead of averaging a few cluster medians.
- The rule measures effect size only. Whether groups exist at all is decided
  before it, by the clusterability test (with family-wise error control in
  refinement) and by Monte Carlo-precise stability; structureless groups are
  not split by these steps (section 8, "Why step 1 is needed").
- The ratio 1 is a convention ("as different as states are wide"), not an
  estimate. Section 8 shows how the decisions depend on it.
- With only two top-level clusters there is no other cluster to define the
  scale, and the pair is not tested. The report then gives the distance
  between the two medoids and the median distance of each cluster's members to
  their own medoid, so that the reader can judge the pair.
- For experimental structures, groups closer than the coordinate error are
  noise whatever the ensemble's scale. `selection.refine_min_separation` sets
  such an absolute floor (in Å), e.g. from the diffraction-component precision
  index (Cruickshank 1999); it is off by default because predicted models and
  MD frames have no defined coordinate error. When RMSD is the metric, the
  report lists every pair of clusters whose medoids are closer than 0.5 Å, the
  upper end of typical coordinate errors of experimental structures.

**What it means for a result.** Clusters are reported as found by the
reproducibility and group-structure tests. Near-identical models from one
protocol next to near-identical models from another can therefore be two
clusters; their ratio in `separation.csv` and the report then shows whether
they are closer than the ensemble's states are wide. With the optional rule,
such sets become one state with two labelled sub-groups; two folds of the same
molecule, or groups several state widths apart, stay separate either way.

### Subsample parameters

A subsample containing 80% of the structures holds about 80% of every cluster.
In each subsample, `min_cluster_size` and `min_samples` are therefore scaled by
the subsample fraction. Without this scaling, clusters close to
`min_cluster_size` could never be found in a subsample, and would look unstable
for purely arithmetic reasons.

### Dependent structures (`input.groups`)

Resampling single structures treats them as independent. Structures of one
source are not: the chain copies and NMR models of one PDB entry, or the models of
one prediction run, are more alike than structures of different sources. Such a
set forms a tight group that is present in every subsample, so it reappears as a
reproducible cluster whether or not it is a state of the molecule (section 12,
"Copies of entries"). The standard remedy for dependent data is to resample the
independent units, not the observations (the cluster bootstrap; Davison &
Hinkley 1997; Field & Welsh 2007); trajectories are already resampled in blocks
for the same reason (section 9).

With `input.groups` (a source per structure), Struct2Clust therefore:
1. draws subsamples of **groups**: `input.group_fraction` (0.5) of the groups, all
   structures of a group together. Half of the data is also the subset size of
   the subsetting scheme in Hennig's implementation and of complementary-pairs
   stability selection (Shah & Samworth 2013, developed for variable selection).
   These are precedents, not a validation of 0.5 for this method; the fraction
   was chosen for the reason in point 2 and its cost was measured in simulations
   (section 12, "Resampling by group").
2. counts a cluster none of whose members is in a subsample as **not reproduced**
   there (Jaccard 0). Without this rule a cluster drawn from one group would be
   found whenever its group is drawn and ignored otherwise, and would look stable.
   With it, a cluster drawn from one group reaches a mean Jaccard of at most about
   0.5, below the threshold of 0.75; a cluster must reappear from independent
   groups, which is the question that matters for dependent data. The idea is
   related to prediction strength (Tibshirani & Walther 2005), which asks whether
   clusters found in one part of the data are predicted in another; the statistic
   here remains Hennig's Jaccard stability, not prediction strength.
3. does not split a cluster in refinement whose members come from fewer than 8
   groups, and does not run with fewer than 8 groups in all: half of 8 groups
   gives 70 different subsamples, half of 4 only 6. The value 8 is an
   operational safeguard, not a validated optimum; its sensitivity is examined
   in section 12 ("Resampling by group").
   A cluster that is not
   split for this reason is reported with that reason; this is not evidence
   that it has no sub-structure.

Everything else is unchanged: the candidates, the sequential test and its
guarantee (which now concern subsamples of groups), clusterability, refinement
and the effect size. Without `input.groups` every structure is its own group and
the method is exactly as described above. The cost is power: the effective
sample size is the number of groups, so few groups support few or no clusters.

## 7. Cluster-wise stability, tiers and membership

The selected clustering is re-assessed with 200 subsamples
(`stability.n_resamples`), following Hennig (2007):
- Each subsample holds 80% of the structures, drawn without replacement. The
  bootstrap is not used, because duplicated points create artificial density for
  HDBSCAN.
- The subsample is clustered with the same, scaled parameters.
- For every cluster C, the Jaccard similarity between C (restricted to the
  subsample) and its best-matching subsample cluster is recorded.

| Mean Jaccard | Hennig's interpretation | Tier |
|---|---|---|
| ≥ 0.75 | stable cluster | HIGH |
| 0.60 - 0.75 | a pattern, but membership doubtful | MODERATE |
| < 0.60 | should not be trusted | LOW |

The tier is then limited by the clusterability verdict (section 4). A subsample
in which a cluster's Jaccard is ≤ 0.5 counts as "dissolving" it; the fraction
of such subsamples is reported.

**Selection decision, estimates and tier.** Every cluster carries four values
(`cluster_summary.csv`, OUTPUTS.md), which answer different questions:
- `selection_decision`: whether the resolution selection established that the
  cluster's mean Jaccard is at least 0.75, at the stated Monte Carlo risk
  (`established`), or no reproducible partition was found (`not_established`).
- `selection_estimate`: the mean Jaccard from the selection's own subsamples. The
  sequential test stops as soon as the decision is established, so this estimate
  can rest on few subsamples.
- `independent_stability_estimate`: the mean Jaccard from 200 fresh subsamples,
  with the 2.5% and 97.5% quantiles of the single-subsample values. It is not
  biased towards the candidate that was chosen; like every resampling estimate,
  it uses the same data.
- `stability_tier`: Hennig's band of the independent estimate, capped by the
  clusterability verdict. It describes the independent estimate; it is not a
  second decision.

A cluster whose true mean Jaccard is close to 0.75 can be `established` and have an
independent estimate just below 0.75 (tier MODERATE): the decision and the
estimate come from different subsamples. The report notes such clusters.

The same subsamples give every structure a **membership probability**: the
fraction of subsamples containing the structure in which it lies inside the
best match of its own cluster. For noise structures, the probability for the
closest cluster is reported. It is a frequency over subsamples, not a
calibrated probability that the structure belongs to a state.

### Overall conclusion

Every run ends with one of seven conclusions (`result.conclusion` in
`run_summary.json`, first line of `report.md`):

| Conclusion | Condition | Meaning |
|---|---|---|
| `distinct_states` | reproducible partition, CLUSTERABLE | the ensemble consists of distinct states |
| `partition_not_confirmed` | reproducible partition, WEAK (or clusterability off) | the clusters reappear in subsamples, but group structure was not confirmed: the data may be a continuum cut consistently, or states of unequal size or many states, for which the dip test has little power (section 4); treat them as reproducible regions of the ensemble |
| `no_group_structure` | reproducible partition, NOT_CLUSTERABLE | no group structure, so no evidence for more than one state; partition shown for inspection only |
| `no_reproducible_partition` | no partition with all clusters reproducible | no reproducible structure; partition exploratory |
| `no_clusters` | no setting gives two or more clusters | no evidence for more than one state; every structure is noise |
| `insufficient_clusters` | one of the first three conditions, but fewer than two clusters remain after the effect-size rule and refinement | the groups found are closer than the separation threshold, so the final partition does not support distinct states; this is not evidence for one state |
| `insufficient_sampling` | trajectories only: any condition except `no_clusters`, but the sampling check failed (section 9) | the blocks cannot be treated as independent, so whether the partition is reproducible cannot be assessed from the data; the partition is exploratory, and this is evidence neither for nor against states |

The first three conditions are evaluated on the partition of the resolution
selection; the effect-size rule (section 6) and refinement (section 8) come
after them. A conclusion about several clusters is therefore given only if at
least two clusters remain in the reported partition; otherwise the conclusion
is `insufficient_clusters`, and the conclusion the selection's partition alone
would give is kept in `result.conclusion_before_effect_size` as a diagnostic,
not as the conclusion of the run. With default settings no clusters are
merged (section 6), so `insufficient_clusters` arises only when the optional
effect-size rule or an absolute floor (`selection.refine_min_separation`) is
used. For trajectories whose sampling check fails (section 9), the conclusion is
`insufficient_sampling`, and the conclusion the resampling would give with
independent blocks is kept in `result.conclusion_if_sampling_sufficient`, again
as a diagnostic. This separates a partition that is not reproducible from one
whose reproducibility cannot be measured.

Stability and clusterability answer different questions, so the conclusion
needs both. No conclusion states that the ensemble *is* one state: absence of
evidence for several states is not evidence for one, which would need an
equivalence margin set by the field (Schuirmann 1987). What counts as a "true" cluster depends on the aim of the analysis
(Hennig 2015); the conclusion states which claim the evidence supports.

**Why stability rather than a significance test.** Stability asks "would this
cluster appear again?". That question can be answered from the data without
circularity. A significance test asks "is this cluster better than chance?".
When the same data both define and test the clusters, that question requires
selective inference (Gao et al. 2024). Naive tests reuse the data that defined
the clusters; Gao et al. (2024) show that such tests do not control the type I
error, because the clusters were chosen to differ.

## 8. Hierarchical refinement

**The problem.** Excess-of-mass extraction compares the persistence of a group
with the persistence of its parts. When a few groups are far apart, splitting
one of them never pays off, so every candidate setting returns the same few
dominant groups. The states inside them stay hidden, even when they are well
separated. In a synthetic control with one far-away group and three groups 6σ
apart, every setting gives 2 clusters.

**The rule.** Every selected cluster is treated as a data set of its own. The
tests start at the top-level clusters and go down; a cluster that is not split
is not examined further.
1. **Group structure.** Its members are tested for clusterability (section 4).
   Only CLUSTERABLE members can be split. The dip test of a cluster with N_j of
   the N structures uses the significance level α · (N_j − 1) / (N − 1). This
   controls the family-wise error rate of the whole sequence of tests
   (Meinshausen 2008), as in significance testing for hierarchical clustering
   (Kimes et al. 2017). Because dip-test p-values on pairwise distances are
   approximate (section 4), this guarantee is nominal; the error rates actually
   reached are measured in section 12.
2. **Resolution.** The resolution selection (section 6) is run on its members
   only, with the same rule and the same Monte Carlo precision. A partition
   qualifies if every sub-cluster is reproducible (mean Jaccard ≥ 0.75),
   decided at the stated resampling risk ("Monte Carlo precision of the
   decision" in section 6). The same rule at every level means that a
   sub-cluster needs exactly the evidence a top-level cluster needs.
3. **Effect size (optional, off by default).** With the optional rule,
   sub-clusters closer than the state scale are merged ("Effect size" in
   section 6; the state scale is that of the top-level clusters other than the
   parent). If one group remains, the cluster is kept and its sub-groups are
   reported as sub-structure; otherwise it is split into the remaining groups.
   A merged group takes the stability statistics of its least stable part and
   the summed membership probability of its parts. Without the rule, the
   effect size of every pair of final clusters is reported (section 6).
4. Steps 1-3 are repeated for every sub-cluster until nothing splits.

**Stability of sub-clusters** is measured within the parent: the parent's
members are subsampled and re-clustered with the sub-cluster's settings.
- The tier of a sub-cluster is capped by the stricter of the data-set verdict
  and the parent-level verdict.
- Its membership probability is the parent membership times the membership
  within the parent.
- Members of a split cluster that fall in none of its sub-clusters become
  noise. Their `top_level` still names the top-level cluster they belong to
  (OUTPUTS.md).
- Every cluster has one number. Sub-clusters of one top-level cluster follow
  one another and share the column `top_level`; the report says which clusters
  were separated by refinement.

**Why step 1 is needed.** Reproducible sub-partitions also exist inside
structureless clusters, because a curved or elongated cloud can be cut in the
same place in every subsample (Ben-David et al. 2006). Two groups 40σ apart,
each without inner structure, were refined with and without step 1. Without
step 1: 10 samples per shape, run with the optional effect-size rule at ratio 1
(`validation/run_study.py --without-gate`; table written to `validation/results/without_gate/`). With step 1: 50 samples per shape,
default settings (section 12):

| Inner shape of the groups | Split without step 1 | Split with step 1 |
|---|---|---|
| Gaussian, uniform, one dominant group | 0/30 (0-12%) | 0/150 (0-2%) |
| Density gradient | 2/10 (3-56%) | 0/50 (0-7%) |
| Elongated (5:1) | 4/10 (12-74%) | 0/50 (0-7%) |
| Curved | 7/10 (35-93%) | 5/50 (3-22%) |

Step 1 reduces these false splits but does not remove them for a curved cloud.
The two columns differ in sample size and in the effect-size setting, so they
show the size of the effect, not a test of it. Two drifting MD trajectories
that never meet were split in none of 50 samples (0-7%; section 12). Whether the optional
effect-size rule would have merged the five false splits of the curved cloud
has not been tested.

**Cost.** Step 1 and the FWER level make refinement conservative: hidden
sub-groups 4σ apart were recovered in 2 of 50 samples (0-14%), 5σ apart in 36
of 50 (58-84%) and 6σ apart in 50 of 50 (93-100%; section 12). Missing a weak
split is preferred to
reporting a false one, because the sequence of tests then has a stated error
guarantee (Meinshausen 2008).

**The optional effect-size rule (step 3) in simulation.** Two tight sub-groups (SD 0.1 per
dimension) were placed inside one group, with medoids f times the state scale
apart, next to one ordinary Gaussian group (5 samples each; numbers are the
clusters reported, out of 3 groups present; script
[`validation/effect_size_sensitivity.py`](../validation/effect_size_sensitivity.py)):

| Medoid distance f | Ratio 0.5 | Ratio 1 | Ratio 2 |
|---|---|---|---|
| 0.5 × state scale | 2 in 3 samples, 3 in 2 | 2 (one state, sub-structure) | 2 |
| 0.8 × state scale | 3 | 2 (one state, sub-structure) | 2 |
| 1.5 × state scale | 3 | 3 | 2 |

With ratio 1 the rule decides as designed: groups closer than the state
scale are one state, groups farther apart are two. Halving or doubling the
ratio moves the boundary accordingly; the result depends on the convention
only for groups between 0.5 and 2 state widths apart, and the report lists
the medoid distances and ratios so that a reader can apply a different one. In
all of these samples the selection found the two sub-groups as reproducible
clusters; the rule then merged them, labelling the parts as sub-structure.

## 9. Trajectories

Neighbouring MD frames are correlated. Drawing single frames would make
subsamples look more alike than independent data would. Resampling contiguous
blocks keeps the dependence within each block, the standard approach for
dependent data (Carlstein 1986; Künsch 1989).
- Each trajectory is therefore split into contiguous blocks of frames. By
  default (`trajectory_blocks: null`) a trajectory gets as many blocks as it
  holds independent frames, n/τ (τ below), i.e. blocks of about one
  autocorrelation time, with at least 8 and at most 50 blocks. Resampling only as
  many units as there are independent samples is the remedy for correlated data
  given by Grossfield et al. (2018, section 7.6). Above 50 blocks, further blocks
  add resampling units but no independence.
- Subsamples keep or drop whole blocks.
- A block never spans two trajectories.
- These blocks are resampled like groups of dependent structures (section 6,
  "Dependent structures"): every subsample keeps `input.group_fraction` (half) of
  the blocks, and a cluster none of whose frames is in a subsample counts as not
  reproduced there (Jaccard 0) instead of being left out. A cluster is defined by
  the clustering of all frames; whether it is found again is assessed in every
  subsample, and subsamples that miss it are not removed from its mean. Without
  this, a state visited once, within one block, would be either wholly present
  or wholly absent and would look perfectly reproducible whenever present.
- A cluster whose frames come from fewer than 8 blocks is not split in
  refinement. The 8 blocks (as for groups) are a pre-specified quality-control
  rule, not a statistical threshold (half of 8 blocks gives 70 different
  subsamples, half of 4 only 6), and 8 blocks are not by themselves enough to
  claim convergence.
- A number in `trajectory_blocks` fixes the blocks of every trajectory; they are
  resampled like structures (`subsample_fraction` of the blocks, clusters absent
  from a subsample left out). The sampling check below applies to these blocks as well: fixed blocks shorter than
  τ fail it, and the conclusion is then `insufficient_sampling` although the
  partition is the same.

**Autocorrelation.** Blocks only remove the correlation between frames if they
are longer than the correlation time. τ is defined here as the integrated
autocorrelation time, τ = 1 + 2 Σ ρ(t), the factor by which correlation reduces
the number of independent frames (n/τ), of one observable: the first principal
coordinate of the trajectory's distance matrix (classical scaling; Gower 1966),
the one-dimensional coordinate that preserves most of the pairwise distances and
so follows the largest conformational change that the clustering distance sees.
Struct2Clust estimates it with Sokal's automatic window (Sokal 1997;
window factor 6, within the usual range of 6-10). The relative statistical error
of τ is about √(2(2W + 1)/n) for a window of W frames (Madras & Sokal 1988;
Wolff 2004). The report and `run_summary.json` give, per trajectory, τ with its
relative error, the effective number of independent frames n/τ (Lyman &
Zuckerman 2007) and the block length. Because only one coordinate is followed,
slower motions along other coordinates are not seen: τ is a lower bound on the
slowest correlation time. Two notes are added where they apply:
- the relative error of τ exceeds 50% (a reporting choice): the trajectory is
  short compared with its correlation time;
- a block is shorter than τ, i.e. the trajectory holds fewer independent frames
  than there are blocks. With the automatic rule this happens only when the
  trajectory holds fewer than 8 independent frames. Neighbouring blocks are then
  correlated, and stability and populations are less certain than they appear;
  more sampling helps.

Because τ is a lower bound and uncertain (the relative error above), blocks of
one τ remove most, not necessarily all, of the correlation between blocks; in
particular they do not guarantee independence when slow conformational
transitions are rare.

**Sampling check.** The observables whose stability and populations are
reported are the cluster memberships. After clustering, the integrated
autocorrelation time of the indicator "frame in cluster c" is estimated for every
cluster along every trajectory, with Geyer's initial positive sequence estimator
(Geyer 1992), which follows a slowly decaying correlation to its end (a window
that grows with the estimate stops too early for a state visited only a few
times). It is reported per cluster (`membership_autocorrelation_time`, the
largest over the trajectories). The sampling check fails if
- a trajectory holds fewer independent frames than blocks (blocks shorter than
  τ; with the automatic rule, fewer than 8 independent frames), or
- a cluster's membership stays correlated for longer than one block in some
  trajectory: its visits are then too few and too long for the blocks to be
  independent with respect to that cluster, or
- no trajectory enters or leaves a cluster (noise frames aside): the cluster
  then consists of whole trajectories. Its blocks come from one simulation (or
  a few), so every subsample that holds any of them finds it, and resampling
  blocks only shows that a trajectory resembles itself. The independent unit of
  such a cluster is the trajectory, and one trajectory per cluster cannot show
  that the cluster is reproducible. The membership autocorrelation time is not
  defined for a trajectory that never leaves a cluster, so the second rule
  cannot catch this case. Without this rule, two replicas of one structureless
  drifting process that never meet were reported as two distinct states with
  HIGH tiers.

All three are pre-specified quality-control rules.

**Blocks of one membership autocorrelation time (`input.membership_blocks`,
on by default).** The blocks are set before clustering, from the geometry; the
memberships of the clusters found can stay correlated for longer, because the
first principal coordinate mixes the switching between states with faster
motion. The check then fails on the second rule although the trajectory may
hold enough independent visits. A run whose
check failed on that rule alone (blocks at least one geometric τ long, every
cluster entered and left), and whose trajectories are each at least 8 times as
long as the longest membership autocorrelation time in them, is resampled once
more: every trajectory in n / τ_m blocks (τ_m the largest of its geometric and
membership autocorrelation times; 8-50 blocks), with selection, stability,
refinement and the sampling check repeated on the new blocks. The run reports this
second pass, and keeps the first in `result.first_pass` and in the report. The
blocks of the second pass depend on the clustering of the first, so this is a
two-stage procedure whose error rates are those measured for it in the
simulation study (section 12, "Trajectories"): on three structureless
trajectory scenarios it reported no state in 150 samples, as the single pass,
while on two-state trajectories it raised `distinct_states` from 11 to 125 of
200 samples. `input.membership_blocks: false` keeps the single pass.

**Comparing conditions.** Simulations of different conditions (for example a
molecule bound and free) that each stay in their own cluster end as
`insufficient_sampling`. The populations per trajectory (`populations.csv`) still
show that the runs differ; whether the difference is reproducible needs several
independent simulations per condition, each of which visits its cluster. A failed check gives the
conclusion `insufficient_sampling` (section 7) with the reasons in the log, the
report and `result.sampling_check`; a passed check does not establish
convergence. Whether block averages have reached a plateau as the blocks grow
(blocking analysis; Flyvbjerg & Petersen 1989) needs many more blocks than a
short trajectory provides and is not decided by the run;
`validation/block_sensitivity.py` reports it, together with the clustering at
other block numbers, for a finished run.

A trajectory with only a few independent frames cannot support state
populations, whatever the resampling.

With several trajectories, the output also reports:
- every cluster's population in every trajectory,
- a population heatmap across trajectories,
- a timeline of the cluster of every frame (up to four trajectories).

**Large trajectories.** The distance matrix holds N² values (0.8 GB for 10,000
frames), a run needs several times that (section 13, "Memory"), and every
candidate setting is clustered on many subsamples,
so the computing time grows faster than N². Neighbouring frames of a
well-sampled trajectory carry little independent information, so long
trajectories are best thinned with `input.stride` or `input.max_frames`
(CONFIGURATION.md) to a few thousand frames in total; `max_frames` also gives
every trajectory the same weight.

**Population uncertainty.** The population of a cluster (its fraction of the
frames) is given with an approximate 95% interval from a block bootstrap (Künsch
1989). The blocks are about three autocorrelation times long (n/(3τ) blocks per
trajectory, at least 8 and at most 50; the blocks given in
`input.trajectory_blocks` if fixed), where τ is the longer of the geometric
autocorrelation time and the longest membership autocorrelation time of the final
clusters in that trajectory (section 9, sampling check). A population is the mean
of a membership indicator, so the correlation of the memberships, not of the
geometry, sets how many independent blocks a trajectory holds. The blocks are
longer than the resampling blocks of the clustering, because blocks of one τ gave
intervals that were too narrow (below).
The blocks of every trajectory are drawn with replacement, as many as the
trajectory has, and the populations recomputed 2000 times; the 0.05% and 99.95%
quantiles give the interval (`fraction_low`, `fraction_high`; `populations.csv`).
This nominal 99.9% level was chosen because nominal 95% intervals covered the
true population too rarely; at 99.9% the coverage was about 95% (below), and the
intervals are reported as approximate 95% intervals.
The partition is held fixed: the interval describes how much the populations of
these clusters depend on the frames that were sampled, not the uncertainty of the
clustering. With few blocks the interval is approximate; bootstrap intervals are
not quantitatively reliable for small samples (Grossfield et al. 2018).

**How approximate.** The intervals and the agreement flag below were checked on
stationary two-state Markov chains with known populations (6-D, states 6 units
apart, correlated noise; 200 samples per row; the true states used as clusters,
so only the interval procedure is tested). First the block length was compared
at the nominal 95% level
([`validation/population_coverage.py`](../validation/population_coverage.py)):

| Frames | Autocorrelation time of the states | Blocks of 1 τ: mean blocks, coverage, flag | Blocks of 3 τ (PC1): mean blocks, coverage, flag |
|---|---|---|---|
| 1000 | 19 | 46, 88%, 12% | 23, 92%, 8% |
| 1000 | 49 | 34, 79%, 20% | 13, 89%, 14% |
| 1000 | 79 (populations 0.2 / 0.8) | 32, 73%, 23% | 12, 82%, 15% |
| 1000 | 99 | 27, 74%, 24% | 10, 84%, 20% |
| 2000 | 99 | 35, 83%, 16% | 13, 90%, 12% |
| 4000 | 99 | 45, 82%, 9% | 21, 88%, 9% |

Coverage: share of samples whose interval contains the true population (nominal
95%). Flag: share of samples whose halves were flagged as differing (nominal 5%).
With blocks of one τ the intervals were too narrow and the flag was raised too
often, because τ of the first principal coordinate underestimates that of the
state memberships (by 14-45% here) and because blocks of one τ leave correlation
between neighbouring blocks. Blocks of two τ, and of one or two membership τ,
were also compared; no block length reached the nominal values, because with few
transitions (5-20 in a trajectory here) the number of independent blocks, not
their length, limits the interval.

The nominal level was then chosen on the rule used (three times the longer of
the geometric and the membership τ;
[`validation/population_calibration.py`](../validation/population_calibration.py),
eight chains, 200 samples each, nominal levels 95%, 97.5%, 99%, 99.5% and 99.9%).
The rule, fixed before the results were seen, was: the smallest level whose
coverage is at least 93% in every chain and at least 95% on average, with the
halves flagged in at most 5% of samples on average (flag at the same level
here); if no level qualifies, 99.9%,
and the intervals stay labelled approximate.

| Frames | Transition probabilities | Mean blocks | Nominal 95%: coverage, flag, width | Nominal 99.9% (used for the intervals): coverage, flag, width |
|---|---|---|---|---|
| 1000 | 0.005 / 0.02 (populations 0.2 / 0.8) | 9 | 83%, 15%, 0.29 | 92%, 6%, 0.44 |
| 1000 | 0.01 | 8 | 85%, 20%, 0.43 | 98%, 9%, 0.66 |
| 1000 | 0.02 | 9 | 89%, 16%, 0.35 | 96%, 6%, 0.54 |
| 1000 | 0.05 | 17 | 93%, 10%, 0.24 | 99.5%, 2%, 0.39 |
| 1000 | 0.1 | 34 | 93%, 9%, 0.17 | 98%, 2%, 0.28 |
| 2000 | 0.005 | 8 | 85%, 19%, 0.44 | 95%, 11%, 0.66 |
| 2000 | 0.01 | 9 | 89%, 15%, 0.36 | 97%, 7%, 0.56 |
| 4000 | 0.01 | 14 | 89%, 9%, 0.27 | 99%, 3%, 0.42 |
| Mean | | 13 | 88%, 14%, 0.32 | 97%, 5.4%, 0.49 |

No level met the rule: at 99.9% the mean coverage was 97%, but one chain (about
5-10 transitions) reached 92% and the halves were flagged in 5.4% of samples. The
pipeline therefore uses 99.9% for the intervals; the report gives them as
approximate 95% intervals, together with the measured coverage (92-99.5%), not as
99.9% intervals. They are about 1.5 times as wide as nominal 95% intervals. With
fewer transitions than in these chains they are less reliable still. The
comparison of the parts below keeps the nominal 95% level: at 99.9% (Bonferroni
over clusters and pairs) it missed two trajectories of 40 frames, 8 blocks each,
that sampled different states, because a block bootstrap of 8 blocks leaves out
all blocks of a state too often for such extreme quantiles; for a warning about
convergence, a missed difference is worse than a false alarm.

**Agreement between trajectories.** Clustering all trajectories together and
comparing the populations of every cluster between them is a standard check of
convergence (Lyman & Zuckerman 2006; Grossfield et al. 2018, section 4.4). The
parts compared are the trajectories, or the first and second half (by blocks) of
a single trajectory. For every pair of parts and every cluster, the difference of
the populations gets a bootstrap interval, each part resampled on its own blocks,
with Bonferroni adjustment over clusters and pairs (nominal 95% family-wise). A
cluster whose interval excludes zero is reported, in the log and the report, as
differing between the parts by more than their approximate sampling uncertainty
(`population_differences.csv`); in the simulations above, stationary trajectories
were flagged in 8-20% of samples, so a flag is a warning, not a test at the 5%
level.

**Interpreting populations.** Cluster populations are equilibrium populations
only if the trajectories have converged. Three signs indicate that they have not:
- populations that differ between trajectories (or halves) by more than their
  sampling uncertainty (above),
- replicas that occupy disjoint sets of clusters,
- timelines in which states are left and never revisited.

Agreement does not prove convergence: states that no trajectory has visited are
not seen, and parts that share a starting structure can agree for that reason.

In that case the clusters describe phases of the simulations, not an
equilibrium distribution. Geometric clusters are regions of conformational
space; kinetic states (metastable sets) require a kinetic model such as a Markov
state model (Prinz et al. 2011).

## 10. Descriptive characterisation

These comparisons describe clusters that were derived from the same structures.
They therefore report effect sizes, not p-values: testing differences between
groups chosen to be different would be circular (Gao et al. 2024). Noise
structures are excluded.

- **Where the clusters differ.**
  - Every pair of cluster medoids is superposed as for the distance itself: by
    least squares on the atoms used for clustering for RMSD, and for
    `--metric tm` by the superposition that maximises the TM-score on C3′/CA
    (the search of Zhang & Skolnick 2004).
  - The squared RMSD is the mean of the squared atomic deviations, so it splits
    exactly into the contributions of the residues. A residue's share is its
    part of the squared RMSD of the pair.
  - The report names the fewest residues that carry at least half of the
    squared deviation, averaged over the pairs of clusters, and does the same
    for every pair. `medoid_difference.csv` gives every residue's share and
    RMSD for every pair.
  - A few residues with a large share mean that the clusters differ locally,
    for example in a flexible terminus or loop, while the rest of the molecule
    is alike. RMSD squares the deviations, so a few strongly displaced residues
    can dominate the distance between whole structures. Whether such a region
    should take part in the comparison is a scientific choice; to leave it out,
    give the residues of interest in `input.residues`. When at most 10% of the
    residues carry half of the squared deviation (a reporting choice), the report
    says so and suggests a run without them. For example, the four clusters of the
    ubiquitin example (PDB 2K39) differ mainly in the flexible C-terminal residues
    74-76; restricted to residues 1-70 (`input.residues: "1-70"`), the same models
    are NOT_CLUSTERABLE and show no reproducible partition.

The following analyses use every heavy atom, independent of the atom subset
used for clustering.

- **RMSF.**
  - The members of each cluster are iteratively superposed onto their mean
    structure, starting from the medoid.
  - Per-residue RMSF is the root-mean-square deviation from the mean, averaged
    over the residue's heavy atoms.
- **Contacts.**
  - Two residues are in contact when any of their heavy atoms are within
    4.5 Å. Residues adjacent in sequence are excluded (|i − j| ≥ 2).
  - The frequency of each contact within each cluster is reported.
- **Torsions.**
  - Nucleic acids: α, β, γ, δ, ε, ζ and χ (IUPAC-IUB 1983). Proteins: φ and ψ
    (IUPAC-IUB 1970).
  - For every cluster, the circular mean and circular standard deviation are
    reported (Fisher 1993).
  - How strongly a torsion separates clusters is given by the size-weighted mean
    resultant length within clusters minus the resultant length of all angles
    pooled. This is the circular analogue of the between-group share of
    variance, and the quantity on which the Watson-Williams comparison of
    circular means is based (Mardia & Jupp 2000). It is high only when each
    cluster is internally consistent and the clusters differ.
  - A linear test on angles would treat −179° and +179° as far apart.
  - A torsion that involves a neighbouring residue (α, ε, ζ, φ, ψ) is computed
    only when the two residues are covalently linked (C-N or O3′-P shorter than
    `biology.bond_max_distance`, 2 Å; the bonds are about 1.33 and 1.61 Å), so
    no angle spans residues missing from the structures.
- **Radius of gyration.** Computed for every structure and summarised per
  cluster.

### Composition by source

`struct2clust composition` compares the clusters of a finished run with a label
of every structure that the clustering did not use: the prediction method, the
PDB entry, a bound ligand, the simulation. It reports how many structures of
each source every cluster holds, where the structures of each source went, and
the agreement between clusters and sources, with noise and structures without
a source excluded:
- Cramér's V (Cramér 1946), and its bias-corrected form (Bergsma 2013). The
  classical estimate is biased upwards in small samples and large tables; the
  correction reduces, but does not remove, that bias. The share of cells with an
  expected count below 5 is reported with them as a measure of sparseness.
- The adjusted Rand index (Hubert & Arabie 1985) and the adjusted mutual
  information (Vinh et al. 2010).

The values are descriptive; the structures of one source are not independent
samples, so no test is performed and no p-values are given.

Agreement has two readings. Clusters that follow known biology (bound and free,
agonist and antagonist) support the clusters as conformational states.
Clusters that follow how the structures were made (prediction tool,
laboratory, processing) point to technical differences (Leek et al. 2010).
Which reading applies depends on what the sources are, not on the size of the
agreement.

## 11. Visualisation

The 2-D map in `overview` is a UMAP embedding (McInnes et al. 2018) of the
distance matrix, with n_neighbors = 15 and min_dist = 0.5.
- **Trustworthiness.** How faithfully the map preserves each structure's k = 10
  nearest neighbours (Venna & Kaski 2001) is reported in `run_summary.json`.
- **Limits of the map.** UMAP preserves local neighbourhoods. It does not
  preserve distances between clusters or their sizes (Chari & Pachter 2023). The
  map is an illustration; the clustering is done on the full distance matrix.
- **Superposition.** The medoids (`medoids/`) and the members of every cluster
  (`cluster_views/`) are superposed as for the distance: by least squares on
  the clustering atoms for RMSD, by TM-score superposition on C3′/CA for
  `--metric tm`.
- **Colour.** Cluster identity never relies on colour alone. Clusters are
  numbered in the map, and elsewhere they are identified by their position on an
  axis. The palette was checked for colour-vision deficiency.

## 12. Validation

### Test suite

The test suite checks the method on synthetic data with known answers. It runs
on every change.

| Control | Expected | Result |
|---|---|---|
| Gaussian noise (2 seeds), uniform noise | no reliable cluster | NOT_CLUSTERABLE, no partition reproducible, all tiers LOW |
| Three separated groups | 3 clusters, all HIGH | ARI to truth > 0.95, all HIGH |
| Two super-groups of three sub-groups | 6 clusters, not 2 | 6 clusters, ARI to truth > 0.9 |
| Three groups on a uniform background | 3 clusters, background as noise | 3 clusters, no split |
| One far-away group + three groups 6σ apart | 4 clusters, not 2 | 2 at the top level, 4 after refinement, ARI > 0.95 |
| Three separated groups, refined | no split | no split |
| Four disconnected groups | eigengap k = 4 | 4 |
| Two trajectories: one shared state, one private state each | shared state populated in both | recovered |
| Two tight sub-groups, medoids much closer than the scatter of the other state | by default two clusters, ratio < 1 reported; with the optional rule, one state with two labelled sub-groups | as expected, also inside a refined cluster |
| Two clusters 0.3 Å apart | no state scale; within coordinate error | both noted in `run_summary.json` and the report |
| Two close sub-groups and a distant one inside one cluster | the close pair is one sub-cluster, the distant group another | split into 2; only the parts of the pair carry `substructure` labels |
| Sequential test on values whose mean is exactly the threshold, looking after every value | wrong decision in at most α of the runs | at most α (400 runs, α = 0.05) |
| Near-duplicates given in three different orders | the same structures kept | the same |
| A gap of missing residues | no torsion across the gap | none |
| AR(1) series with known autocorrelation time 19; independent frames | 19; 1 | within 20% |
| A state switching every few frames; a state visited twice for 100 frames each | membership autocorrelation time short; longer than a block of 25 frames, sampling check fails | as expected |
| Two trajectories, each staying in its own cluster | no trajectory enters or leaves either cluster, sampling check fails | as expected |
| Two trajectories with the same cluster populations; two trajectories in different clusters; one trajectory whose halves are in different clusters | intervals contain the observed populations; populations agree; populations differ | as expected |
| One tight group | no clusters, report written | `no_clusters`, report and tables written |

### Simulation study

Error rates and detection rates were measured on 36 synthetic scenarios with
known answers (6 dimensions, Euclidean distances, default settings), with 50
independent samples each (1800 runs), with the method as described here (Python
3.12, numpy 2.5, scipy 1.18, scikit-learn 1.9). Scenarios with replica labels
are clustered as trajectories (section 9). The analyses of the following
subsections (resampling by group, minimum number of groups, Monte Carlo
reproducibility) and those cited in sections 4, 8 and 9 were run with the same
version. The scripts are in [`validation/`](../validation/), the result tables
behind the numbers of this document in
[`validation/results/`](../validation/results/); the per-run records are written
when a study is run and are not part of the repository. Every rate is given as a count with its 95%
Clopper-Pearson interval (Clopper & Pearson 1934). With 50 samples, an observed
rate of 0% is compatible with a true rate of up to 7% and a rate of 100% with
one of 93% or more. Whether two intervals overlap is not a test of the
difference between the two rates; the intervals describe each rate on its own.

**One structureless group** (300 points). Nothing should be reported as
distinct states.

| Shape | Reproducible partition | `distinct_states` | `partition_not_confirmed` | Any HIGH cluster |
|---|---|---|---|---|
| Gaussian | 0/50 (0-7%) | 0/50 (0-7%) | 0/50 (0-7%) | 0/50 (0-7%) |
| Uniform | 0/50 (0-7%) | 0/50 (0-7%) | 0/50 (0-7%) | 0/50 (0-7%) |
| Elongated (5:1) | 4/50 (2-19%) | 0/50 (0-7%) | 4/50 (2-19%) | 0/50 (0-7%) |
| Density gradient | 5/50 (3-22%) | 0/50 (0-7%) | 5/50 (3-22%) | 0/50 (0-7%) |
| Curved | 13/50 (15-40%) | 2/50 (0-14%) | 11/50 (12-36%) | 4/50 (2-19%) |

A reproducible partition is either conclusion, `distinct_states` or
`partition_not_confirmed`. For the elongated and density-gradient clouds
together it occurred in 9 of 100 samples (4-16%).

**Several groups** (100 points each). The groups should be found.

| Groups | `distinct_states` | `partition_not_confirmed` | Clusters found (mean) | ARI to truth |
|---|---|---|---|---|
| 2 groups, 6σ | 48/50 (86-100%) | 2/50 (0-14%) | 2.0 | 0.89 |
| 3 groups, 6σ | 50/50 (93-100%) | 0/50 (0-7%) | 3.0 | 0.87 |
| 3 groups, 5σ | 47/50 (83-99%) | 0/50 (0-7%) | 2.8 | 0.57 |
| 3 elongated groups, 6σ | 35/50 (55-82%) | 11/50 (12-36%) | 2.9 | 0.62 |
| 3 groups, 4σ | 1/50 (0-11%) | 19/50 (25-53%) | 2.2 | 0.19 |

ARI counts noise as its own label; the overlap between groups 4-5σ apart is
mostly labelled noise, which lowers the ARI.

**Unequal and many groups** (6σ apart; two groups of 300 points split 80:20 or
90:10; 4 or 6 groups of 50 points with every pair 6σ apart). The groups should
be found.

| Groups | `distinct_states` | `partition_not_confirmed` | Clusters found (mean) | ARI to truth |
|---|---|---|---|---|
| 2 groups, 80:20 | 18/50 (23-51%) | 32/50 (49-77%) | 2.0 | 0.92 |
| 2 groups, 90:10 | 0/50 (0-7%) | 46/50 (81-98%) | 2.0 | 0.94 |
| 4 equidistant groups | 11/50 (12-36%) | 38/50 (62-87%) | 3.9 | 0.86 |
| 6 equidistant groups | 0/50 (0-7%) | 47/50 (83-99%) | 5.7 | 0.76 |

The groups were found and reproduced, but the test for group structure could
not confirm them (section 4, "Power of the dip test"), so most of these
partitions are reported as `partition_not_confirmed`: the claim is weaker than
the data support, the partition itself is not missed.

**Trajectories** (default settings, with the second pass of section 9; in
brackets: one pass with the geometric blocks, `input.membership_blocks: false`).
A structureless AR(1) drift (500 frames); two replicas of it that never meet
(500 frames each, 40σ apart) or that overlap; two states 6σ apart with AR(1)
noise, 1000 frames, switching about 100 (fast), 20 (medium) or 5 (slow) times,
with populations 80:20 (about 20 switches), or in two replicas of 500 frames
starting in different states.

| Scenario | Should | `distinct_states` | `partition_not_confirmed` | `insufficient_sampling` | ARI |
|---|---|---|---|---|---|
| Drift | no state | 0/50 (0) | 0/50 (0) | 45/50 (49) | - |
| Two replicas, never meeting | no state | 0/50 (0) | 0/50 (0) | 50/50 (50) | - |
| Two replicas, overlapping | no state | 0/50 (0) | 0/50 (0) | 49/50 (50) | - |
| Two states, fast | two states | 49/50 (49) | 1/50 (1) | 0/50 (0) | 0.96 |
| Two states, medium | two states | 49/50 (3) | 0/50 (0) | 0/50 (47) | 0.97 |
| Two states, slow | two states | 18/50 (0) | 6/50 (1) | 22/50 (48) | 0.94 |
| Two states, 80:20 | two states | 26/50 (8) | 21/50 (7) | 0/50 (34) | 0.98 |
| Two states, two replicas | two states | 32/50 (0) | 2/50 (0) | 15/50 (50) | 0.97 |

The remaining samples gave `no_reproducible_partition`. No structureless
trajectory was reported as a state or a reproducible partition, with or without
the second pass (0 of 150; 0-2%). With one pass, the states were found (ARI
0.94-0.98) but mostly reported as `insufficient_sampling`, because the blocks set
from the geometry were shorter than the memberships' correlation; the second
pass ran in 142 of the 200 samples with two states and raised `distinct_states`
from 11 to 125 of 200 (55-69%). Two replicas that never meet are always
`insufficient_sampling`: each cluster is one trajectory (section 9). With few
transitions (slow) many samples remain `insufficient_sampling`: the data do not
hold enough independent visits.

Seven further trajectory scenarios were not used while the second pass was
designed (`traj_heldout_*`, 50 samples each): structureless drift that is faster
(AR(1) φ = 0.95, 1000 frames), slower (φ = 0.995, 2000 frames), anisotropic (φ =
0.999 along one axis) or a slow walk along a curve; three states (every pair 6σ
apart, 1500 frames), two states joined by transition paths of about 10 frames,
and two states 4σ apart (switching probability 0.02 per frame).

| Scenario | Should | `distinct_states` | `partition_not_confirmed` | `insufficient_sampling` |
|---|---|---|---|---|
| Drift, faster | no state | 0/50 (0) | 0/50 (0) | 1/50 (19) |
| Drift, slower | no state | 0/50 (0) | 0/50 (0) | 43/50 (47) |
| Drift, anisotropic | no state | 0/50 (0) | 0/50 (0) | 22/50 (29) |
| Walk along a curve | no state | 0/50 (0) | 1/50 (1) | 19/50 (26) |
| Three states | three states | 40/50 (4) | 0/50 (0) | 2/50 (46) |
| Two states with transition paths | two states | 3/50 (2) | 12/50 (3) | 2/50 (26) |
| Two states, 4σ | two states | 0/50 (0) | 4/50 (1) | 0/50 (18) |

On these structureless trajectories no state was reported (0 of 200; 0-2%) and
one reproducible partition (1 of 200; 0-3%), with and without the second pass.
Three states were found in 40 of 50 samples (ARI 0.91). States joined by
transition paths or only 4σ apart were mostly not resolved (ARI 0.49 and 0.31):
the frames between the states, and the overlap of states 4σ apart, leave no gap
that a density-based clustering can separate.

**Refinement** (two structureless groups 40σ apart; three sub-groups 4-8σ apart
hidden by a far-away group; two tight sub-groups 0.5-3 state scales apart next
to one ordinary group):

| Scenario | Should | Result |
|---|---|---|
| 6 kinds of structureless groups (Gaussian, uniform, one dominant, elongated, curved, density gradient) | not split | 5/300 split (1-4%): curved 5/50 (3-22%), the other five 0/250 (0-1%) |
| Hidden sub-groups 4σ apart | be recovered | 2/50 (0-14%) |
| Hidden sub-groups 5σ apart | be recovered | 36/50 (58-84%) |
| Hidden sub-groups 6σ apart | be recovered | 50/50 (93-100%) |
| Sub-groups 8σ apart | be found | 50/50 (93-100%), at the top level |
| Tight sub-groups 0.5 state scales apart | be one state | 50/50 (93-100%) |
| Tight sub-groups 1.5 or 3 state scales apart | be two states | 100/100 (96-100%) |

The scenarios with tight sub-groups were run with the optional effect-size rule
at ratio 1; all others with default settings, in which the rule is off. The
sub-groups 0.5 state scales apart (SD 0.25 per dimension) were kept together by
the clusterability test in refinement, so the rule had nothing to merge: it did
not act in any of the 1800 runs, and these scenarios do not test it. Its effect
is shown by the separate sensitivity analysis in section 8, in which tighter
sub-groups pass the clusterability test.

**Copies of entries.** Forty points drawn from one structureless Gaussian, each
repeated as six near-identical copies (SD 0.05 per dimension), as when chain
copies or NMR models of one entry enter an ensemble: nothing should be reported
as distinct states. The result was `distinct_states` in 47 of 50 samples
(83-99%) and `partition_not_confirmed` in the other 3, with a median of 40
clusters, one per set of copies. Copies form tight, reproducible groups, and
resampling cannot tell them from states, because every subsample contains
copies of the same entries. A cluster made of the copies of one entry says that
the entry differs from the others, not that a conformational state exists. Read
the entry composition of the clusters (`struct2clust composition`, section 10)
before interpreting such clusters (section 13).

The group-resampling and minimum-group analyses below were designed after the
main simulation study and after the primary analyses had been conducted. They
are post hoc robustness analyses, not prespecified tests.

**Resampling by group** (`input.groups`, section 6, "Dependent structures";
[`validation/group_study.py`](../validation/group_study.py), 20 samples per
scenario, so 0/20 is compatible with a true rate of up to 17%). Four scenarios
with dependent structures, each with the source of every point as its group:
copies (as above: 40 structureless entries × 6 copies; and three groups 6σ apart,
each of 12 entries × 6 copies) and runs (one structureless group from 30 runs of
10 points, each run offset by SD 1 and tight inside, SD 0.3, like the models of
one prediction run; and three groups 6σ apart, each from 10 runs of 10).

| Scenario | Without groups | With groups |
|---|---|---|
| Copies, structureless: `distinct_states` or `partition_not_confirmed` | 20/20 (83-100%) | 0/20 (0-17%) |
| Runs, structureless: `distinct_states` or `partition_not_confirmed` | 18/20 (68-99%) | 0/20 (0-17%) |
| Copies, 3 groups: `distinct_states` | 20/20 (83-100%), but 36 clusters, one per set of copies | 15/20 (51-91%) |
| Runs, 3 groups: `distinct_states` with the 3 groups | 20/20 (83-100%) | 20/20 (83-100%) |

The cost of grouping was measured on scenarios of the main study whose points
were given 30 random sources unrelated to their structure:

| Scenario | Without groups (fraction 0.8) | Without groups, fraction 0.5 | With groups |
|---|---|---|---|
| Gaussian, elongated, curved cloud: any positive conclusion | 9/60 (7-27%) | 0/60 (0-6%) | 0/60 (0-6%) |
| 2 groups, 6σ: `distinct_states` | 20/20 (83-100%) | 20/20 (83-100%) | 20/20 (83-100%) |
| 3 groups, 6σ: `distinct_states` | 20/20 (83-100%) | 17/20 (62-97%) | 17/20 (62-97%) |
| 3 groups, 5σ: `distinct_states` | 18/20 (68-99%) | 5/20 (9-49%) | 4/20 (6-44%) |
| Hidden sub-groups 5σ / 6σ recovered | 16/20, 20/20 | 6/20, 19/20 | 7/20, 18/20 |

Resampling by group removed the false findings caused by copies and runs.
Halving the subsample, not the grouping itself, costs power: without groups,
a fraction of 0.5 gives the same detection rates as grouping and the same lower
rate of false partitions of continua. Halving the subsample did not remove the
clusters of copies (`distinct_states` in 18 of 20 samples); grouping did. The
group fraction of 0.5 is kept because with 0.8 a cluster drawn from a single
group would still be present in about 80% of the subsamples and pass the
threshold of 0.75. Measured: with groups and a fraction of 0.8, the structureless
runs scenario gave `partition_not_confirmed` in 3 of 20 samples (3-38%), with 0.5
in none.

The minimum of 8 groups for refinement (section 6, point 3) was examined
([`validation/min_groups_study.py`](../validation/min_groups_study.py)) with a
cluster of 180 points next to a far-away group, made of 6, 9 or 12 runs (run
offsets SD 0.7, spread inside a run SD 0.7): either one structureless group, or
the same with run offsets SD 1.0 and spread SD 0.5, or three sub-groups 6σ apart.
Refinement was run without a minimum and every node was recorded, so the result
for a minimum T follows from the same runs (20 samples each):

| Scenario | Minimum 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| Structureless, 6-12 runs (120 samples): split | 0/120 (0-3%) | 0/120 | 0/120 | 0/120 |
| Sub-groups from 6 runs: all 3 found | 4/20 (6-44%) | 4/20 | 4/20 | 4/20 |
| Sub-groups from 9 runs: all 3 found | 16/20 (56-94%) | 16/20 | 16/20 | 13/20 (41-85%) |
| Sub-groups from 12 runs: all 3 found | 20/20 (83-100%) | 20/20 | 20/20 | 20/20 |

No structureless cluster was split at any minimum: such clusters were stopped
by the clusterability test or the stability threshold, not by the number of
groups. With sub-groups from 6 runs refinement never established a split even
without a minimum (the 4 of 20 come from the top level, where the partition was
not reproducible), so a minimum
below 8 gained nothing here; a minimum of 10 lost the sub-groups from 9 runs in 3
samples. The minimum of 8 is therefore kept, as an operational safeguard, not as
a validated optimum: in these scenarios it was not what prevented false splits,
so its protective effect was not tested directly. It ensures that the resampling
has enough distinct subsamples to mean anything.

**The defaults are trade-offs, not optimal settings.** A group fraction of 0.5
accepts lower sensitivity (three groups 5σ apart: 4 instead of 18 of 20) in
exchange for not reproducing clusters drawn from single groups. The structure
fraction of 0.8 keeps the behaviour of the method without groups; it does not by
itself deal with dependent structures (copies: 20 of 20 positive with 0.8 and
with 0.5). Resampling the independent units is the standard approach for
dependent data, but bootstrap results depend on whether the chosen units
capture the dependence (Davison & Hinkley 1997; Field & Welsh 2007); the
literature supports the approach, not these particular values.

**Reading the numbers.** A `distinct_states` conclusion was reached in 2 of 250
samples of one structureless group (1%; 0-3%), both curved clouds, and in none
of 150 structureless trajectories. It was reached in 145 of 150 samples with round groups at
least 5σ apart (97%; 92-99%). Of the samples with three groups 4σ apart, 19 of
50 were reported as `partition_not_confirmed` and 1 as `distinct_states`; the
others gave no reproducible partition: the method does not claim what the data
cannot distinguish from a continuum. A stricter dip-test level trades the two
errors against each other: at α = 0.001 one of the two false
`distinct_states` (dip p = 0.005) would not occur, while the other (dip p
below 0.0001) would remain, and elongated groups 6σ apart would be confirmed in
21 instead of 35 of 50 samples.

### Real molecules

Two molecules whose states are known independently of the clustering, clustered
with default settings ([`validation/real_benchmark.py`](../validation/real_benchmark.py)).

**Alanine dipeptide** in water: three 250 ns trajectories saved every 1 ps
(Wehmeyer et al. 2019), clustered together after thinning to the number of frames
per trajectory below. The true state of every frame comes from its backbone
dihedrals (αR, β/C7eq, αL). Struct2Clust leaves out the ACE and NME caps, so only
the five heavy atoms of the alanine are compared and φ is not seen; the answer to
recover is αR against β (αL holds 3% of the frames).

| Frames per trajectory | Conclusion | Clusters | ARI, αR against β | ARI, three states | Sampling check |
|---|---|---|---|---|---|
| 100 | `distinct_states` | 2 | 0.95 | 0.92 | sufficient |
| 300 | `distinct_states` | 2 | 0.99 | 0.95 | sufficient |
| 1000 | `distinct_states` | 2 | 0.93 | 0.89 | sufficient |
| 2000 | `distinct_states` | 2 | 0.94 | 0.90 | sufficient |

**Adenylate kinase** of *E. coli*: ensembles of 200 structures made from the open
(PDB 4AKE) and the closed (PDB 1AKE) form, chain A, heavy atoms common to both,
each structure displaced along the normal modes of an anisotropic network model
(Atilgan et al. 2001) so that its mean RMSD to the starting structure is 1, 2 or
3 Å; 20 samples per scenario. The two forms are 7 Å apart, so the two-state
ensembles are an easy case; the one-state ensembles test whether thermal-like
motion of a single form is reported as states.

| Ensemble | `distinct_states` | `partition_not_confirmed` | ARI to the forms |
|---|---|---|---|
| One form (open or closed; 1, 2 and 3 Å) | 0/120 (0-3%) | 1/120 (0-5%) | - |
| Two forms, 50:50, 80:20 and 90:10 (1 and 2 Å) | 120/120 (97-100%) | 0/120 (0-3%) | 1.00 |

### Monte Carlo reproducibility

Three scenarios close to the stability threshold (a curved cloud without groups,
three groups 4σ apart, and sub-groups 5σ apart hidden by a far-away group), three
samples each, were clustered with eight different random seeds
([`validation/seed_reproducibility.py`](../validation/seed_reproducibility.py)):

| | Samples | Same result for all 8 seeds |
|---|---|---|
| Conclusion (`distinct_states`, `partition_not_confirmed`, …) | 9 | 9 |
| Partition, when a reproducible partition was found | 4 | 4 |
| Partition shown for inspection, when none was found | 5 | 2 |

The conclusion and every reproducible partition are independent of the seed, as
the sequential test promises. When no partition is reproducible, the partition
shown for inspection is the candidate with the highest estimated weakest-cluster
Jaccard; candidates that do not qualify are stopped early, so that estimate, and
with it the partition shown, can change with the seed. It carries no claim.

## 13. Limitations

- **Copies of entries.** Near-identical copies of one entry (chain copies,
  NMR models, repeated depositions) form reproducible clusters that resampling
  cannot tell from states: forty structureless entries with six copies each
  were reported as `distinct_states` in 47 of 50 samples (83-99%; section 12).
  A cluster drawn from the copies of a single entry is a difference between
  entries, not evidence of a conformational state. Give the source of every
  structure (`input.groups`, section 6, "Dependent structures"), so that
  subsamples are drawn by source (with groups: 0 of 20 samples; section 12,
  "Resampling by group"); otherwise check the entry composition of every
  cluster (`struct2clust composition`, section 10).
- **Resampling by group costs power.** With half of the groups per subsample,
  three groups 5σ apart were reported as `distinct_states` in 4 of 20 samples
  (6-44%) instead of 18 of 20 (section 12). Few groups support few clusters,
  and a cluster drawn from only a few groups has a wide stability interval.
  Groups also remove only the dependence they describe: models of different
  prediction runs of the same tool share the tool's model and template, so a
  cluster from one tool can still be a signature of that tool. Check the tool
  composition of every cluster (`struct2clust composition`, section 10).
  One group column describes one level of dependence; shared templates, related
  model families or nested sources (runs within tools) are not all captured by
  it. `input.groups` reduces spurious stability from the dependence it is given;
  it is not a general remedy for dependent input, and the group fraction and the
  minimum of 8 groups are trade-offs examined only in the simulations of
  section 12.
- **Curved continua.** A strongly curved cloud without groups was reported as
  `distinct_states` in 2 of 50 samples (0-14%), and refinement split such a
  cloud in 5 of 50 samples (3-22%; section 12). Whether the optional
  effect-size rule would have merged these splits has not been tested. No
  assumption-free test separates such continua from groups reliably; check the
  UMAP map and, for MD, the timeline.
- **Closely spaced groups.** Three groups 4σ apart were reported as
  `partition_not_confirmed` in 19 of 50 samples (25-53%) and otherwise mostly
  not found, and nested sub-groups 5σ apart are recovered in 36 of 50 samples
  (58-84%; section 12): the method prefers missing a weak split to reporting a
  false one.
- **Refined sub-clusters of MD data** may be phases of one simulation rather
  than distinct states (section 9). Check them in the timeline figure.
- **Effect size is reported, not decided.** By default, reproducible groups
  are reported as clusters however close they are; whether they are as
  distinct as the ensemble's states is given by their ratio in
  `separation.csv` (section 6, "Effect size"). The ratio is noisy and its
  reference value 1 is a convention. With the optional rule, a small
  difference that is functional (e.g. Fraser et al. 2009, 2011), or a distinct
  local backbone conformer (Richardson et al. 2008), is reported as
  sub-structure, not as a state, if the ensemble's states are much wider.
- **Groups can reflect how structures were made.** Groups above the state
  scale can still come from modelling tools, laboratories, temperatures or
  processing pipelines rather than from conformational biology; such technical
  ("batch") effects are common in experimental data (Leek et al. 2010). Compare
  the clusters with what is known about the sources of the structures before
  calling them distinct conformations (`struct2clust composition`, section 10).
- **NMR ensembles.** The spread of the models of an NMR entry reflects how
  tightly the data restrain the structure as well as motion (Bonvin & Brünger
  1995). Clusters of NMR models are regions of the structure calculation, not
  populations of states, unless the ensemble was refined to represent motion
  (e.g. against residual dipolar couplings), and even then a few residues can
  decide the clusters (section 10).
- **Populations of predicted and deposited ensembles.** The size of a cluster
  of predicted models reflects how many models each method or protocol
  produced, and that of a cluster of PDB entries reflects what was deposited.
  Neither is a thermodynamic population; for MD, see section 9.
- **Decisions at the threshold.** A partition whose true weakest-cluster
  Jaccard lies just above 0.75 cannot be established with 1000 subsamples and
  is not selected (flag `borderline`). Such a partition may be real but weakly
  supported; increase `selection.max_resamples` to resolve it.
- **Reproducibility is not correctness.** Stability shows that a cluster
  reappears in subsamples, not that it is a meaningful state (Ben-David et al.
  2006); this is why distinct states also require group structure (section 4).
  Hennig's thresholds (0.75, 0.6) are guidelines for interpreting the mean
  Jaccard, not derived error rates.
- **Resampling conventions.** The subsample fraction and the Jaccard threshold
  are conventions (Hennig 2007 gives 0.75 as a guide, not a derived error rate),
  and a result close to the threshold can change with them. Ensembles without
  groups use subsamples of 80% of the structures; grouped ensembles and
  trajectories with automatic blocks use half of the groups or blocks (section 6,
  "Dependent structures"), so "reproducible" is a stricter requirement there.
  On the tar and ubiquitin examples, subsamples of 50% instead of 80% turned
  `partition_not_confirmed` into `no_reproducible_partition`, and a threshold of
  0.8 did so for ubiquitin; the seed did not change either result. In the
  simulations of section 12, a fraction of 0.5 removed the reproducible
  partitions of structureless clouds (9 of 60 at 0.8, 0 of 60 at 0.5) and lowered
  the detection of three groups 5σ apart (18 of 20 at 0.8, 5 of 20 at 0.5): the
  smaller fraction trades over-clustering of continua for under-clustering of
  close groups. The report states the fraction and threshold used.
  `validation/threshold_sensitivity.py` repeats a finished run with thresholds
  0.70, 0.75 and 0.80 and fractions 0.5 and 0.8, so that a reader can see whether
  its conclusion depends on these conventions.
- **Monte Carlo risk only.** The resampling risk bounds the effect of the random
  seed on a fixed data set (section 6). Choosing the finest of many candidates
  on the same data is a selection that fresh subsamples do not undo, and no
  analytical correction for it is applied. The error rates of section 12 are
  measured end to end: every simulated data set went through the whole run,
  including the choice among all candidates, the refinement and every rule, so
  they include the effect of this selection. They hold for the scenarios
  simulated, not for every data set.
- **Noise fraction between equivalent settings.** Among settings that give the
  same number of reproducible clusters, the one with the least noise is chosen
  (section 6). The noise fraction still depends on HDBSCAN's settings, and
  different versions of numpy and scikit-learn can shift it slightly;
  `run_summary.json` records the versions used.
- **Clusterability p-values** are indicative only, because pairwise distances
  are not independent.
- **Block length for trajectories.** The resampling blocks assume that
  correlation between frames decays within one block. By default blocks are
  about one autocorrelation time long, but τ is estimated from one coordinate,
  is a lower bound and is uncertain for short trajectories. The sampling check
  (section 9) catches blocks shorter than τ, clusters whose membership stays
  correlated for longer than a block and clusters that no trajectory enters or
  leaves, and then reports `insufficient_sampling`;
  passing it does not prove that the blocks are independent, and its thresholds
  are quality-control rules, not derived error rates. Short simulations with few
  transitions will often end as `insufficient_sampling`.
  `validation/block_length_politis_white.py` computes, for a finished trajectory
  run, the automatic block length of Politis & White (2004; corrected by Patton
  et al. 2009) for the coordinate that defines the autocorrelation time. That
  rule targets the variance of a mean, not the clustering, so it is a reference
  rather than a replacement. Trajectories whose correlation lasts longer than
  one block hold too few independent frames for 50 blocks, and their stability
  is then less certain than it appears.
- **Memory.** The full distance matrix of N structures takes 8·N² bytes (0.8 GB
  for 10,000 structures), and a run holds several copies: about 8 in the main
  process at its peak and about 2 in every resampling worker (measured at N =
  3000), so about (8 + 2·n_jobs)·8·N² bytes: for 10,000 structures about 6.4 GB
  plus 1.6 GB per worker. The log states the estimate and the memory available,
  and warns when the estimate exceeds it; by default (`n_jobs: null`) no more
  workers are started than the available memory holds. Thin long trajectories
  with `input.stride` or `input.max_frames`. The trustworthiness of the map is
  computed on 3000 randomly chosen structures when there are more.
- **TM-score speed.** `--metric tm` calls US-align once per pair and is much
  slower than RMSD for large inputs.
- **TM-score compares the overall fold.** It uses one atom per residue and
  weights the whole chain, so local and side-chain differences barely change
  it; structures that differ only there are removed as copies (section 3). For
  local and side-chain differences, use RMSD. For short chains it does not
  separate states from noise (section 3).
- **Molecules and atoms.**
  - Only protein and nucleic-acid residues are compared.
  - Ligands, modified residues outside the recognised names, and hydrogens are
    ignored.
  - Different molecules cannot be clustered together.

## References

- Adolfsson A, Ackerman M, Brownstein NC (2019). To cluster, or not to cluster: an analysis of clusterability methods. *Pattern Recognition* 88:13-26.
- Atilgan AR, Durell SR, Jernigan RL, Demirel MC, Keskin O, Bahar I (2001). Anisotropy of fluctuation dynamics of proteins with an elastic network model. *Biophys J* 80:505-515.
- Ben-David S, von Luxburg U, Pál D (2006). A sober look at clustering stability. *COLT*, LNCS 4005:5-19.
- Ben-Hur A, Elisseeff A, Guyon I (2002). A stability based method for discovering structure in clustered data. *Pac Symp Biocomput* 7:6-17.
- Berger RL (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics* 24:295-300.
- Bergsma W (2013). A bias-correction for Cramér's V and Tschuprow's T. *J Korean Stat Soc* 42:323-328.
- Bonvin AMJJ, Brünger AT (1995). Conformational variability of solution nuclear magnetic resonance structures. *J Mol Biol* 250:80-93.
- Butina D (1999). Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto similarity: a fast and automated way to cluster small and large data sets. *J Chem Inf Comput Sci* 39:747-750.
- Campello RJGB, Moulavi D, Sander J (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*, LNCS 7819:160-172.
- Carlstein E (1986). The use of subseries values for estimating the variance of a general statistic from a stationary sequence. *Ann Stat* 14:1171-1179.
- Chari T, Pachter L (2023). The specious art of single-cell genomics. *PLoS Comput Biol* 19:e1011288.
- Clopper CJ, Pearson ES (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26:404-413.
- Cramér H (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- Cruickshank DWJ (1999). Remarks about protein structure precision. *Acta Cryst D* 55:583-601.
- Cumming G (2014). The new statistics: why and how. *Psychol Sci* 25:7-29.
- Davies DL, Bouldin DW (1979). A cluster separation measure. *IEEE Trans Pattern Anal Mach Intell* PAMI-1:224-227.
- Davison AC, Hinkley DV (1997). *Bootstrap Methods and their Application*. Cambridge University Press.
- Ding D, Gandy A, Hahn G (2020). A simple method for implementing Monte Carlo tests. *Comput Stat* 35:1373-1392.
- Field CA, Welsh AH (2007). Bootstrapping clustered data. *J R Stat Soc B* 69:369-390.
- Fisher NI (1993). *Statistical Analysis of Circular Data*. Cambridge University Press.
- Flyvbjerg H, Petersen HG (1989). Error estimates on averages of correlated data. *J Chem Phys* 91:461-466.
- Fraser JS, Clarkson MW, Degnan SC, Erion R, Kern D, Alber T (2009). Hidden alternative structures of proline isomerase essential for catalysis. *Nature* 462:669-673.
- Fraser JS, van den Bedem H, Samelson AJ, Lang PT, Holton JM, Echols N, Alber T (2011). Accessing protein conformational ensembles using room-temperature X-ray crystallography. *Proc Natl Acad Sci USA* 108:16247-16252.
- Gandy A (2009). Sequential implementation of Monte Carlo tests with uniformly bounded resampling risk. *J Am Stat Assoc* 104:1504-1511.
- Gao LL, Bien J, Witten D (2024). Selective inference for hierarchical clustering. *J Am Stat Assoc* 119:332-342.
- Geyer CJ (1992). Practical Markov chain Monte Carlo. *Stat Sci* 7:473-483.
- Gong S, Zhang C, Zhang Y (2019). RNA-align: quick and accurate alignment of RNA 3D structures based on size-independent TM-scoreRNA. *Bioinformatics* 35:4459-4461.
- Gower JC (1966). Some distance properties of latent root and vector methods used in multivariate analysis. *Biometrika* 53:325-338.
- Gowers RJ et al. (2016). MDAnalysis: a Python package for the rapid analysis of molecular dynamics simulations. *Proc 15th Python in Science Conf*, 98-105.
- Grossfield A, Patrone PN, Roe DR, Schultz AJ, Siderius DW, Zuckerman DM (2018). Best practices for quantification of uncertainty and sampling quality in molecular simulations. *Living J Comput Mol Sci* 1:5067.
- Hartigan JA, Hartigan PM (1985). The dip test of unimodality. *Ann Stat* 13:70-84.
- Hennig C (2007). Cluster-wise assessment of cluster stability. *Comput Stat Data Anal* 52:258-271.
- Hennig C (2015). What are the true clusters? *Pattern Recognit Lett* 64:53-62.
- Hubert L, Arabie P (1985). Comparing partitions. *J Classif* 2:193-218.
- IUPAC-IUB Commission on Biochemical Nomenclature (1970). Abbreviations and symbols for the description of the conformation of polypeptide chains. *Biochemistry* 9:3471-3479.
- IUPAC-IUB Joint Commission on Biochemical Nomenclature (1983). Abbreviations and symbols for the description of conformations of polynucleotide chains. *Eur J Biochem* 131:9-15.
- Kabsch W (1976). A solution for the best rotation to relate two sets of vectors. *Acta Cryst A* 32:922-923.
- Kabsch W (1978). A discussion of the solution for the best rotation to relate two sets of vectors. *Acta Cryst A* 34:827-828.
- Kimes PK, Liu Y, Hayes DN, Marron JS (2017). Statistical significance for hierarchical clustering. *Biometrics* 73:811-821.
- Künsch HR (1989). The jackknife and the bootstrap for general stationary observations. *Ann Stat* 17:1217-1241.
- Lange T, Roth V, Braun ML, Buhmann JM (2004). Stability-based validation of clustering solutions. *Neural Comput* 16:1299-1323.
- Leek JT, Scharpf RB, Bravo HC, Simcha D, Langmead B, Johnson WE, Geman D, Baggerly K, Irizarry RA (2010). Tackling the widespread and critical impact of batch effects in high-throughput data. *Nat Rev Genet* 11:733-739.
- Lyman E, Zuckerman DM (2006). Ensemble-based convergence analysis of biomolecular trajectories. *Biophys J* 91:164-172.
- Lyman E, Zuckerman DM (2007). On the structural convergence of biomolecular simulations by determination of the effective sample size. *J Phys Chem B* 111:12876-12882.
- Madras N, Sokal AD (1988). The pivot algorithm: a highly efficient Monte Carlo method for the self-avoiding walk. *J Stat Phys* 50:109-186.
- Mariani V, Biasini M, Barbato A, Schwede T (2013). lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. *Bioinformatics* 29:2722-2728.
- Markley JL, Bax A, Arata Y, Hilbers CW, Kaptein R, Sykes BD, Wright PE, Wüthrich K (1998). Recommendations for the presentation of NMR structures of proteins and nucleic acids. *J Mol Biol* 280:933-952.
- Mardia KV, Jupp PE (2000). *Directional Statistics*. Wiley.
- McInnes L, Healy J, Melville J (2018). UMAP: uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.
- Meinshausen N (2008). Hierarchical testing of variable importance. *Biometrika* 95:265-278.
- Meli R, Biggin PC (2020). spyrmsd: symmetry-corrected RMSD calculations in Python. *J Cheminform* 12:49.
- Michaud-Agrawal N, Denning EJ, Woolf TB, Beckstein O (2011). MDAnalysis: a toolkit for the analysis of molecular dynamics simulations. *J Comput Chem* 32:2319-2327.
- Monti S, Tamayo P, Mesirov J, Golub T (2003). Consensus clustering: a resampling-based method for class discovery and visualization of gene expression microarray data. *Mach Learn* 52:91-118.
- Moulavi D, Jaskowiak PA, Campello RJGB, Zimek A, Sander J (2014). Density-based clustering validation. *SIAM Int Conf Data Mining*, 839-847.
- Patton A, Politis DN, White H (2009). Correction to "Automatic block-length selection for the dependent bootstrap" by D. Politis and H. White. *Econometric Reviews* 28:372-375.
- Pedregosa F et al. (2011). Scikit-learn: machine learning in Python. *J Mach Learn Res* 12:2825-2830.
- Politis DN, White H (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews* 23:53-70.
- Prinz J-H, Wu H, Sarich M, Keller B, Senne M, Held M, Chodera JD, Schütte C, Noé F (2011). Markov models of molecular kinetics: generation and validation. *J Chem Phys* 134:174105.
- Richardson JS, Schneider B, Murray LW et al. (2008). RNA backbone: consensus all-angle conformers and modular string nomenclature (an RNA Ontology Consortium contribution). *RNA* 14:465-481.
- Schuirmann DJ (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *J Pharmacokinet Biopharm* 15:657-680.
- Shah RD, Samworth RJ (2013). Variable selection with error control: another look at stability selection. *J R Stat Soc B* 75:55-80.
- Sokal AD (1997). Monte Carlo methods in statistical mechanics: foundations and new algorithms. In DeWitt-Morette C, Cartier P, Folacci A (eds), *Functional Integration: Basics and Applications*, NATO ASI Series B 361:131-192. Plenum.
- Tibshirani R, Walther G (2005). Cluster validation by prediction strength. *J Comput Graph Stat* 14:511-528.
- Venna J, Kaski S (2001). Neighborhood preservation in nonlinear projection methods: an experimental study. *ICANN*, LNCS 2130:485-491.
- Ville J (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
- Vinh NX, Epps J, Bailey J (2010). Information theoretic measures for clusterings comparison: variants, properties, normalization and correction for chance. *J Mach Learn Res* 11:2837-2854.
- von Luxburg U (2007). A tutorial on spectral clustering. *Stat Comput* 17:395-416.
- von Luxburg U (2010). Clustering stability: an overview. *Found Trends Mach Learn* 2:235-274.
- Wasserstein RL, Lazar NA (2016). The ASA statement on p-values: context, process, and purpose. *Am Stat* 70:129-133.
- Waudby-Smith I, Ramdas A (2024). Estimating means of bounded random variables by betting. *J R Stat Soc B* 86:1-27.
- Wehmeyer C, Scherer MK, Hempel T, Husic BE, Olsson S, Noé F (2019). Introduction to Markov state modeling with the PyEMMA software. *Living J Comput Mol Sci* 1:5965.
- Wolff U (2004). Monte Carlo errors with less errors. *Comput Phys Commun* 156:143-153. Erratum: *Comput Phys Commun* 176:383 (2007).
- Xu J, Zhang Y (2010). How significant is a protein structure similarity with TM-score = 0.5? *Bioinformatics* 26:889-895.
- Zelnik-Manor L, Perona P (2004). Self-tuning spectral clustering. *Adv Neural Inf Process Syst* 17:1601-1608.
- Zhang C, Shine M, Pyle AM, Zhang Y (2022). US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes. *Nat Methods* 19:1109-1115.
- Zhang Y, Skolnick J (2004). Scoring function for automated assessment of protein structure template quality. *Proteins* 57:702-710.
