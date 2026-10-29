# Examples

Six data sets: a synthetic ensemble with a known answer, and five public data
sets of different kinds (NMR ensembles, MD trajectories and crystal structures).
The data and the results are not part of the repository: the scripts below
write the data to `data/`, and the commands below write the results to
`results/`.

| File | Purpose |
|---|---|
| `make_example.py` | Writes the synthetic ensemble (`data/synthetic/`) |
| `download_examples.py` | Downloads TAR, ubiquitin and 1RKJ into `data/` |
| `prepare_crystal_ensembles.py` | Builds the endothiapepsin and ERα crystal-structure ensembles from the PDB (needs Biopython) |
| `1rkj_rna_bound_vs_free.yaml` | Configuration for the 1RKJ trajectories |
| `large_ensembles.yaml` | Configuration for the two crystal-structure ensembles: `cluster_views/` shows the 50 members closest to each medoid (`figures.cluster_view_max_members`), which keeps the viewer files small |
| `source_tables.py` | Writes the source of every structure (PDB entry, bound partner, space group, simulation) to `data/<run>_sources.csv`, for `struct2clust composition` |

## Data sets

| Name | Content | Source and licence |
|---|---|---|
| `synthetic` | 60 models of an 8-nucleotide RNA in three known states, 20 each | `make_example.py` |
| `tar` | HIV-1 TAR RNA, 29 nt: NMR entries 7JU1, 1ANR, 1ARJ, 2L8H, 1UTS, 1UUD, 1UUI (88 models, free and ligand-bound) | RCSB Protein Data Bank, CC0 |
| `ubiquitin` | Ubiquitin, 76 residues: PDB 2K39, 116 models refined against residual dipolar couplings (Lange et al. 2008, *Science* 320:1471-1475) | RCSB Protein Data Bank, CC0 |
| `1RKJ` | RNA of the nucleolin RBD1-2 / RNA complex (PDB 1RKJ), 21 nt: 500 ns MD bound in the complex and free in solution. RNA atoms only, 500 frames per run spread evenly over each run | Sabei A, Hognon C, Martin J, Frezza E (2023), Zenodo, doi:[10.5281/zenodo.10040600](https://doi.org/10.5281/zenodo.10040600), CC-BY-4.0; reduced by `download_examples.py` |
| `endothiapepsin` | Endothiapepsin (UniProt P11838): 948 chains from all X-ray entries, mature protein (residues 90-419) | RCSB Protein Data Bank, CC0; built by `prepare_crystal_ensembles.py` |
| `eralpha` | Estrogen receptor α ligand-binding domain (UniProt P03372): 409 chains from all X-ray entries, 199 core residues including helix 12 | RCSB Protein Data Bank, CC0; built by `prepare_crystal_ensembles.py` |

`prepare_crystal_ensembles.py` also writes `data/<name>_meta.csv`, which lists,
for every chain of the two crystal-structure ensembles, whether it was used and
why not, its resolution, space group, ligands and other polymers. The PDB grows,
so running the script at a later date can give more chains; the meta file
records the entries that were used.

## Running them

From the repository root:

```bash
python example/make_example.py
python example/download_examples.py                 # tar, ubiquitin, 1rkj
python example/prepare_crystal_ensembles.py endothiapepsin
python example/prepare_crystal_ensembles.py eralpha

struct2clust run example/data/synthetic -o example/results
struct2clust run example/data/tar -o example/results
struct2clust run example/data/ubiquitin -o example/results
struct2clust run -c example/1rkj_rna_bound_vs_free.yaml -o example/results -n 1RKJ_rna
struct2clust run example/data/endothiapepsin -c example/large_ensembles.yaml -o example/results
struct2clust run example/data/eralpha -c example/large_ensembles.yaml -o example/results

python example/source_tables.py
struct2clust composition example/results/tar example/data/tar_sources.csv --label "PDB entry"
struct2clust composition example/results/eralpha example/data/eralpha_sources.csv --label "Bound partner"
struct2clust composition example/results/endothiapepsin example/data/endothiapepsin_sources.csv --label "Space group"
struct2clust composition example/results/1RKJ_rna example/data/1RKJ_rna_sources.csv --label "Simulation"
```

The first four commands build or download the data. They need internet access
(RCSB PDB and Zenodo); `prepare_crystal_ensembles.py` also needs Biopython, and
`download_examples.py` downloads the 820 MB 1RKJ archive from Zenodo and keeps
a reduced copy. The data keep the licences given in the table above: the PDB
data are CC0; the 1RKJ trajectories are CC-BY-4.0 and are to be cited as given
there.

## Reading the results

The examples show how the method behaves on real data; they are not studies of
these molecules. Points to keep in mind:

- **Dependent structures.** The examples run without `input.groups`, so the
  models of one NMR entry and the chain copies of one crystal entry count as
  independent structures. A cluster can then reflect the entries rather than
  states of the molecule ([METHODS.md](../docs/METHODS.md), section 6, "Dependent
  structures"). Check the composition of the clusters by entry
  (`struct2clust composition`). The 88 tar models come from 7 entries, fewer
  than the 8 groups that resampling by entry needs, so tar cannot be run with
  groups by entry. For the crystal ensembles, the column `pdb` of
  `data/<name>_meta.csv` gives the entry of every chain; a file with the columns
  `structure` and `group` built from it can be passed as `input.groups`. The
  `*_sources.csv` files hold other labels (bound partner, space group) for
  `composition`, not groups.
- **NMR ensembles.** The spread of NMR models reflects the precision of the
  structure determination as well as motion; their clusters are not populations
  of states ([METHODS.md](../docs/METHODS.md), section 13).
- **1RKJ.** The bound and the free RNA are one simulation each, and neither
  leaves its cluster, so the run ends as `insufficient_sampling`: the clusters
  show that the two simulations differ (`populations.csv`), but one simulation per
  condition cannot show that the difference is reproducible ([METHODS.md](../docs/METHODS.md),
  section 9, "Comparing conditions").
