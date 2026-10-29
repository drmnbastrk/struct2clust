# Which atoms are compared

`--atoms` (config `metric.atoms`) chooses the atoms used by `--metric rmsd`.
`--metric tm` ignores it: US-align always uses one atom per residue
(C3′ for nucleotides, CA for amino acids).

Every run logs the atoms it actually compared, per residue type, and writes the
same list to `run_summary.json`.

## General rules (all selections)

- Only protein and nucleic-acid residues are used. Water, ions, ligands and
  other hetero groups are ignored. Force-field residue names are recognised,
  for example `HIE`/`HID`/`HIP` and `CYX`. Terminal nucleotides are recognised
  for every base: a nucleotide name followed by `5` (5′ end), `3` (3′ end) or
  `N` (both ends), e.g. `G5`, `U3`, `RA5`, `DT3`, `AN`.
- Hydrogens are never used. An atom is treated as a hydrogen when its element
  column is `H` or `D`. If the element column is blank, the first letter of the
  atom name decides.
- Force-field naming variants are unified before matching: residue names
  `G5`/`G3`/`GN` = `G` (likewise for every base), `HIE`/`HID`/`HSD` = `HIS`,
  `RA` = `A`, `MSE` (selenomethionine) = `MET` (its `SE` and the `SD` of
  methionine have no counterpart in the other and are left out); atom names `O1P` =
  `OP1`, `O2P` = `OP2`, `C4*` = `C4'`, `C5M` = `C7`. Outputs use the current PDB
  atom names.
- Atoms are matched between structures by residue position and atom name. An
  atom that is missing from **any** structure is left out for all of them. The
  number of such atoms is logged.
- For alternate locations, the first one listed in the file is used.

## `heavy-atoms` (default)

This selection uses every non-hydrogen atom of every residue.

### Nucleic acids

| Part | Atoms |
|---|---|
| Phosphate | `P`, `OP1`, `OP2` (and `OP3` on a 5′ phosphate, if present in all structures) |
| Sugar-phosphate backbone | `O5'`, `C5'`, `C4'`, `C3'`, `O3'` |
| Sugar ring | `O4'`, `C1'`, `C2'` |
| 2′-hydroxyl (RNA only) | `O2'` |
| Adenine (A, DA) | `N9`, `C8`, `N7`, `C5`, `C6`, `N6`, `N1`, `C2`, `N3`, `C4` |
| Guanine (G, DG) | `N9`, `C8`, `N7`, `C5`, `C6`, `O6`, `N1`, `C2`, `N2`, `N3`, `C4` |
| Cytosine (C, DC) | `N1`, `C2`, `O2`, `N3`, `C4`, `N4`, `C5`, `C6` |
| Uracil (U) | `N1`, `C2`, `O2`, `N3`, `C4`, `O4`, `C5`, `C6` |
| Thymine (DT) | `N1`, `C2`, `O2`, `N3`, `C4`, `O4`, `C5`, `C7` (older name `C5M`), `C6` |

The first residue of a chain has no `P`, `OP1` or `OP2` unless the file contains
a 5′ phosphate.

### Proteins

Every amino acid contributes the backbone atoms `N`, `CA`, `C` and `O`, plus `OXT`
at the C-terminus if all structures have it. The side-chain atoms are:

| Residue | Side-chain atoms |
|---|---|
| ALA | `CB` |
| ARG | `CB`, `CG`, `CD`, `NE`, `CZ`, `NH1`, `NH2` |
| ASN | `CB`, `CG`, `OD1`, `ND2` |
| ASP | `CB`, `CG`, `OD1`, `OD2` |
| CYS | `CB`, `SG` |
| GLN | `CB`, `CG`, `CD`, `OE1`, `NE2` |
| GLU | `CB`, `CG`, `CD`, `OE1`, `OE2` |
| GLY | none |
| HIS | `CB`, `CG`, `ND1`, `CD2`, `CE1`, `NE2` |
| ILE | `CB`, `CG1`, `CG2`, `CD1` |
| LEU | `CB`, `CG`, `CD1`, `CD2` |
| LYS | `CB`, `CG`, `CD`, `CE`, `NZ` |
| MET | `CB`, `CG`, `SD`, `CE` |
| PHE | `CB`, `CG`, `CD1`, `CD2`, `CE1`, `CE2`, `CZ` |
| PRO | `CB`, `CG`, `CD` |
| SER | `CB`, `OG` |
| THR | `CB`, `OG1`, `CG2` |
| TRP | `CB`, `CG`, `CD1`, `CD2`, `NE1`, `CE2`, `CE3`, `CZ2`, `CZ3`, `CH2` |
| TYR | `CB`, `CG`, `CD1`, `CD2`, `CE1`, `CE2`, `CZ`, `OH` |
| VAL | `CB`, `CG1`, `CG2` |

**Chemically equivalent atoms.** In ARG (`NH1`/`NH2`), ASP (`OD1`/`OD2`), GLU
(`OE1`/`OE2`), LEU (`CD1`/`CD2`), VAL (`CG1`/`CG2`), PHE and TYR (`CD1`/`CD2` with
`CE1`/`CE2`), at a free C terminus (`O`/`OXT`) and in phosphates (`OP1`/`OP2`,
and `OP3` at a 5' end), the heavy-atom structure cannot tell the atoms apart, so
their naming is a convention. By default RMSD pairs them in the order that gives
the lower RMSD (`metric.equivalent_atoms`; [METHODS.md](METHODS.md), section 2);
with `equivalent_atoms: false`, inconsistent naming inflates heavy-atom RMSD.

## `backbone`

| Molecule | Atoms |
|---|---|
| Nucleic acids | `P`, `O5'`, `C5'`, `C4'`, `O4'`, `C3'`, `O3'`, `C2'`, `O2'` (RNA only), `C1'` |
| Proteins | `N`, `CA`, `C`, `O` |

The nucleic-acid backbone excludes the non-bridging phosphate oxygens `OP1` and
`OP2`. They are prochiral, and some structure-generation tools swap their names,
which would inflate RMSD without any real conformational difference.

## `anchor`

This selection uses one atom per residue: `C3'` for nucleotides and `CA` for
amino acids. These are the same atoms US-align uses for TM-score.
