"""Reading structural ensembles into one coordinate array.

Structure ensembles (PDB files) and MD trajectories are handled the same way.
All members must describe the same molecule: the same residue sequence after
unifying force-field naming variants (G5 = G, HIE = HIS, O1P = OP1). Atoms are
matched by (residue position, atom name); only atoms present in every member are
kept, so every metric sees exactly the same atoms. Hydrogens, solvent, ions and
other non-polymer residues are always dropped.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STRUCTURE_SUFFIXES = {".pdb", ".ent"}

AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL",
    # common force-field protonation/disulfide variants
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
})
_NUCLEOTIDE_BASES = frozenset({
    "A", "C", "G", "U", "T", "I", "DA", "DC", "DG", "DT", "DU", "DI",
    "RA", "RC", "RG", "RU", "ADE", "CYT", "GUA", "URA", "THY",
})

# Atom selections (see docs/ATOMS.md for the per-residue lists):
#
#   heavy-atoms  every non-hydrogen atom of every protein / nucleic-acid residue
#                nucleic: phosphate (P, OP1, OP2), sugar-phosphate backbone,
#                         ribose/deoxyribose ring, 2'-OH (O2', RNA) and all base atoms
#                protein: backbone (N, CA, C, O, OXT) and all side-chain atoms
#   backbone     nucleic: NUCLEIC_BACKBONE; protein: PROTEIN_BACKBONE
#   anchor       one atom per residue: C3' (nucleic) / CA (protein), as in US-align
PROTEIN_BACKBONE = frozenset({"N", "CA", "C", "O"})
# The non-bridging phosphate oxygens (OP1/OP2) are left out of "backbone": they
# are prochiral and occasionally swapped by structure-generation tools, which
# would inflate RMSD without any conformational difference.
NUCLEIC_BACKBONE = frozenset({
    "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
})
ANCHOR_ATOM = {"protein": "CA", "nucleic": "C3'"}


def residue_kind(resname: str) -> Optional[str]:
    """'protein', 'nucleic', or None (solvent, ion, ligand, cap...).

    Force-field terminal variants are recognised as nucleotides: any known
    nucleotide name followed by 5 (5' terminus), 3 (3' terminus) or N (a single
    nucleotide that is both), e.g. G5, U3, RA5, DT3, AN.
    """
    name = resname.strip().upper()
    if name in AMINO_ACIDS:
        return "protein"
    if name in _NUCLEOTIDE_BASES:
        return "nucleic"
    if _terminal_base(name) is not None:
        return "nucleic"
    return None


def _terminal_base(name: str) -> Optional[str]:
    """Nucleotide name without its terminal suffix (5, 3 or N), or None."""
    if len(name) >= 2 and name[-1] in "35N" and name[:-1] in _NUCLEOTIDE_BASES:
        return name[:-1]
    return None


_RESNAME_ALIASES = {
    "RA": "A", "RC": "C", "RG": "G", "RU": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U", "THY": "DT",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    "MSE": "MET",   # selenomethionine in crystal structures; SE and SD are not matched
}
_ATOM_ALIASES = {"O1P": "OP1", "O2P": "OP2", "O3P": "OP3", "C5M": "C7"}


def canonical_resname(resname: str) -> str:
    """Residue name with force-field variants unified (G5 -> G, HIE -> HIS, RA -> A).

    Used to decide whether two inputs describe the same molecule; the original
    names are kept for output.
    """
    name = resname.strip().upper()
    if name not in _NUCLEOTIDE_BASES and _terminal_base(name) is not None:
        name = _terminal_base(name)
    return _RESNAME_ALIASES.get(name, name)


def canonical_atom_name(name: str) -> str:
    """Atom name in current PDB convention (O1P -> OP1, C4* -> C4', C5M -> C7)."""
    name = name.strip().replace("*", "'")
    return _ATOM_ALIASES.get(name, name)


def _is_hydrogen(atom_name: str, element: str) -> bool:
    if element:
        return element.upper() in ("H", "D")
    return atom_name.lstrip("0123456789")[:1].upper() in ("H", "D")


@dataclass
class Atoms:
    """Per-atom annotation shared by every member of an ensemble."""

    chain: np.ndarray    # chainID (or segid when chainID is blank)
    resid: np.ndarray    # original residue number (from the first member)
    resname: np.ndarray
    name: np.ndarray
    element: np.ndarray
    kind: np.ndarray     # 'protein' | 'nucleic'
    residue: np.ndarray  # 0-based residue position within the molecule

    def __len__(self) -> int:
        return len(self.name)

    def subset(self, mask: np.ndarray) -> "Atoms":
        return Atoms(**{k: v[mask] for k, v in self.__dict__.items()})


@dataclass
class Ensemble:
    names: list[str]              # one label per member
    sources: list[Optional[Path]]  # original file per member (None for trajectory frames)
    atoms: Atoms
    coords: np.ndarray            # (n_members, n_atoms, 3) float64
    replica: Optional[list[str]] = None  # trajectory mode: replica label per frame
    frame: Optional[list[int]] = None    # trajectory mode: frame index in its trajectory

    def __len__(self) -> int:
        return len(self.names)

    @property
    def replicas(self) -> list[str]:
        """Replica labels in input order (empty for ensembles)."""
        return list(dict.fromkeys(self.replica)) if self.replica else []

    @property
    def molecule_type(self) -> str:
        kinds = set(self.atoms.kind.tolist())
        if kinds == {"protein"}:
            return "protein"
        if kinds == {"nucleic"}:
            return "nucleic"
        return "complex"

    @property
    def n_residues(self) -> int:
        return int(self.atoms.residue.max()) + 1 if len(self.atoms) else 0

    def subset(self, index) -> "Ensemble":
        index = np.asarray(index)

        def pick(xs):
            return [xs[i] for i in index] if xs is not None else None

        return Ensemble(
            names=pick(self.names),
            sources=pick(self.sources),
            atoms=self.atoms,
            coords=self.coords[index],
            replica=pick(self.replica),
            frame=pick(self.frame),
        )


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

@dataclass
class _Record:
    """One parsed model: per-atom annotation lists and coordinates."""

    keys: list          # (residue position, atom name)
    resnames: list      # residue name per residue position
    chain: list
    resid: list
    name: list
    element: list
    kind: list
    xyz: np.ndarray


def choose_conformers(atoms: list[tuple]) -> tuple[list[tuple], int]:
    """One alternate conformer per residue.

    ``atoms`` holds (chain, resseq, icode, ..., altloc, occupancy) tuples. Atoms
    without an alternate-location label belong to every conformer and are always
    kept. Of the labelled atoms of a residue, only those of one conformer are kept:
    the label whose atoms have the highest mean occupancy (the mean over the atoms
    that carry the label), ties going to the earlier label (A before B). Atoms are
    never taken from two conformers, so an atom present only in another conformer
    is missing from this residue, and is then left out of the comparison like any
    other missing atom. Returns (kept atoms, number of residues with conformers).
    """
    occupancy: dict[tuple, dict[str, list[float]]] = {}
    for a in atoms:
        if a[-2]:
            occupancy.setdefault(a[:3], {}).setdefault(a[-2], []).append(a[-1])
    chosen = {res: min(alts, key=lambda alt: (-np.mean(alts[alt]), alt))
              for res, alts in occupancy.items()}
    kept = [a for a in atoms if not a[-2] or chosen[a[:3]] == a[-2]]
    return kept, len(chosen)


def _parse_models(path: Path) -> list[_Record]:
    """Parse every MODEL of a PDB file (a file without MODEL records is one model).

    Only polymer heavy atoms are kept; of residues with alternate locations, one
    conformer is kept (``choose_conformers``).
    """
    models: list[_Record] = []
    atoms: list[tuple] = []
    dropped: list[str] = []
    skipped: list[tuple] = []      # first model: residues that are not protein / nucleic
    het_keys: set[tuple] = set()   # residues written as HETATM in the current model
    with_conformers = []

    def close_model():
        if not atoms:
            return
        kept, n_conf = choose_conformers(atoms)
        if not models:
            with_conformers.append(n_conf)
        atoms[:] = [a[:8] for a in kept]
        # residues in file order, to find free ligands with standard residue names
        order: dict[tuple, tuple] = {}
        for chain, resseq, icode, resname, name, _, kind, pos in atoms:
            entry = order.setdefault((chain, resseq, icode), (chain, kind, {}, resname))
            entry[2].setdefault(name, pos)
        keys = list(order)
        free = {keys[i] for i in unlinked_residues([order[k][:3] for k in keys],
                                                    [k in het_keys for k in keys])}
        if free and not models:
            dropped.extend(f"{order[k][3]}{k[1]}{k[2]}:{k[0] or '-'}" for k in keys
                           if k in free)
        residue_index: dict[tuple, int] = {}
        resnames: list[str] = []
        seen: set[tuple] = set()
        rec = _Record([], resnames, [], [], [], [], [], np.empty((0, 3)))
        xyz = []
        for chain, resseq, icode, resname, name, element, kind, pos in atoms:
            rkey = (chain, resseq, icode)
            if rkey in free:
                continue
            if rkey not in residue_index:
                residue_index[rkey] = len(resnames)
                resnames.append(resname)
            akey = (residue_index[rkey], name)
            if akey in seen:  # alternate location or duplicate atom
                continue
            seen.add(akey)
            rec.keys.append(akey)
            rec.chain.append(chain)
            rec.resid.append(resseq)
            rec.name.append(name)
            rec.element.append(element)
            rec.kind.append(kind)
            xyz.append(pos)
        rec.xyz = np.asarray(xyz, dtype=np.float64)
        models.append(rec)
        atoms.clear()
        het_keys.clear()

    with open(path, encoding="utf-8", errors="replace") as fh:
        for number, line in enumerate(fh, start=1):
            record = line[:6]
            if record.startswith("MODEL"):
                close_model()
            elif record.startswith("ENDMDL"):
                close_model()
            elif record in ("ATOM  ", "HETATM"):
                resname = line[17:21].strip()
                kind = residue_kind(resname)
                if kind is None:
                    if not models and line[22:26].strip().lstrip("-").isdigit():
                        skipped.append(((line[21].strip() or line[72:76].strip()),
                                        int(line[22:26]), resname))
                    continue
                raw_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) > 76 else ""
                if _is_hydrogen(raw_name, element):
                    continue
                name = canonical_atom_name(raw_name)
                segid = line[72:76].strip() if len(line) > 72 else ""
                chain = line[21].strip() or segid
                try:
                    resseq = int(line[22:26])
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    raise ValueError(
                        f"{path.name}, line {number}: residue number or coordinates cannot "
                        "be read (fixed PDB columns 23-26 and 31-54 expected)") from None
                if record == "HETATM":
                    het_keys.add((chain, resseq, line[26].strip()))
                try:
                    occupancy = float(line[54:60])
                except ValueError:      # no occupancy column: all conformers equal
                    occupancy = 1.0
                atoms.append((
                    chain, resseq, line[26].strip(), resname, name, element, kind,
                    xyz, line[16].strip(), occupancy,
                ))
    close_model()
    if with_conformers and with_conformers[0]:
        logger.info("%s: %d residue(s) with alternate locations; the conformer with the "
                    "highest mean occupancy is used", path.name, with_conformers[0])
    if dropped:
        logger.info("%s: ignored %d residue(s) not linked to a polymer chain (free ligands "
                    "with standard names): %s", path.name, len(dropped), ", ".join(dropped))
    inside = residues_inside_chains(skipped, models[0] if models else None)
    if inside:
        logger.warning(
            "%s: %d residue(s) inside a polymer chain are not standard protein or "
            "nucleic-acid residues (e.g. modified residues) and are left out, which leaves "
            "a gap in the chain: %s", path.name, len(inside),
            ", ".join(f"{r}{n}:{c or '-'}" for c, n, r in inside[:10])
            + (" ..." if len(inside) > 10 else ""))
    return models


def residues_inside_chains(skipped: list[tuple], record) -> list[tuple]:
    """Skipped residues (chain, number, name) that lie between two kept residues of
    the same chain, i.e. within a polymer rather than ligands, water or ions."""
    if record is None or not skipped:
        return []
    span: dict[str, tuple[int, int]] = {}
    for chain, resid in zip(record.chain, record.resid):
        lo, hi = span.get(chain, (resid, resid))
        span[chain] = (min(lo, resid), max(hi, resid))
    out = []
    for chain, number, name in dict.fromkeys(skipped):
        if chain in span and span[chain][0] < number < span[chain][1]:
            out.append((chain, number, name))
    return out


LINK_ATOMS = {"protein": ("C", "N"), "nucleic": ("O3'", "P")}
LINK_MAX_DISTANCE = 2.0   # Angstrom; peptide C-N ~1.33, phosphodiester O3'-P ~1.61


def unlinked_residues(
    residues: list[tuple[str, str, dict]], hetatm: Optional[list[bool]] = None
) -> list[int]:
    """Indices of free molecules that only carry a standard residue name.

    ``residues`` is a list of (chain, kind, {atom name: xyz}) in file order. Two
    consecutive residues of the same chain and kind are linked when C(i)-N(i+1)
    (protein) or O3'(i)-P(i+1) (nucleic acid) is shorter than 2 A. A residue is
    reported only if it is linked to neither neighbour AND it looks like a
    ligand: it is written as HETATM (``hetatm``), or it is the only residue of
    its chain. A polymer residue whose backbone geometry is broken (e.g. a
    terminal nucleotide slightly too far from its neighbour in a model) is
    therefore kept. If no residue is linked at all (single residues, CA-only
    models), nothing is reported.
    """
    n = len(residues)
    linked = [False] * n
    for i in range(n - 1):
        (ca, ka, a), (cb, kb, b) = residues[i], residues[i + 1]
        if ca != cb or ka != kb:
            continue
        first, second = LINK_ATOMS[ka]
        if first in a and second in b:
            if np.linalg.norm(np.asarray(a[first]) - np.asarray(b[second])) < LINK_MAX_DISTANCE:
                linked[i] = linked[i + 1] = True
    if not any(linked):
        return []
    chain_size: dict[str, int] = {}
    for chain, _, _ in residues:
        chain_size[chain] = chain_size.get(chain, 0) + 1
    het = hetatm or [False] * n
    return [i for i in range(n)
            if not linked[i] and (het[i] or chain_size[residues[i][0]] == 1)]


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def find_structure_files(path: Path) -> list[Path]:
    """A single file, or all PDB files in a directory (non-recursive, natural order)."""
    path = Path(path)
    if path.is_file():
        return [path]
    files = [p for p in path.iterdir() if p.suffix.lower() in STRUCTURE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No PDB files (*.pdb, *.ent) found in {path}")
    return sorted(files, key=_natural_key)


def load_ensemble(path: Path) -> Ensemble:
    """Load a directory of PDB files, or a single multi-model PDB, as one ensemble."""
    files = find_structure_files(path)
    records: list[_Record] = []
    names: list[str] = []
    sources: list[Optional[Path]] = []
    for f in files:
        models = _parse_models(f)
        if not models:
            logger.warning("Skipping %s: no protein or nucleic-acid atoms", f.name)
            continue
        if len(models) == 1:
            names.append(f.stem)
            sources.append(f)
            records.append(models[0])
        else:
            for k, model in enumerate(models, start=1):
                names.append(f"{f.stem}_m{k}")
                sources.append(None)
                records.append(model)
    if len(records) < 2:
        raise ValueError(f"Need at least 2 structures, found {len(records)} in {path}")
    return _assemble(records, names, sources)


def _match_atoms(records: list, names: list[str], what: str = "structures"):
    """Atoms common to all records, in the order of the first one.

    Records must describe the same molecule: identical residue sequences after
    unifying force-field residue-name variants. Returns (Atoms, [index array per
    record mapping its atoms onto the common order]).
    """
    ref = records[0]
    ref_seq = [canonical_resname(r) for r in ref.resnames]
    for rec, name in zip(records[1:], names[1:]):
        seq = [canonical_resname(r) for r in rec.resnames]
        if seq != ref_seq:
            first = next((i for i, (a, b) in enumerate(zip(seq, ref_seq)) if a != b),
                         min(len(seq), len(ref_seq)))
            raise ValueError(
                f"{name}: residue sequence differs from {names[0]} "
                f"({len(seq)} vs {len(ref_seq)} residues; first difference at position "
                f"{first + 1}). All {what} must be the same molecule; different molecules "
                "cannot be compared atom by atom."
            )

    common = set(ref.keys)
    for rec in records[1:]:
        common.intersection_update(rec.keys)
    keep = [i for i, k in enumerate(ref.keys) if k in common]
    if not keep:
        raise ValueError(f"No atoms are shared by all {what}.")
    order = [ref.keys[i] for i in keep]
    indices = []
    for rec in records:
        lookup = {k: i for i, k in enumerate(rec.keys)}
        indices.append(np.array([lookup[k] for k in order]))

    missing: dict[tuple, int] = {}
    for rec in records:
        for key in rec.keys:
            if key not in common:
                missing[key] = missing.get(key, 0) + 1
    if missing:
        shown = sorted(missing)[:10]
        logger.warning(
            "%d atom(s) absent from some %s were excluded from all; %d atoms are compared. "
            "Excluded (residue position:name, present in n %s): %s%s",
            len(missing), what, len(order), what,
            ", ".join(f"{ref.resnames[r]}{r + 1}:{a} ({missing[(r, a)]})" for r, a in shown),
            " ..." if len(missing) > 10 else "",
        )

    atoms = Atoms(
        chain=np.array([ref.chain[i] for i in keep]),
        resid=np.array([ref.resid[i] for i in keep]),
        resname=np.array([ref.resnames[ref.keys[i][0]] for i in keep]),
        name=np.array([ref.name[i] for i in keep]),
        element=np.array([ref.element[i] for i in keep]),
        kind=np.array([ref.kind[i] for i in keep]),
        residue=np.array([ref.keys[i][0] for i in keep]),
    )
    return atoms, indices


def _assemble(records: list[_Record], names: list[str], sources: list) -> Ensemble:
    atoms, indices = _match_atoms(records, names, "structures")
    coords = np.stack([rec.xyz[idx] for rec, idx in zip(records, indices)]).astype(np.float64)
    ens = Ensemble(names=names, sources=sources, atoms=atoms, coords=coords)
    logger.info(
        "Loaded %d structures: %s, %d residues, %d heavy atoms",
        len(ens), ens.molecule_type, ens.n_residues, len(atoms),
    )
    return ens


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def replica_labels(paths: list[Path]) -> list[str]:
    """Short, unique labels for trajectory files: the path components that differ.

    .../SET1/trajectory/nucleic.dcd and .../SET2/trajectory/nucleic.dcd -> SET1, SET2;
    run_a.dcd and run_b.dcd in one folder -> run_a, run_b.
    """
    parts = [list(Path(p).with_suffix("").parts) for p in paths]
    if len(parts) == 1:
        return [parts[0][-1]]
    while all(len(p) > 1 for p in parts) and len({p[0] for p in parts}) == 1:
        parts = [p[1:] for p in parts]
    while all(len(p) > 1 for p in parts) and len({p[-1] for p in parts}) == 1:
        parts = [p[:-1] for p in parts]
    labels = ["/".join(p) for p in parts]
    if len(set(labels)) < len(labels):
        labels = [f"rep{i + 1}" for i in range(len(paths))]
    return labels


def select_frames(n_frames: int, stride: int = 1, max_frames: Optional[int] = None) -> list[int]:
    """Frame indices: every ``stride``-th frame, then at most ``max_frames`` of them
    spread evenly over the whole trajectory (not just its beginning)."""
    frames = list(range(0, n_frames, stride))
    if max_frames is not None and len(frames) > max_frames:
        pick = np.unique(np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(int))
        frames = [frames[i] for i in pick]
    return frames


def _trajectory_record(topology: Path, trajectory: Path, selection: str, stride: int,
                       max_frames: Optional[int]):
    """(_Record describing the selected atoms, (n_frames, n_atoms, 3) coordinates, frame ids)."""
    import MDAnalysis as mda

    with warnings.catch_warnings():
        # the DCD reader announces an upcoming (3.0) behaviour change on opening
        warnings.simplefilter("ignore", DeprecationWarning)
        u = mda.Universe(str(topology), str(trajectory))
    group = u.select_atoms(selection)
    elements = list(group.elements) if hasattr(group, "elements") else [""] * len(group)
    keep = [i for i, (r, n, e) in enumerate(zip(group.resnames, group.names, elements))
            if residue_kind(r) is not None and not _is_hydrogen(n, e)]
    if not keep:
        raise ValueError(f"{trajectory}: selection {selection!r} has no protein/nucleic "
                         "heavy atoms")
    group, elements = group[keep], [elements[i] for i in keep]

    # free molecules with standard residue names (e.g. an arginine ligand) are not
    # part of the polymer: drop residues with no covalent link to a neighbour
    chains = group.chainIDs if hasattr(group, "chainIDs") else group.segids
    res_ids = list(dict.fromkeys(group.resindices))
    per_res = {r: (None, None, {}) for r in res_ids}
    for atom, chain in zip(group, chains):
        c, k, pos = per_res[atom.resindex]
        pos.setdefault(canonical_atom_name(atom.name), atom.position.copy())
        per_res[atom.resindex] = (str(chain), residue_kind(atom.resname), pos)
    free = {res_ids[i] for i in unlinked_residues([per_res[r] for r in res_ids])}
    if free:
        logger.info("%s: ignored %d residue(s) not linked to a polymer chain (free ligands "
                    "with standard names)", Path(trajectory).name, len(free))
        mask = [i for i, r in enumerate(group.resindices) if r not in free]
        group, elements = group[mask], [elements[i] for i in mask]

    chains = group.chainIDs if hasattr(group, "chainIDs") else group.segids
    residue_pos = {r: i for i, r in enumerate(dict.fromkeys(group.resindices))}
    rec = _Record([], [], [], [], [], [], [], np.empty((0, 3)))
    rec.resnames = [str(r.resname) for r in group.residues]
    seen, atom_index = set(), []
    for i, atom in enumerate(group):
        key = (residue_pos[atom.resindex], canonical_atom_name(atom.name))
        if key in seen:
            continue
        seen.add(key)
        atom_index.append(i)
        rec.keys.append(key)
        rec.chain.append(str(chains[i]))
        rec.resid.append(int(atom.resid))
        rec.name.append(key[1])
        rec.element.append(str(elements[i]))
        rec.kind.append(residue_kind(atom.resname))

    frames = select_frames(len(u.trajectory), stride, max_frames)
    xyz = np.empty((len(frames), len(atom_index), 3), dtype=np.float64)
    for k, _ in enumerate(u.trajectory[frames]):
        xyz[k] = group.positions[atom_index]
    return rec, xyz, frames


def _as_paths(value) -> list[Path]:
    return [Path(value)] if isinstance(value, (str, Path)) else [Path(v) for v in value]


def load_trajectory(
    topology,
    trajectories,
    selection: str = "all",
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Ensemble:
    """Read one or more MD trajectories of the same molecule.

    ``topology`` is either one file shared by all trajectories or one file per
    trajectory (in the same order). Topologies may differ in atom order,
    hydrogens, protonation or force-field naming; atoms are matched by residue
    position and name, as for structure ensembles. Frames of all trajectories
    become members of one ensemble, labelled by replica, so they are clustered
    on a common set of states. ``stride`` and ``max_frames`` apply per trajectory.
    """
    paths, tops = _as_paths(trajectories), _as_paths(topology)
    if len(tops) == 1:
        tops = tops * len(paths)
    if len(tops) != len(paths):
        raise ValueError(f"{len(tops)} topologies for {len(paths)} trajectories: give one "
                         "shared topology or one per trajectory")
    labels = replica_labels(paths)

    records, stacks, frames_per = [], [], []
    for top, path, label in zip(tops, paths, labels):
        rec, xyz, frames = _trajectory_record(top, path, selection, stride, max_frames)
        records.append(rec)
        stacks.append(xyz)
        frames_per.append(frames)
        logger.info("Loaded %d frames from %s (replica %s, stride %d)",
                    len(frames), path.name, label, stride)

    atoms, indices = _match_atoms(records, labels, "trajectories")
    coords = np.concatenate([xyz[:, idx] for xyz, idx in zip(stacks, indices)])
    names, replica, frame_idx = [], [], []
    for label, frames in zip(labels, frames_per):
        prefix = f"{label}:" if len(paths) > 1 else ""
        names += [f"{prefix}frame_{i}" for i in frames]
        replica += [label] * len(frames)
        frame_idx += frames
    ens = Ensemble(names=names, sources=[None] * len(names), atoms=atoms, coords=coords,
                   replica=replica, frame=frame_idx)
    logger.info(
        "Trajectory ensemble: %d frames from %d trajectories; %s, %d residues, %d heavy atoms",
        len(ens), len(paths), ens.molecule_type, ens.n_residues, len(atoms),
    )
    return ens


# ---------------------------------------------------------------------------
# Atom selection and output
# ---------------------------------------------------------------------------

_RANGE = re.compile(r"^(?:([^:\s]+):)?(-?\d+)(?:-(-?\d+))?$")


def parse_residue_ranges(spec: str) -> list[tuple[Optional[str], int, int]]:
    """"1-70" or "A:1-70,B:5-30" -> [(chain or None, first, last)], ranges inclusive."""
    out = []
    for item in (s.strip() for s in spec.split(",")):
        m = _RANGE.match(item)
        if not m:
            raise ValueError(f"input.residues: cannot read {item!r} (expected e.g. 1-70 or "
                             "A:1-70, items separated by commas)")
        chain, first, last = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        if last < first:
            raise ValueError(f"input.residues: range {item!r} ends before it starts")
        out.append((chain, first, last))
    return out


def restrict_residues(ens: Ensemble, spec: str) -> Ensemble:
    """The ensemble restricted to the residues of ``spec`` (parse_residue_ranges), by the
    residue numbers and chains of the input files; residue positions are renumbered."""
    a = ens.atoms
    keep = np.zeros(len(a), dtype=bool)
    for chain, first, last in parse_residue_ranges(spec):
        keep |= (a.resid >= first) & (a.resid <= last) & (
            True if chain is None else a.chain == chain)
    if not keep.any():
        raise ValueError(f"input.residues {spec!r} selects no residue of the input (residue "
                         f"numbers {a.resid.min()}-{a.resid.max()}, chains "
                         f"{', '.join(sorted(set(a.chain.tolist())))})")
    atoms = a.subset(keep)
    atoms.residue = np.unique(atoms.residue, return_inverse=True)[1]
    out = Ensemble(names=ens.names, sources=ens.sources, atoms=atoms,
                   coords=ens.coords[:, keep], replica=ens.replica, frame=ens.frame)
    logger.info("input.residues %s: %d of %d residues (%d heavy atoms) compared", spec,
                out.n_residues, ens.n_residues, len(atoms))
    return out

def atom_mask(atoms: Atoms, selection: str) -> np.ndarray:
    """Boolean mask for 'heavy-atoms', 'backbone' or 'anchor' (CA / C3')."""
    if selection == "heavy-atoms":
        return np.ones(len(atoms), dtype=bool)
    if selection == "backbone":
        return np.array([
            n in (PROTEIN_BACKBONE if k == "protein" else NUCLEIC_BACKBONE)
            for n, k in zip(atoms.name, atoms.kind)
        ])
    if selection == "anchor":
        return np.array([n == ANCHOR_ATOM[k] for n, k in zip(atoms.name, atoms.kind)])
    raise ValueError(f"Unknown atom selection {selection!r}")


def describe_selection(atoms: Atoms, mask: np.ndarray) -> dict:
    """Which atoms a selection compares, per molecule kind (for logs and the run summary).

    Returns {kind: {"n_atoms": int, "n_residues": int,
                    "atoms_by_residue": {resname: [atom names]}}}.
    """
    out = {}
    sel = atoms.subset(mask)
    for kind in ("nucleic", "protein"):
        k = sel.kind == kind
        if not k.any():
            continue
        names: dict[str, list[str]] = {}
        for rn, an in zip(sel.resname[k], sel.name[k]):
            lst = names.setdefault(str(rn), [])
            if an not in lst:
                lst.append(str(an))
        out[kind] = {
            "n_atoms": int(k.sum()),
            "n_residues": int(len(np.unique(sel.residue[k]))),
            "atoms_by_residue": dict(sorted(names.items())),
        }
    return out


CHAIN_LETTERS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def pdb_chain_ids(chains: np.ndarray) -> dict[str, str]:
    """One-character PDB chain ID for every chain or segment name.

    Names that are already distinct single characters are kept. Otherwise (e.g.
    MD segment names such as PROA and PROB, which would both become "P") every
    chain gets the next letter in order of appearance, and the mapping is logged.
    """
    names = list(dict.fromkeys(str(c) for c in chains))
    if all(len(n) == 1 for n in names):
        return {n: n for n in names}
    if len(names) > len(CHAIN_LETTERS):
        raise ValueError(f"{len(names)} chains cannot be written to one PDB file")
    mapping = dict(zip(names, CHAIN_LETTERS))
    logger.debug("PDB chain IDs: %s", ", ".join(f"{k or '(blank)'} -> {v}"
                                                for k, v in mapping.items()))
    return mapping


def write_pdb(path: Path, atoms: Atoms, xyz: np.ndarray, renumber: bool = False) -> None:
    """Write one member as a minimal PDB file (medoids, and input for US-align).

    ``renumber`` numbers the residues 1, 2, ... in order (input for US-align, which
    pairs residues by number: insertion codes and numbers above 9999 cannot be
    written unambiguously)."""
    lines = _atom_lines(atoms, xyz, renumber)
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def write_models_pdb(path: Path, atoms: Atoms, models: list[np.ndarray], names: list[str]) -> None:
    """Write several members as one multi-model PDB file (MODEL 1..n, in the given order).

    The member name of each model is recorded in a REMARK line before it.
    """
    lines = []
    for k, (xyz, name) in enumerate(zip(models, names), start=1):
        lines.append(f"REMARK   1 MODEL {k} {name}")
        lines.append(f"MODEL     {k:4d}")
        lines += _atom_lines(atoms, xyz)
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def _atom_lines(atoms: Atoms, xyz: np.ndarray, renumber: bool = False) -> list[str]:
    chain_id = pdb_chain_ids(atoms.chain)
    lines = []
    for i in range(len(atoms)):
        name = atoms.name[i]
        padded = f" {name:<3}" if len(name) < 4 else name
        element = atoms.element[i] or name.lstrip("0123456789")[:1]
        x, y, z = xyz[i]
        resid = int(atoms.residue[i]) + 1 if renumber else int(atoms.resid[i])
        lines.append(
            f"ATOM  {(i + 1) % 100000:5d} {padded:4s} {atoms.resname[i]:>3s} "
            f"{chain_id[str(atoms.chain[i])] or 'A':1s}{resid % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    return lines

