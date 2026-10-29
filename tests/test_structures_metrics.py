import numpy as np
import pytest
from helpers import RNA_ATOMS, rna_pdb
from scipy.spatial.transform import Rotation

from struct2clust import metrics
from struct2clust.structures import (
    atom_mask,
    describe_selection,
    load_ensemble,
    load_trajectory,
    residue_kind,
    write_pdb,
)


def _reference_rmsd(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(b0.T @ a0)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return np.sqrt(((b0 @ r.T - a0) ** 2).sum(1).mean())


def test_residue_kind():
    assert residue_kind("G") == "nucleic"
    assert residue_kind("G5") == "nucleic"      # force-field 5' terminus
    assert residue_kind("HIE") == "protein"
    assert residue_kind("HOH") is None
    assert residue_kind("MG") is None
    assert residue_kind("NA") is None                    # sodium, not a nucleotide


@pytest.mark.parametrize("name,base", [
    ("G5", "G"), ("A3", "A"), ("C5", "C"), ("U3", "U"),     # RNA termini, any base
    ("RG5", "G"), ("RU3", "U"),                              # AMBER RNA names
    ("DA5", "DA"), ("DT3", "DT"), ("DG5", "DG"), ("DC3", "DC"),  # DNA termini
    ("AN", "A"), ("DTN", "DT"),                              # single nucleotide (both ends)
])
def test_terminal_nucleotides_of_every_base(name, base):
    from struct2clust.structures import canonical_resname

    assert residue_kind(name) == "nucleic"
    assert canonical_resname(name) == base


@pytest.mark.parametrize("name", ["X5", "N3", "5G", "G5'"])
def test_unknown_terminal_names_are_not_nucleotides(name):
    assert residue_kind(name) is None


def test_rmsd_matrix_matches_explicit_kabsch():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(6, 20, 3)) * 3
    coords[1] = Rotation.random(random_state=1).apply(coords[0]) + [4, -2, 7]  # rigid copy
    coords[2] = coords[0] * [1, 1, -1]                                           # mirror image
    d = metrics.rmsd_matrix(coords, n_jobs=2)
    # RMSD comes from a difference of sums of squares; for identical structures the
    # cancellation leaves ~1e-7 A of round-off (far below any physical meaning).
    assert d[0, 1] == pytest.approx(0, abs=1e-6)
    assert d[0, 2] > 0.1  # a reflection is not a rotation
    for i in range(6):
        for j in range(i + 1, 6):
            assert d[i, j] == pytest.approx(_reference_rmsd(coords[i], coords[j]), abs=1e-6)
    assert np.allclose(d, d.T)


def test_load_ensemble_filters_and_matches_atoms(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.normal(size=(24, 3)) * 5
    for k in range(3):
        rna_pdb(tmp_path / f"{k + 1}_model.pdb", base + rng.normal(scale=0.5, size=base.shape))
    # a 10th file name must sort after 2 and 3 (natural order)
    rna_pdb(tmp_path / "10_model.pdb", base)
    ens = load_ensemble(tmp_path)
    assert ens.names == ["1_model", "2_model", "3_model", "10_model"]
    assert ens.coords.shape == (4, 24, 3)          # hydrogens and water dropped
    assert ens.molecule_type == "nucleic"
    assert ens.n_residues == 3
    assert atom_mask(ens.atoms, "backbone").sum() == 21   # N9 is not backbone
    assert atom_mask(ens.atoms, "anchor").sum() == 3      # one C3' per residue


def test_load_ensemble_rejects_different_sequence(tmp_path):
    xyz = np.zeros((24, 3))
    rna_pdb(tmp_path / "a.pdb", xyz, seq="GAC")
    rna_pdb(tmp_path / "b.pdb", xyz, seq="GAU")
    with pytest.raises(ValueError, match="residue sequence differs"):
        load_ensemble(tmp_path)


def test_multimodel_file_and_write_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    xyz = rng.normal(size=(24, 3)) * 4
    single = tmp_path / "one.pdb"
    rna_pdb(single, xyz, extra=False)
    body = single.read_text().replace("END\n", "")
    multi = tmp_path / "multi.pdb"
    multi.write_text("".join(f"MODEL {k}\n{body}ENDMDL\n" for k in (1, 2, 3)) + "END\n")
    ens = load_ensemble(multi)
    assert ens.names == ["multi_m1", "multi_m2", "multi_m3"]

    out = tmp_path / "out"
    out.mkdir()
    for k in range(2):
        write_pdb(out / f"w{k}.pdb", ens.atoms, ens.coords[k])
    again = load_ensemble(out)
    assert np.allclose(again.coords, ens.coords[:2], atol=1e-3)
    assert list(again.atoms.name) == list(ens.atoms.name)


def test_trajectory_loader(tmp_path):
    rng = np.random.default_rng(3)
    xyz = rng.normal(size=(24, 3)) * 4
    single = tmp_path / "top.pdb"
    rna_pdb(single, xyz, extra=False)
    body = single.read_text().replace("END\n", "")
    traj = tmp_path / "traj.pdb"
    traj.write_text("".join(f"MODEL {k}\n{body}ENDMDL\n" for k in range(6)) + "END\n")
    ens = load_trajectory(single, traj, selection="all", stride=2)
    assert ens.names == ["frame_0", "frame_2", "frame_4"]
    assert ens.coords.shape == (3, 24, 3)


def test_deduplicate_keeps_first():
    d = np.array([[0, 0.05, 3], [0.05, 0, 3], [3, 3, 0]], dtype=float)
    kept, log = metrics.deduplicate(d, 0.1)
    assert kept.tolist() == [0, 2]
    assert log == [(0, 1, 0.05)]


def test_deduplicate_does_not_depend_on_the_input_order():
    """b has two copies (a, c); a and c are not copies of each other. b is kept,
    whatever the order of the input."""
    pos = {"a": 0.0, "b": 0.08, "c": 0.16, "d": 5.0}
    for order in (["a", "b", "c", "d"], ["c", "d", "b", "a"], ["d", "a", "c", "b"]):
        x = np.array([pos[k] for k in order])
        d = np.abs(x[:, None] - x[None, :])
        kept, log = metrics.deduplicate(d, 0.1, names=order)
        assert sorted(order[i] for i in kept) == ["b", "d"]
        assert sorted(order[j] for _, j, _ in log) == ["a", "c"]


def test_parse_usalign_table():
    out = (
        "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali\n"
        "a.pdb:A\tb.pdb:A\t0.8000\t0.6000\t1.20\t1.0\t1.0\t1.0\t42\t42\t42\n"
    )
    assert metrics.parse_usalign_table(out) == pytest.approx(0.7)


def test_missing_usalign_is_clear_error():
    with pytest.raises(FileNotFoundError, match="US-align"):
        metrics.find_usalign("definitely-not-usalign-xyz")


def test_describe_selection_lists_atoms_per_residue(tmp_path):
    rng = np.random.default_rng(6)
    for k in range(2):
        rna_pdb(tmp_path / f"s{k}.pdb", rng.normal(size=(24, 3)) * 4)
    ens = load_ensemble(tmp_path)
    info = describe_selection(ens.atoms, atom_mask(ens.atoms, "heavy-atoms"))
    assert list(info) == ["nucleic"]
    assert info["nucleic"]["n_atoms"] == 24 and info["nucleic"]["n_residues"] == 3
    assert info["nucleic"]["atoms_by_residue"]["G"] == RNA_ATOMS
    bb = describe_selection(ens.atoms, atom_mask(ens.atoms, "backbone"))
    assert "N9" not in bb["nucleic"]["atoms_by_residue"]["G"]


def test_replica_labels():
    from pathlib import Path

    from struct2clust.structures import replica_labels

    a = Path("/x/system/SET1/trajectory/nucleic.dcd")
    b = Path("/x/system/SET2/trajectory/nucleic.dcd")
    assert replica_labels([a, b]) == ["SET1", "SET2"]
    assert replica_labels([Path("/d/run_a.dcd"), Path("/d/run_b.dcd")]) == ["run_a", "run_b"]
    assert replica_labels([Path("/p/C1/SET1/t.dcd"), Path("/p/C2/SET1/t.dcd")]) == ["C1", "C2"]
    assert replica_labels([a]) == ["nucleic"]
    assert replica_labels([a, a]) == ["rep1", "rep2"]


def test_two_trajectories_become_one_ensemble(tmp_path):
    rng = np.random.default_rng(9)
    xyz = rng.normal(size=(24, 3)) * 4
    top = tmp_path / "top.pdb"
    rna_pdb(top, xyz, extra=False)
    body = top.read_text().replace("END\n", "")
    paths = []
    for rep, n in (("SET1", 4), ("SET2", 6)):
        (tmp_path / rep).mkdir()
        p = tmp_path / rep / "traj.pdb"
        p.write_text("".join(f"MODEL {k}\n{body}ENDMDL\n" for k in range(n)) + "END\n")
        paths.append(p)
    ens = load_trajectory(top, paths, selection="all")
    assert len(ens) == 10
    assert ens.replicas == ["SET1", "SET2"]
    assert ens.names[0] == "SET1:frame_0" and ens.names[-1] == "SET2:frame_5"
    sub = ens.subset([0, 9])
    assert sub.replica == ["SET1", "SET2"] and sub.frame == [0, 5]


def test_canonical_names():
    from struct2clust.structures import canonical_atom_name, canonical_resname

    assert canonical_resname("G5") == canonical_resname("G3") == canonical_resname("RG") == "G"
    assert canonical_resname("HIE") == canonical_resname("HSD") == "HIS"
    assert canonical_resname("DA") == "DA"          # DNA stays distinct from RNA
    assert canonical_atom_name("O1P") == "OP1"
    assert canonical_atom_name("C4*") == "C4'"


def _traj_file(path, models):
    path.write_text("".join(f"MODEL {k}\n{m}ENDMDL\n" for k, m in enumerate(models)) + "END\n")


def test_trajectories_with_different_topologies(tmp_path):
    """Same molecule, differently prepared topologies: atoms are matched by residue and name."""
    from helpers import atom_line

    rng = np.random.default_rng(10)
    xyz = rng.normal(size=(24, 3)) * 4
    # topology A: canonical names, file order
    a_lines = []
    k = 0
    for r, base in enumerate("GAC", start=1):
        for name in RNA_ATOMS:
            a_lines.append(atom_line(k + 1, name, base, "A", r, xyz[k], name[0]))
            k += 1
    body_a = "\n".join(a_lines) + "\n"
    # topology B: terminal residue names, reversed atom order inside residues, extra hydrogen
    b_lines = []
    k = 0
    for r, base in enumerate("GAC", start=1):
        resname = {1: "G5", 3: "C3"}.get(r, base)
        block = []
        for name in RNA_ATOMS:
            block.append(atom_line(k + 1, name, resname, "A", r, xyz[k], name[0]))
            k += 1
        b_lines += block[::-1] + [atom_line(900 + r, "H1'", resname, "A", r, (0, 0, 0), "H")]
    body_b = "\n".join(b_lines) + "\n"
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    (tmp_path / "A" / "top.pdb").write_text(body_a + "END\n")
    (tmp_path / "B" / "top.pdb").write_text(body_b + "END\n")
    _traj_file(tmp_path / "A" / "traj.pdb", [body_a] * 3)
    _traj_file(tmp_path / "B" / "traj.pdb", [body_b] * 5)
    ens = load_trajectory([tmp_path / "A" / "top.pdb", tmp_path / "B" / "top.pdb"],
                          [tmp_path / "A" / "traj.pdb", tmp_path / "B" / "traj.pdb"])
    assert ens.replicas == ["A", "B"] and len(ens) == 8
    assert ens.coords.shape == (8, 24, 3)
    assert np.allclose(ens.coords[0], ens.coords[-1])   # same conformation after matching


def test_trajectories_of_different_molecules_are_rejected(tmp_path):
    rng = np.random.default_rng(11)
    xyz = rng.normal(size=(24, 3))
    for name, seq in (("A", "GAC"), ("B", "GAU")):
        (tmp_path / name).mkdir()
        rna_pdb(tmp_path / name / "top.pdb", xyz, seq=seq, extra=False)
        body = (tmp_path / name / "top.pdb").read_text().replace("END\n", "")
        _traj_file(tmp_path / name / "traj.pdb", [body] * 2)
    with pytest.raises(ValueError, match="same molecule"):
        load_trajectory([tmp_path / "A" / "top.pdb", tmp_path / "B" / "top.pdb"],
                        [tmp_path / "A" / "traj.pdb", tmp_path / "B" / "traj.pdb"])


def test_free_ligand_with_a_standard_residue_name_is_ignored(tmp_path):
    """An arginine ligand next to an RNA (as in PDB 1ARJ) is not part of the polymer."""
    from helpers import atom_line

    rng = np.random.default_rng(12)
    for k in range(3):
        xyz = rng.normal(size=(24, 3)) * 4
        rna_pdb(tmp_path / f"s{k}.pdb", xyz, extra=False)
        text = (tmp_path / f"s{k}.pdb").read_text().replace("END\n", "")
        arg = [atom_line(500 + i, n, "ARG", "N", 1, xyz[i] + 30, n[0], record="HETATM")
               for i, n in enumerate(["N", "CA", "C", "O", "CB"])]
        (tmp_path / f"s{k}.pdb").write_text(text + "\n".join(arg) + "\nEND\n")
    ens = load_ensemble(tmp_path)
    assert ens.molecule_type == "nucleic" and ens.n_residues == 3


def test_unlinked_residues():
    from struct2clust.structures import unlinked_residues

    bonded = [("A", "nucleic", {"O3'": (0, 0, 0)}), ("A", "nucleic", {"P": (1.6, 0, 0),
                                                                      "O3'": (5, 0, 0)}),
              ("A", "nucleic", {"P": (6.6, 0, 0)}), ("A", "protein", {"N": (50, 0, 0)})]
    # the unlinked protein residue is dropped only as HETATM or alone in its chain
    assert unlinked_residues(bonded) == []
    assert unlinked_residues(bonded, [False, False, False, True]) == [3]
    alone = bonded[:3] + [("N", "protein", {"N": (50, 0, 0)})]
    assert unlinked_residues(alone) == [3]
    # a polymer residue with a broken backbone (ATOM, same chain) is kept
    broken = bonded[:3] + [("A", "nucleic", {"P": (30, 0, 0)})]
    assert unlinked_residues(broken) == []
    assert unlinked_residues([("A", "protein", {"CA": (0, 0, 0)})] * 3) == []   # CA-only


def test_select_frames_spreads_over_whole_trajectory():
    from struct2clust.structures import select_frames

    assert select_frames(10) == list(range(10))
    assert select_frames(10, stride=3) == [0, 3, 6, 9]
    frames = select_frames(1000, max_frames=5)
    assert frames == [0, 250, 500, 749, 999]          # not the first five
    assert select_frames(1000, stride=10, max_frames=3) == [0, 500, 990]


def test_tm_superposition_fits_the_core_not_the_moved_tail():
    """TM-score superposition places the unchanged core exactly and leaves the moved
    tail off; least squares spreads the error over both."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(3)
    core = np.cumsum(rng.normal(size=(60, 3)) * 2.2, axis=0)
    tail = core[-1] + np.cumsum(rng.normal(size=(15, 3)) * 2.2, axis=0)
    mobile = np.vstack([core, tail])
    target = mobile @ Rotation.from_euler("xyz", [20, 40, 60], degrees=True).as_matrix().T
    target[60:] += [12.0, 0.0, 0.0]                  # the tail moves
    d0 = metrics.tm_d0(75, "protein")
    for fit, core_error in (("tm", 1e-6), ("rmsd", 0.5)):
        r, t = metrics.superpose(mobile, target, fit, d0)
        dev = np.linalg.norm(mobile @ r.T + t - target, axis=1)
        if fit == "tm":
            assert dev[:60].max() < core_error
        else:
            assert dev[:60].mean() > core_error


def test_tm_d0():
    assert metrics.tm_d0(42, "nucleic") == pytest.approx(1.365, abs=1e-3)
    assert metrics.tm_d0(20, "nucleic") == pytest.approx(0.6)
    assert metrics.tm_d0(300, "protein") == pytest.approx(6.36, abs=0.01)
    assert metrics.tm_d0(10, "protein") == pytest.approx(0.5)


def test_pdb_chain_ids_stay_distinct():
    from struct2clust.structures import pdb_chain_ids

    assert pdb_chain_ids(np.array(["A", "A", "B"])) == {"A": "A", "B": "B"}
    ids = pdb_chain_ids(np.array(["PROA", "PROB", "RNA", "PROA"]))
    assert len(set(ids.values())) == 3 and all(len(v) == 1 for v in ids.values())


def _altloc_pdb(path, rows):
    """rows: (atom name, residue number, altloc, occupancy, xyz)."""
    lines = []
    for i, (name, resid, alt, occ, xyz) in enumerate(rows, start=1):
        lines.append(f"ATOM  {i:5d}  {name:<3s}{alt or ' '}ALA A{resid:4d}    "
                     f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}  0.00           "
                     f"{name[0]}")
    path.write_text("\n".join(lines) + "\nEND\n")


def _conformer_coords(tmp_path, rows):
    from struct2clust.structures import load_ensemble

    for k in range(2):
        _altloc_pdb(tmp_path / f"m{k}.pdb", rows)
    ens = load_ensemble(tmp_path)
    return {(int(r), str(n)): tuple(x) for r, n, x in
            zip(ens.atoms.residue, ens.atoms.name, ens.coords[0])}


BACKBONE = [("N", 1, "", 1.0, (0, 0, 0)), ("CA", 1, "", 1.0, (1.4, 0, 0)),
            ("C", 1, "", 1.0, (2.0, 1.2, 0))]


def test_alternate_conformers_are_never_mixed(tmp_path):
    """Conformer B is listed first for N, A first for CA; with equal occupancy A is
    kept, and N, present only in B, is missing rather than taken from B."""
    rows = BACKBONE + [("N", 2, "B", 0.5, (3.3, 1.3, 9)), ("CA", 2, "A", 0.5, (4.0, 2.5, 0)),
                       ("CA", 2, "B", 0.5, (4.0, 2.5, 9)), ("C", 2, "", 1.0, (5.4, 2.5, 0))]
    got = _conformer_coords(tmp_path, rows)
    assert got[(1, "CA")] == (4.0, 2.5, 0.0)           # conformer A
    assert (1, "N") not in got                          # never borrowed from B
    assert all(xyz[2] != 9 for xyz in got.values())     # nothing from B


def test_conformer_with_the_highest_occupancy_is_kept(tmp_path):
    rows = BACKBONE + [("N", 2, "A", 0.3, (3.3, 1.3, 0)), ("N", 2, "B", 0.7, (3.3, 1.3, 9)),
                       ("CA", 2, "A", 0.3, (4.0, 2.5, 0)), ("CA", 2, "B", 0.7, (4.0, 2.5, 9)),
                       ("C", 2, "", 1.0, (5.4, 2.5, 0))]
    got = _conformer_coords(tmp_path, rows)
    assert got[(1, "N")][2] == 9 and got[(1, "CA")][2] == 9   # all of conformer B
    assert got[(1, "C")] == (5.4, 2.5, 0.0)                   # unlabelled atom kept


def test_deduplicate_ties_go_to_the_more_central_structure():
    """a, b and c are all mutual copies (two neighbours each); b lies between a and c
    and is kept whatever the names and the input order."""
    pos = {"zz_b": 0.05, "aa_a": 0.0, "mm_c": 0.09}
    for order in (["aa_a", "zz_b", "mm_c"], ["mm_c", "aa_a", "zz_b"]):
        x = np.array([pos[k] for k in order])
        kept, log = metrics.deduplicate(np.abs(x[:, None] - x[None, :]), 0.1, names=order)
        assert [order[i] for i in kept] == ["zz_b"]


def _toy_atoms(residues):
    """Atoms for [(resname, kind, [atom names]), ...]."""
    from struct2clust.structures import Atoms

    rows = [(r, resname, kind, name) for r, (resname, kind, names) in enumerate(residues)
            for name in names]
    col = lambda j: np.array([row[j] for row in rows], dtype=object)  # noqa: E731
    return Atoms(chain=np.array(["A"] * len(rows), dtype=object), resid=col(0) + 1,
                 resname=col(1), name=col(3), element=np.array([n[0] for n in col(3)]),
                 kind=col(2), residue=col(0).astype(int))


def test_equivalent_atoms_naming_is_not_a_difference():
    """A rigid copy whose equivalent atoms are named the other way round (Asp OD1/OD2,
    Leu CD1/CD2, Phe ring, C-terminal O/OXT, a 5'-phosphate's three oxygens) is at
    RMSD 0 once equivalent atoms are matched; a real change still counts."""
    atoms = _toy_atoms([
        ("ASP", "protein", ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]),
        ("LEU", "protein", ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"]),
        ("PHE", "protein", ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
                            "OXT"]),
        ("G", "nucleic", ["OP3", "P", "OP1", "OP2", "O5'", "C5'"]),
    ])
    groups = metrics.equivalent_atom_groups(atoms)
    assert sum(len(index) for index, _ in groups) == 5
    rng = np.random.default_rng(3)
    a = rng.normal(size=(len(atoms), 3)) * 3
    names = list(atoms.name)

    def at(r, n):
        return next(i for i, (rr, nn) in enumerate(zip(atoms.residue, names))
                    if rr == r and nn == n)

    perm = np.arange(len(atoms))
    for r, x, y in [(0, "OD1", "OD2"), (1, "CD1", "CD2"), (2, "CD1", "CD2"), (2, "CE1", "CE2"),
                    (2, "O", "OXT")]:
        perm[[at(r, x), at(r, y)]] = perm[[at(r, y), at(r, x)]]
    op = [at(3, "OP1"), at(3, "OP2"), at(3, "OP3")]
    perm[op] = perm[[op[1], op[2], op[0]]]                       # cyclic renaming
    b = Rotation.random(random_state=4).apply(a[perm]) + [1, 2, 3]
    c = b.copy()
    c[at(1, "CA")] += 2.0                                         # a real difference
    coords = np.stack([a, b, c])
    plain = metrics.rmsd_matrix(coords, n_jobs=1)
    matched = metrics.rmsd_matrix(coords, n_jobs=1, groups=groups)
    assert plain[0, 1] > 1.0
    assert matched[0, 1] == pytest.approx(0, abs=1e-6)
    assert matched[0, 2] == pytest.approx(_reference_rmsd(a, c[np.argsort(perm)]),
                                          abs=1e-6)
    assert matched[0, 2] > 0.1
    assert np.all(matched <= plain + 1e-9)


def test_equivalent_atoms_match_exhaustive_search_on_noisy_copies():
    """With few groups, the alternating renaming reaches the minimum over all namings."""
    import itertools

    atoms = _toy_atoms([("ASP", "protein", ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]),
                        ("VAL", "protein", ["N", "CA", "C", "O", "CB", "CG1", "CG2"]),
                        ("ARG", "protein", ["N", "CA", "CZ", "NE", "NH1", "NH2"])])
    groups = metrics.equivalent_atom_groups(atoms)
    pairs = [(6, 7), (13, 14), (19, 20)]
    rng = np.random.default_rng(7)
    a = rng.normal(size=(len(atoms), 3)) * 2
    for _ in range(20):
        b = a + rng.normal(size=a.shape) * 0.6
        best = min(_reference_rmsd(a, b[_swap(len(a), pairs, flips)])
                   for flips in itertools.product([0, 1], repeat=3))
        got = metrics.rmsd_matrix(np.stack([a, b]), n_jobs=1, groups=groups)[0, 1]
        assert got == pytest.approx(best, abs=1e-6)


def _swap(n, pairs, flips):
    perm = np.arange(n)
    for (i, j), f in zip(pairs, flips):
        if f:
            perm[[i, j]] = perm[[j, i]]
    return perm


def _rna_lines(xyz, seq, skip=(), reverse=False, record=None):
    """PDB lines of an RNA chain like helpers.rna_pdb; ``skip`` holds (residue, atom)
    pairs to leave out, ``reverse`` writes each residue's atoms in reverse order and
    ``record`` maps residue numbers to HETATM."""
    from helpers import atom_line, link_chain

    xyz = link_chain(xyz, len(RNA_ATOMS), RNA_ATOMS.index("O3'"), RNA_ATOMS.index("P"))
    lines = []
    for r, base in enumerate(seq, start=1):
        names = list(enumerate(RNA_ATOMS))
        for j, name in (reversed(names) if reverse else names):
            if (r, name) in skip:
                continue
            rec = "HETATM" if record and r in record else "ATOM  "
            lines.append(atom_line(len(lines) + 1, name, base, "A", r,
                                   xyz[(r - 1) * len(RNA_ATOMS) + j], name[0], record=rec))
    return lines


def test_atom_missing_in_one_structure_is_left_out_of_all(tmp_path):
    """An atom absent from one structure is compared in none; the other atoms keep
    their own coordinates."""
    rng = np.random.default_rng(4)
    xyz = [rng.normal(size=(24, 3)) * 5 for _ in range(3)]
    for k in range(3):
        skip = {(2, "N9")} if k == 1 else set()
        (tmp_path / f"{k}.pdb").write_text("\n".join(_rna_lines(xyz[k], "GAC", skip)) + "\nEND\n")
    ens = load_ensemble(tmp_path)
    assert ens.coords.shape == (3, 23, 3)
    names = list(zip(ens.atoms.residue.tolist(), ens.atoms.name.tolist()))
    assert (1, "N9") not in names and (0, "N9") in names
    c1 = names.index((2, "C1'"))
    assert np.allclose(ens.coords[2, c1], np.round(xyz[2][2 * 8 + 6], 3))


def test_atoms_are_matched_by_name_not_by_file_order(tmp_path):
    rng = np.random.default_rng(5)
    xyz = rng.normal(size=(24, 3)) * 5
    (tmp_path / "a.pdb").write_text("\n".join(_rna_lines(xyz, "GAC")) + "\nEND\n")
    (tmp_path / "b.pdb").write_text("\n".join(_rna_lines(xyz, "GAC", reverse=True)) + "\nEND\n")
    ens = load_ensemble(tmp_path)
    assert np.allclose(ens.coords[0], ens.coords[1])
    assert metrics.rmsd_matrix(ens.coords, n_jobs=1)[0, 1] == pytest.approx(0, abs=1e-6)


def test_construct_with_an_extra_residue_is_rejected(tmp_path):
    """Different constructs must be cut to their common region before clustering."""
    rng = np.random.default_rng(6)
    rna_pdb(tmp_path / "a.pdb", rng.normal(size=(24, 3)) * 5, seq="GAC")
    rna_pdb(tmp_path / "b.pdb", rng.normal(size=(32, 3)) * 5, seq="GACU")
    with pytest.raises(ValueError, match="residue sequence differs"):
        load_ensemble(tmp_path)


def test_modified_residue_inside_a_chain_is_reported(tmp_path, caplog):
    """A modified nucleotide (pseudouridine, PSU) is not a recognised residue: it is
    left out, and the log says so because it leaves a gap in the chain; water is not
    reported."""
    rng = np.random.default_rng(7)
    for k in range(2):
        lines = _rna_lines(rng.normal(size=(24, 3)) * 5, ["G", "PSU", "C"], record={2})
        from helpers import atom_line
        lines.append(atom_line(999, "O", "HOH", "W", 100, (5, 5, 5), "O", record="HETATM"))
        (tmp_path / f"{k}.pdb").write_text("\n".join(lines) + "\nEND\n")
    with caplog.at_level("WARNING"):
        ens = load_ensemble(tmp_path)
    assert ens.n_residues == 2
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "PSU2:A" in text and "HOH" not in text


def test_residue_ranges_restrict_every_input():
    """input.residues keeps the residues given by number (optionally per chain) and
    renumbers their positions; unreadable or empty selections are refused."""
    from struct2clust.config import Config
    from struct2clust.structures import Atoms, Ensemble, parse_residue_ranges, restrict_residues

    assert parse_residue_ranges("1-70") == [(None, 1, 70)]
    assert parse_residue_ranges("A:1-3, B:7") == [("A", 1, 3), ("B", 7, 7)]
    for bad in ("1..70", "5-1", ""):
        with pytest.raises(ValueError):
            parse_residue_ranges(bad)
    with pytest.raises(ValueError):
        Config.model_validate({"input": {"residues": "x"}})
    resid = np.array([1, 1, 2, 3, 5, 5])
    chain = np.array(["A", "A", "A", "A", "B", "B"])
    atoms = Atoms(chain=chain, resid=resid, resname=np.array(["ALA"] * 6),
                  name=np.array(["N", "CA", "CA", "CA", "N", "CA"]), element=np.array(["N"] * 6),
                  kind=np.array(["protein"] * 6), residue=np.array([0, 0, 1, 2, 3, 3]))
    ens = Ensemble(names=["a", "b"], sources=[None, None], atoms=atoms,
                   coords=np.arange(36, dtype=float).reshape(2, 6, 3))
    sub = restrict_residues(ens, "A:2-3,B:5")
    assert sub.atoms.resid.tolist() == [2, 3, 5, 5] and sub.n_residues == 3
    assert sub.atoms.residue.tolist() == [0, 1, 2, 2]
    assert np.array_equal(sub.coords, ens.coords[:, 2:])
    with pytest.raises(ValueError, match="selects no residue"):
        restrict_residues(ens, "C:1-9")
