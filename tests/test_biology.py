import numpy as np
import pytest
from helpers import rna_pdb
from scipy.spatial.transform import Rotation

from struct2clust.biology import (
    angle_difference,
    characterise,
    circular_separation,
    circular_summary,
    contact_frequencies,
    dihedral,
    rmsf_per_residue,
    superpose_on_mean,
    torsion_angles,
)
from struct2clust.structures import load_ensemble


def test_dihedral_known_values():
    trans = np.array([[[1, 1, 0], [1, 0, 0], [-1, 0, 0], [-1, -1, 0]]], float)
    cis = np.array([[[1, 1, 0], [1, 0, 0], [-1, 0, 0], [-1, 1, 0]]], float)
    plus90 = np.array([[[1, 1, 0], [1, 0, 0], [-1, 0, 0], [-1, 0, 1]]], float)
    assert abs(dihedral(trans)[0]) == pytest.approx(180)
    assert dihedral(cis)[0] == pytest.approx(0, abs=1e-9)
    assert abs(dihedral(plus90)[0]) == pytest.approx(90)


def test_circular_statistics_handle_wraparound():
    mean, sd = circular_summary(np.array([179.0, -179.0, 178.0, -178.0]))
    assert abs(angle_difference(mean, 180.0)) < 1e-6   # not 0, as a linear mean would give
    assert sd < 3
    assert angle_difference(170, -170) == pytest.approx(20)


def test_rmsf_zero_for_rigid_copies_and_superposition():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 3)) * 5
    coords = np.stack([Rotation.random(random_state=i).apply(base) + i for i in range(5)])
    aligned = superpose_on_mean(coords)
    residue = np.repeat(np.arange(4), 3)
    assert np.allclose(rmsf_per_residue(aligned, residue), 0, atol=1e-6)
    moved = coords.copy()
    moved[:, :3] += rng.normal(scale=1.0, size=(5, 3, 3))   # residue 0 flexible
    rmsf = rmsf_per_residue(superpose_on_mean(moved), residue)
    assert rmsf[0] == rmsf.max() and rmsf[0] > 0.3


def test_contact_frequencies():
    residue = np.array([0, 1, 2, 3])
    close = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [0, 3, 0]], float)   # 0-3 in contact
    far = close.copy()
    far[3] = [0, 30, 0]
    freq = contact_frequencies(np.stack([close, far]), residue, cutoff=4.5, min_sep=2)
    assert freq[0, 3] == freq[3, 0] == 0.5
    assert freq[0, 1] == 0          # neighbours excluded by min_sep
    assert freq.sum() == 1.0


def test_characterise_rna_ensemble(tmp_path):
    rng = np.random.default_rng(1)
    base = rng.normal(size=(24, 3)) * 4
    for k in range(8):
        rna_pdb(tmp_path / f"s{k}.pdb", base + rng.normal(scale=0.3, size=base.shape))
    ens = load_ensemble(tmp_path)
    tors = torsion_angles(ens)
    assert {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"} <= set(tors["torsion"])
    assert tors.query("torsion == 'alpha'")["residue"].min() == 2   # needs O3' of residue 1
    labels = np.array([0, 0, 0, 0, 1, 1, 1, -1])
    bio = characterise(ens, labels, {0: 0, 1: 4})
    assert set(bio.rmsf["cluster"]) == {1, 2}
    assert set(bio.contacts) == {1, 2}
    assert bio.contacts[1].shape == (3, 3)
    assert set(bio.torsions["cluster"]) == {1, 2}
    assert (bio.torsions["n"].groupby(bio.torsions["cluster"]).max().tolist()) == [4, 3]
    assert bio.rg.loc[7, "cluster"] == 0   # noise
    assert not bio.torsion_spread.empty


def test_circular_separation():
    rng = np.random.default_rng(2)
    a = rng.normal(60, 5, 50)
    b = rng.normal(-60, 5, 50)
    spread_a = rng.uniform(-180, 180, 50)
    spread_b = rng.uniform(-180, 180, 50)
    assert circular_separation([a, b]) > 0.4          # distinct, tight clusters
    assert circular_separation([a, a + 1]) < 0.05       # same angle in both
    assert circular_separation([spread_a, spread_b]) < 0.2   # dispersed: no separation
    # wrap-around: 179 and -179 are the same conformation
    assert circular_separation([rng.normal(179, 2, 30), rng.normal(-179, 2, 30)]) < 0.05


def test_medoid_differences_locate_the_differing_residue():
    """Two medoids that differ only in the last residue: that residue carries the whole
    squared deviation, and the shares of the pair add up to 1."""
    from struct2clust.biology import main_residues, medoid_differences
    from struct2clust.pipeline import Superposition
    from struct2clust.structures import Atoms, Ensemble

    rng = np.random.default_rng(0)
    n_res, per = 6, 4
    a = rng.normal(scale=5.0, size=(n_res * per, 3))
    b = a.copy()
    b[-per:] += [8.0, 0.0, 0.0]                      # move the last residue only
    residue = np.repeat(np.arange(n_res), per)
    atoms = Atoms(chain=np.full(n_res * per, "A"), resid=residue + 1,
                  resname=np.full(n_res * per, "ALA"),
                  name=np.tile(["N", "CA", "C", "O"], n_res),
                  element=np.tile(["N", "C", "C", "O"], n_res),
                  kind=np.full(n_res * per, "protein"), residue=residue)
    ens = Ensemble(names=["a", "b"], sources=[None, None], atoms=atoms, coords=np.stack([a, b]))
    fit = Superposition(np.ones(n_res * per, bool), "rmsd")
    md = medoid_differences(ens, np.array([0, 1]), {0: 0, 1: 1}, fit)
    assert md["share"].sum() == pytest.approx(1.0)
    main, share = main_residues(md.set_index("residue")["share"])
    assert main == [n_res] and share > 0.5


def test_no_torsion_across_missing_residues(tmp_path):
    """Residues on either side of a gap (no O3'-P bond) share no torsion."""
    from struct2clust.biology import linked_to_next, torsion_angles

    rng = np.random.default_rng(2)
    for k in range(2):
        xyz = np.cumsum(rng.normal(size=(5 * 8, 3)), axis=0)
        rna_pdb(tmp_path / f"m{k}.pdb", xyz, seq="GACUG", extra=False)
    ens = load_ensemble(tmp_path)
    after_gap = ens.atoms.residue >= 2
    ens.coords[:, after_gap] += [15.0, 0.0, 0.0]       # residues 1-2 | 3-5
    assert linked_to_next(ens).tolist() == [True, False, True, True, False]
    tors = torsion_angles(ens)
    have = set(zip(tors["residue"], tors["torsion"]))
    assert (2, "alpha") in have and (4, "alpha") in have
    assert (3, "alpha") not in have and (2, "epsilon") not in have and (2, "zeta") not in have
