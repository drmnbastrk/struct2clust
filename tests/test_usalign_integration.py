"""TM-score distances with the real US-align program (skipped where it is not installed).

The executable is taken from the environment variable STRUCT2CLUST_USALIGN, else
``USalign`` on PATH.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from helpers import atom_line

from struct2clust import metrics
from struct2clust.structures import load_ensemble

EXE = os.environ.get("STRUCT2CLUST_USALIGN", "USalign")

pytestmark = pytest.mark.skipif(
    shutil.which(EXE) is None and not Path(EXE).is_file(),
    reason="US-align not installed (set STRUCT2CLUST_USALIGN or put USalign on PATH)")


def helix_pdb(path, rng, noise):
    """A 30-residue alanine alpha helix (backbone N, CA, C, O, and CB), every atom
    displaced by Gaussian noise of SD `noise` (A)."""
    lines, serial = [], 1
    offsets = {"N": (-0.6, -1.0, -0.5), "CA": (0.0, 0.0, 0.0), "C": (0.6, 1.0, 0.5),
               "O": (1.4, 1.6, 0.8), "CB": (-0.9, 0.9, -0.6)}
    for i in range(30):
        angle = np.radians(100.0 * i)
        ca = np.array([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * i])
        for name, off in offsets.items():
            xyz = ca + off + rng.normal(scale=noise, size=3)
            lines.append(atom_line(serial, name, "ALA", "A", i + 1, xyz, name[0]))
            serial += 1
    path.write_text("\n".join(lines) + "\nEND\n")


def test_tm_distances_with_usalign(tmp_path):
    rng = np.random.default_rng(0)
    files = []
    for k in range(3):
        files.append(tmp_path / f"helix_{k}.pdb")
        helix_pdb(files[-1], rng, 0.1)
    shutil.copy(files[0], tmp_path / "copy_of_first.pdb")    # an identical structure
    ens = load_ensemble(tmp_path)
    d = metrics.distance_matrix(ens, "tm", usalign_exe=EXE, n_jobs=2)
    assert d.shape == (4, 4)
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)
    assert np.all((d >= 0) & (d < 1))
    k = ens.names.index("copy_of_first")
    first = ens.names.index(files[0].stem)
    assert d[k, first] <= 1e-3                 # identical structures: TM-score 1
    assert d.max() < 0.1                       # one fold: TM-scores above 0.9
