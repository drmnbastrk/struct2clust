"""Download the public example data sets into example/data/.

    python example/download_examples.py              # all data sets
    python example/download_examples.py tar ubiquitin

  tar        HIV-1 TAR RNA: 7 PDB entries, 88 structures
             (NMR ensembles 7JU1, 1ANR, 1ARJ, 2L8H, 1UTS, 1UUD, 1UUI; RCSB PDB, CC0)
  ubiquitin  Ubiquitin: PDB 2K39, 116-model ensemble refined against residual dipolar
             couplings (Lange et al. 2008; RCSB PDB, CC0)
  1rkj       Nucleolin RBD1-2 / RNA (PDB 1RKJ): 500 ns MD of the RNA bound in the
             complex and free in solution (Sabei, Hognon, Martin & Frezza 2023,
             doi:10.5281/zenodo.10040600, CC-BY-4.0). The 820 MB archive is reduced to
             the RNA atoms of 500 frames per run, spread evenly over each run.

The crystal-structure ensembles are built with prepare_crystal_ensembles.py.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
RCSB = "https://files.rcsb.org/download/{}.pdb"
TAR_ENTRIES = ["7JU1", "1ANR", "1ARJ", "2L8H", "1UTS", "1UUD", "1UUI"]
ZENODO_1RKJ = "https://zenodo.org/records/10040600/files/1RKJ.zip?download=1"
RUNS_1RKJ = ("complex", "rna")
FRAMES_1RKJ = 500


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  {dest} exists, skipped")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    tmp.rename(dest)
    return dest


def tar() -> None:
    for entry in TAR_ENTRIES:
        fetch(RCSB.format(entry), OUT / "tar" / f"{entry}.pdb")


def ubiquitin() -> None:
    fetch(RCSB.format("2K39"), OUT / "ubiquitin" / "2K39.pdb")


def reduce_1rkj(source: Path, target: Path, n_frames: int = FRAMES_1RKJ) -> None:
    """RNA atoms of ``n_frames`` frames spread evenly over each run (the frames that
    ``input.max_frames`` would select from the full trajectory)."""
    import MDAnalysis as mda

    from struct2clust.structures import select_frames

    for run in RUNS_1RKJ:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(str(source / run / "start.gro"), str(source / run / "md_500.xtc"))
            rna = u.select_atoms("nucleic")
            frames = select_frames(len(u.trajectory), 1, n_frames)
            out = target / run
            out.mkdir(parents=True, exist_ok=True)
            u.trajectory[frames[0]]
            rna.write(str(out / "rna.gro"))
            with mda.Writer(str(out / "rna.xtc"), rna.n_atoms) as writer:
                for _ in u.trajectory[frames]:
                    writer.write(rna)
        print(f"  {run}: {len(frames)} of {len(u.trajectory)} frames, {rna.n_atoms} RNA atoms")


def rkj() -> None:
    target = OUT / "1RKJ"
    if all((target / run / "rna.xtc").exists() for run in RUNS_1RKJ):
        print(f"  {target} exists, skipped")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = fetch(ZENODO_1RKJ, Path(tmp) / "1RKJ.zip")
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist()
                       if m.startswith(tuple(f"1RKJ/{r}/" for r in RUNS_1RKJ))]
            zf.extractall(tmp, members)
        reduce_1rkj(Path(tmp) / "1RKJ", target)


DATASETS = {"tar": tar, "ubiquitin": ubiquitin, "1rkj": rkj}


def main(names: list[str]) -> None:
    unknown = set(names) - set(DATASETS)
    if unknown:
        sys.exit(f"Unknown data set(s): {', '.join(sorted(unknown))}. "
                 f"Choose from: {', '.join(DATASETS)}")
    for name in names or DATASETS:
        print(f"{name}:")
        DATASETS[name]()
    print(f"Done. Data are in {OUT}")


if __name__ == "__main__":
    main([a.lower() for a in sys.argv[1:]])
