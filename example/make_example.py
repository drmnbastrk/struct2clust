"""Write a small synthetic RNA ensemble with three known conformational states.

    python example/make_example.py
    struct2clust run example/data/synthetic -o example/results

The ensemble has 60 models of an 8-nucleotide chain: 20 per state, with
thermal-like noise. Struct2Clust should report three HIGH clusters.
"""
import sys
from pathlib import Path

import numpy as np

SEQ = "GACUGACU"
ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N9"]


def write_pdb(path: Path, xyz: np.ndarray) -> None:
    xyz = xyz.copy()
    per = len(ATOMS)
    for r in range(1, len(SEQ)):   # phosphodiester bond: P(r) 1.6 A from O3'(r-1)
        xyz[r * per + ATOMS.index("P")] = xyz[(r - 1) * per + ATOMS.index("O3'")] + [1.6, 0, 0]
    lines, k = [], 0
    for r, base in enumerate(SEQ, start=1):
        for name in ATOMS:
            x, y, z = xyz[k]
            lines.append(f"ATOM  {k + 1:5d}  {name:<3s} {base:>3s} A{r:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}")
            k += 1
    path.write_text("\n".join(lines) + "\nEND\n")


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    n_atoms = len(SEQ) * len(ATOMS)
    base = np.cumsum(rng.normal(size=(n_atoms, 3)) * 1.5, axis=0)
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for state in range(3):
        xyz = base.copy()
        xyz[n_atoms // 2:] += state * np.array([0.0, 6.0, 0.0])   # move the 3' half
        for _ in range(20):
            write_pdb(out / f"model_{k + 1}.pdb", xyz + rng.normal(scale=0.4, size=xyz.shape))
            k += 1
    print(f"Wrote {k} models to {out}")


if __name__ == "__main__":
    default = Path(__file__).resolve().parent / "data" / "synthetic"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
