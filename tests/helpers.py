"""Shared test helpers: tiny synthetic PDB files."""


def atom_line(serial, name, resname, chain, resid, xyz, element, record="ATOM  ", altloc=" "):
    padded = f" {name:<3}" if len(name) < 4 else name
    return (
        f"{record}{serial:5d} {padded:4s}{altloc}{resname:>3s} {chain}{resid:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
    )


RNA_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "N9"]


def link_chain(xyz, per_residue, o3_index, p_index):
    """Copy of ``xyz`` with every P bonded (1.6 A) to the previous residue's O3'."""
    import numpy as np

    xyz = np.array(xyz, dtype=float)
    for r in range(1, len(xyz) // per_residue):
        xyz[r * per_residue + p_index] = xyz[(r - 1) * per_residue + o3_index] + [1.6, 0, 0]
    return xyz


def rna_pdb(path, xyz, seq="GAC", extra=True):
    """Write a small RNA chain; ``xyz`` has one row per heavy atom (len(seq) * 8).

    The P of each residue is placed 1.6 A from the O3' of the previous one (a
    phosphodiester bond), so the residues form one covalently linked chain.
    """
    xyz = link_chain(xyz, len(RNA_ATOMS), RNA_ATOMS.index("O3'"), RNA_ATOMS.index("P"))
    lines, k = [], 0
    for r, base in enumerate(seq, start=1):
        for name in RNA_ATOMS:
            lines.append(atom_line(k + 1, name, base, "A", r, xyz[k], name[0]))
            k += 1
        if extra:
            lines.append(atom_line(900 + r, "H5'", base, "A", r, (0, 0, 0), "H"))
    if extra:
        lines.append(atom_line(999, "O", "HOH", "W", 1, (5, 5, 5), "O", record="HETATM"))
    path.write_text("\n".join(lines) + "\nEND\n")
