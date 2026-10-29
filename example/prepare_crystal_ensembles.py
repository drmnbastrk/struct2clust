"""Build crystal-structure ensembles of one protein from the PDB.

    python example/prepare_crystal_ensembles.py endothiapepsin   (UniProt P11838)
    python example/prepare_crystal_ensembles.py eralpha          (UniProt P03372, LBD)

Needs Biopython (pip install biopython) and internet access. The PDB grows, so a
later run can find more entries; the entries used are listed in
data/<name>_meta.csv.

1. RCSB search: every X-ray entry that contains the protein.
2. Download the mmCIF files (cached in data/raw_mmcif/<name>/).
3. Every chain of the protein is renumbered to UniProt numbering with
   _struct_ref_seq and compared with the UniProt sequence (mutations recorded).
4. Chains with at least min_residues inside the domain are used; others are
   fragments or other domains. Chains with more than MAX_MISMATCH differences from
   the UniProt sequence are dropped: their numbering does not follow UniProt.
5. Core: residues whose backbone is present in at least COVERAGE of the chains
   (often-disordered loops drop out), plus must_contain (ERa helix 12, which
   carries the known agonist / antagonist difference). Positions mutated in at
   least 1% of the chains are left out of the core, so those mutants can be kept.
6. Kept: chains complete over the core (all backbone atoms) with the wild-type
   sequence there, written as data/<name>/<pdb>_<chain>.pdb (core residues, heavy
   atoms, first alternate location). data/<name>_meta.csv has one row per chain:
   kept or not, reason, resolution, space group, ligands and other polymers (e.g.
   coactivator peptides).
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import index_to_one, three_to_index

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent / "data"
PROTEINS = {
    # domain: chains with >= min_residues in this UniProt range are used (others are
    # fragments/other domains); must_contain: residues forced into the core
    "endothiapepsin": {"uniprot": "P11838", "domain": (1, 10_000), "min_residues": 250,
                       "must_contain": None},
    "eralpha": {"uniprot": "P03372", "domain": (300, 555), "min_residues": 150,
                "must_contain": (537, 545)},   # helix 12 (546-548 often disordered)
}
COVERAGE = 0.90
MAX_MISMATCH = 5
BACKBONE = ("N", "CA", "C", "O")


def search(acc: str) -> list[str]:
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers."
                         "reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": acc}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match",
            "value": "X-RAY DIFFRACTION"}}]},
        "return_type": "entry", "request_options": {"return_all_hits": True}}
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query",
                                 json.dumps(q).encode(), {"Content-Type": "application/json"})
    return sorted(h["identifier"] for h in json.load(urllib.request.urlopen(req))["result_set"])


def uniprot_seq(acc: str) -> str:
    txt = urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta").read().decode()
    return "".join(txt.splitlines()[1:])


def download(pdb: str, raw: Path):
    """Cached download; None if the file cannot be fetched after 5 attempts."""
    path = raw / f"{pdb}.cif.gz"
    if not path.exists():
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb}.cif.gz",
                                              timeout=60).read()
                path.write_bytes(data)
                break
            except Exception as e:
                if attempt == 4:
                    print(f"  {pdb}: download failed ({e})", flush=True)
                    return None
                time.sleep(2 * (attempt + 1))
    return path


def as_list(d, key):
    v = d.get(key, [])
    return v if isinstance(v, list) else [v]


def one(resname: str) -> str:
    try:
        return index_to_one(three_to_index(resname))
    except (KeyError, ValueError):
        return "X"


def chains_of(path: Path, acc: str, useq: str):
    """Yield (chain_id, {uniprot_pos: residue}, info) for every chain of the protein."""
    with gzip.open(path, "rt") as fh:
        d = MMCIF2Dict(fh)
    pdb = path.name[:4]
    ref_entity = dict(zip(as_list(d, "_struct_ref.id"), as_list(d, "_struct_ref.entity_id")))
    ref_acc = dict(zip(as_list(d, "_struct_ref.id"), as_list(d, "_struct_ref.pdbx_db_accession")))
    # entity -> list of (auth_beg, auth_end, db_beg) alignment segments for our protein
    segments: dict[str, list] = {}
    other_polymers = set()
    for ref_id, strand, a_beg, a_end, db_beg in zip(
            as_list(d, "_struct_ref_seq.ref_id"), as_list(d, "_struct_ref_seq.pdbx_strand_id"),
            as_list(d, "_struct_ref_seq.pdbx_auth_seq_align_beg"),
            as_list(d, "_struct_ref_seq.pdbx_auth_seq_align_end"),
            as_list(d, "_struct_ref_seq.db_align_beg")):
        if ref_acc.get(ref_id) == acc:
            segments.setdefault(strand, []).append((int(a_beg), int(a_end), int(db_beg)))
        else:
            other_polymers.add(ref_acc.get(ref_id, "?"))
    entity_desc = dict(zip(as_list(d, "_entity.id"), as_list(d, "_entity.pdbx_description")))
    entity_type = dict(zip(as_list(d, "_entity.id"), as_list(d, "_entity.type")))
    our_entities = {ref_entity[r] for r in ref_entity if ref_acc[r] == acc}
    other_desc = sorted({entity_desc[e] for e in entity_desc
                         if entity_type[e] == "polymer" and e not in our_entities})
    ligands = sorted(set(as_list(d, "_pdbx_entity_nonpoly.comp_id")) - {"HOH"})
    res = as_list(d, "_refine.ls_d_res_high") or as_list(d, "_reflns.d_resolution_high")
    info = {"pdb": pdb, "resolution": float(res[0]) if res and res[0] not in ("?", ".") else np.nan,
            "space_group": (as_list(d, "_symmetry.space_group_name_H-M") or ["?"])[0],
            "ligands": " ".join(ligands), "other_polymers": " | ".join(other_desc),
            "other_uniprot": " ".join(sorted(other_polymers - {"?"}))}
    with gzip.open(path, "rt") as fh:
        structure = MMCIFParser(QUIET=True).get_structure(pdb, fh)
    model = next(iter(structure))
    for chain in model:
        if chain.id not in segments:
            continue
        residues = {}
        for r in chain:
            het, num, icode = r.id
            if het != " " or icode != " ":
                continue
            for a_beg, a_end, db_beg in segments[chain.id]:
                if a_beg <= num <= a_end:
                    residues[db_beg + num - a_beg] = r
                    break
        mut = [f"{useq[p - 1]}{p}{one(r.get_resname())}" for p, r in residues.items()
               if 1 <= p <= len(useq) and one(r.get_resname()) != useq[p - 1]]
        yield chain.id, residues, {**info, "chain": chain.id, "mutations": " ".join(mut)}


def complete(r) -> bool:
    return all(a in r for a in BACKBONE)


def core_set(presence: pd.DataFrame, must_contain) -> list[int]:
    frac = presence.mean(axis=0)
    core = set(frac.index[frac >= COVERAGE])
    if must_contain:
        core |= set(range(must_contain[0], must_contain[1] + 1))
    return sorted(core)


def ranges(pos: list[int]) -> str:
    out, start = [], pos[0]
    for a, b in zip(pos[:-1], pos[1:]):
        if b != a + 1:
            out.append(f"{start}-{a}")
            start = b
    out.append(f"{start}-{pos[-1]}")
    return ",".join(out)


def write_pdb(residues: dict, core: list[int], path: Path) -> None:
    lines, serial = [], 1
    for pos in core:
        r = residues[pos]
        for atom in r:
            if atom.element == "H":
                continue
            if atom.is_disordered():
                atom = atom.selected_child
            name = atom.get_name()
            x, y, z = atom.coord
            lines.append(f"ATOM  {serial:5d} {name:<4s} {r.get_resname():>3s} A{pos:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{atom.bfactor:6.2f}          "
                         f"{atom.element:>2s}")
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def main(name: str) -> None:
    spec = PROTEINS[name]
    acc = spec["uniprot"]
    raw, out = HERE / "raw_mmcif" / name, HERE / name
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    entries = search(acc)
    useq = uniprot_seq(acc)
    print(f"{name}: {len(entries)} X-ray entries, UniProt length {len(useq)}", flush=True)
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(lambda p: download(p, raw), entries))
    failed = [e for e, p in zip(entries, paths) if p is None]
    paths = [p for p in paths if p is not None]
    print(f"downloaded {len(paths)}, failed {len(failed)} {failed}", flush=True)

    chains, rows = {}, []
    for path in paths:
        try:
            for cid, residues, info in chains_of(path, acc, useq):
                key = f"{info['pdb']}_{cid}"
                chains[key] = residues
                rows.append({"structure": key, **info})
        except Exception as e:   # unreadable entry
            rows.append({"structure": path.name[:4], "pdb": path.name[:4],
                         "reason": f"parse error: {e}"})
    meta = pd.DataFrame(rows).set_index("structure")
    meta["mutations"] = meta["mutations"].fillna("")
    d_lo, d_hi = spec["domain"]
    # > MAX_MISMATCH differences from UniProt: residue numbering does not follow UniProt
    shifted = {k for k in chains if len(meta.at[k, "mutations"].split()) > MAX_MISMATCH}
    domain = {k: r for k, r in chains.items()
              if sum(d_lo <= p <= d_hi for p in r) >= spec["min_residues"] and k not in shifted}
    positions = sorted({p for r in domain.values() for p in r if d_lo <= p <= d_hi})
    presence = pd.DataFrame(
        [[p in r and complete(r[p]) for p in positions] for r in domain.values()],
        index=list(domain), columns=positions)
    core = core_set(presence, spec["must_contain"])
    # positions mutated in >= 1% of domain chains (e.g. ERa Y537S, L536S, C381S/C417S/C530S)
    # are left out of the comparison, so that those mutants can be kept
    counts = pd.Series([int(m[1:-1]) for k in domain for m in meta.at[k, "mutations"].split()],
                       dtype=int).value_counts()
    frequent = sorted(counts.index[counts >= 0.01 * len(domain)])
    core = [p for p in core if p not in frequent]
    print(f"positions left out (mutated in >= 1% of chains): {frequent}", flush=True)
    core_text = ranges(core)
    print(f"{len(chains)} chains, {len(domain)} domain chains; "
          f"core {len(core)} residues: {core_text}", flush=True)

    meta["kept"] = False
    meta["mutations"] = meta["mutations"].fillna("")
    for key, residues in chains.items():
        if key in shifted:
            meta.at[key, "reason"] = (f"numbering mismatch (> {MAX_MISMATCH} differences "
                                      "from UniProt)")
            continue
        if key not in domain:
            meta.at[key, "reason"] = "fragment / other domain"
            continue
        missing = [p for p in core if p not in residues or not complete(residues[p])]
        mutated = [m for m in meta.at[key, "mutations"].split() if int(m[1:-1]) in set(core)]
        if missing:
            meta.at[key, "reason"] = f"{len(missing)} core residues missing/incomplete"
        elif mutated:
            meta.at[key, "reason"] = "mutation in core: " + " ".join(mutated)
        else:
            write_pdb(residues, core, out / f"{key}.pdb")
            meta.at[key, "kept"] = True
            meta.at[key, "reason"] = ""
    meta["core"] = core_text
    meta.to_csv(HERE / f"{name}_meta.csv")
    print(f"kept {int(meta['kept'].sum())} of {len(meta)} chains", flush=True)
    print(meta.loc[~meta["kept"], "reason"].fillna("?").str.split(":").str[0]
          .str.replace(r"^\d+ ", "", regex=True).value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
