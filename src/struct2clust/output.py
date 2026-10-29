"""Writing a run to disk: tables, medoid structures, viewer scripts, summary, report."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from struct2clust import __version__
from struct2clust.biology import main_residues, residue_table
from struct2clust.clustering import medoids
from struct2clust.metrics import superpose
from struct2clust.pipeline import (
    CONCLUSIONS,
    conclusion_if_sampling_sufficient,
    run_conclusion,
    sampling_notes,
    superposition,
)
from struct2clust.structures import write_models_pdb, write_pdb

logger = logging.getLogger(__name__)


def _cluster_name(c: int) -> str:
    return f"C{c + 1}"


# ---------------------------------------------------------------------------
# Cluster numbers
# ---------------------------------------------------------------------------
# Clusters are reported by one number each (1, 2, ...). Refinement works on a
# hierarchy of nodes ("2" a top-level cluster, "2.1" its first sub-cluster); every
# node is reported as the final clusters it contains, and every cluster carries the
# number of the top-level cluster it comes from.

def clusters_under(run):
    """Function: hierarchy node -> sorted final cluster numbers (1-based) inside it."""
    tab = run.stability.clusters if run.stability is not None else None
    paths = ({int(c): str(p) for c, p in zip(tab["cluster"], tab["path"])}
             if tab is not None and len(tab) else {})

    def under(node: str) -> list[int]:
        return sorted(c for c, p in paths.items() if p == node or p.startswith(node + "."))
    return under


def clusters_text(numbers: list[int]) -> str:
    """"cluster 2", "clusters 2-4" or "clusters 2, 5"."""
    if len(numbers) == 1:
        return f"cluster {numbers[0]}"
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"clusters {numbers[0]}-{numbers[-1]}"
    return "clusters " + ", ".join(map(str, numbers))


def substructure_numbers(run) -> Optional[np.ndarray]:
    """Sub-group of every structure within its cluster (1, 2, ...; 0 = none), for clusters
    made of reproducible sub-groups closer than the state scale. None if there are none."""
    if run.substructure is None:
        return None
    out = np.zeros(len(run.labels), dtype=int)
    for c in range(run.n_clusters):
        members = run.labels == c
        labels = sorted(set(run.substructure[members]) - {""})
        if len(labels) > 1:
            for j, label in enumerate(labels):
                out[members & (run.substructure == label)] = j + 1
    return out if out.any() else None


def top_level_numbers(run) -> np.ndarray:
    """Top-level cluster of every structure (1, 2, ...; 0 = top-level noise)."""
    groups = run.groups if run.groups is not None else run.labels
    return np.where(groups >= 0, groups + 1, 0)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def clusters_table(run) -> pd.DataFrame:
    """One row per structure used for clustering."""
    ens, labels, dist = run.ensemble, run.labels, run.distances
    k = run.n_clusters
    medoid_of = {m: c for c, m in run.medoids.items()}
    member = run.stability.membership if run.stability is not None else np.zeros((len(ens), 0))
    top = top_level_numbers(run)
    rows = []
    for i, name in enumerate(ens.names):
        c = int(labels[i])
        if k and member.shape[1]:
            closest = c if c >= 0 else int(np.argmax(member[i]))
            prob = float(member[i, closest])
            if c < 0 and prob == 0:
                closest = -1
        else:
            closest, prob = -1, float("nan")
        rows.append({
            "structure": name,
            "file": ens.sources[i].as_posix() if ens.sources[i] else "",
            "cluster": c + 1,                      # 0 = noise
            "top_level": int(top[i]),              # 0 = top-level noise
            "membership": round(prob, 3),
            "closest_cluster": closest + 1,
            "is_medoid": medoid_of.get(i) is not None,
            "distance_to_medoid": (round(float(dist[i, run.medoids[c]]), 4) if c >= 0 else np.nan),
        })
    table = pd.DataFrame(rows)
    sub = substructure_numbers(run)
    if sub is not None:
        # reproducible sub-groups closer than the state scale (METHODS.md, section 6)
        table.insert(table.columns.get_loc("top_level") + 1, "substructure", sub)
    if ens.replica:
        table.insert(1, "replica", ens.replica)
        table.insert(2, "frame", ens.frame)
        table = table.drop(columns="file")
    if run.biology is not None:
        table["radius_of_gyration"] = run.biology.rg["rg"].round(3).to_numpy()
    return table


def cluster_summary(run) -> pd.DataFrame:
    labels, dist = run.labels, run.distances
    st = run.stability.clusters.set_index("cluster")
    rows = []
    for c in range(run.n_clusters):
        idx = np.flatnonzero(labels == c)
        intra = dist[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        m = run.medoids[c]
        s = st.loc[c + 1]
        row = {
            "cluster": c + 1,
            "top_level": int(str(s.get("path", c + 1)).split(".")[0]),
            "size": len(idx),
            "fraction": round(len(idx) / len(labels), 4),
            **({"fraction_low": round(run.populations["fraction_low"][c], 4),    # populations.LEVEL
                "fraction_high": round(run.populations["fraction_high"][c], 4)}
               if run.populations else {}),
            # the resolution selection's decision and its own estimate ...
            "selection_decision": s.get("selection_decision", ""),
            "selection_estimate": _round(s.get("selection_estimate"), 3),
            # ... and the independent estimate from fresh subsamples, with its tier
            "independent_stability_estimate": round(s["jaccard_mean"], 3),
            "independent_stability_low": round(s["jaccard_q025"], 3),    # 2.5% of subsamples
            "independent_stability_high": round(s["jaccard_q975"], 3),  # 97.5%
            "stability_tier": s["tier"],
            "dissolved_fraction": round(s["dissolved_fraction"], 3),
            "medoid": run.ensemble.names[m],
            "median_intra_distance": round(float(np.median(intra)), 4) if len(intra) else 0.0,
        }
        if run.membership_tau:
            taus = [t[c + 1] for t in run.membership_tau.values() if c + 1 in t]
            row["membership_autocorrelation_time"] = max(taus) if taus else float("nan")
        if run.biology is not None:
            rg = run.biology.rg.loc[run.biology.rg["cluster"] == c + 1, "rg"]
            rmsf = run.biology.rmsf.loc[run.biology.rmsf["cluster"] == c + 1, "rmsf"]
            row.update({"rg_mean": round(rg.mean(), 3), "rg_sd": round(rg.std(ddof=1), 3),
                        "rmsf_mean": round(rmsf.mean(), 3)})
        rows.append(row)
    table = pd.DataFrame(rows)
    for label, frac in replica_populations(run).items():
        table[f"population_{label}"] = frac
    return table


def _round(value, digits):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return float("nan")


def replica_populations(run) -> dict[str, list[float]]:
    """Trajectory mode: fraction of each replica's frames in each cluster."""
    ens = run.ensemble
    if not ens.replica:
        return {}
    replica = np.asarray(ens.replica)
    out = {}
    for label in ens.replicas:
        lab = run.labels[replica == label]
        out[label] = [round(float(np.mean(lab == c)), 4) for c in range(run.n_clusters)]
    return out


def _write_biology(run, out: Path) -> None:
    bio = run.biology
    out.mkdir(exist_ok=True)
    res = bio.residues[["residue", "chain", "resid", "resname"]]
    bio.rmsf.merge(res, on="residue").round(4).to_csv(out / "rmsf.csv", index=False)
    rows = []
    for c, freq in bio.contacts.items():
        i, j = np.nonzero(np.triu(freq) > 0)
        rows.append(pd.DataFrame({"cluster": c, "residue_i": i + 1, "residue_j": j + 1,
                                  "frequency": freq[i, j].round(4)}))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "contacts.csv", index=False)
    if not bio.torsions.empty:
        bio.torsions.merge(res, on="residue").round(3).to_csv(out / "torsions.csv", index=False)
    if not bio.torsion_spread.empty:
        bio.torsion_spread.merge(res, on="residue").round(4).to_csv(
            out / "torsion_separation.csv", index=False)


# ---------------------------------------------------------------------------
# Medoid structures and viewer scripts
# ---------------------------------------------------------------------------

def write_medoids(run, out: Path, colors: list[str]) -> list[Path]:
    """Medoids superposed on the C1 medoid as for the clustering distance (same atoms;
    least squares for RMSD, TM-score superposition for TM), as PDB files."""
    out.mkdir(exist_ok=True)
    ens = run.ensemble
    fit = superposition(run.config, ens)
    mask = fit.mask
    ref = ens.coords[run.medoids[0], mask]
    paths = []
    for c in range(run.n_clusters):
        xyz = ens.coords[run.medoids[c]]
        r, t = superpose(xyz[mask], ref, fit.metric, fit.d0)
        p = out / f"{_cluster_name(c)}.pdb"
        write_pdb(p, ens.atoms, xyz @ r.T + t)
        paths.append(p)

    names = [p.stem for p in paths]
    pml = ["# PyMOL: pymol view_medoids.pml  (structures are pre-superposed)"]
    tcl = ["# VMD: vmd -e view_medoids.tcl  (structures are pre-superposed)"]
    for i, name in enumerate(names):
        hexc = colors[i % len(colors)].lstrip("#")
        pml += [f"load {name}.pdb", f"set_color col_{name}, [0x{hexc[:2]}, 0x{hexc[2:4]}, "
                f"0x{hexc[4:]}]", f"color col_{name}, {name}"]
        rgb = " ".join(f"{int(hexc[j:j + 2], 16) / 255:.3f}" for j in (0, 2, 4))
        tcl += [f"mol new {name}.pdb", f"color change rgb {i + 17} {rgb}",
                "mol delrep 0 top", "mol representation NewCartoon",
                f"mol color ColorID {i + 17}", "mol addrep top", f"mol rename top {name}"]
    pml += ["hide everything", "show cartoon", "set cartoon_ring_mode, 3", "orient"]
    (out / "view_medoids.pml").write_text("\n".join(pml) + "\n")
    (out / "view_medoids.tcl").write_text("\n".join(tcl) + "\n")
    return paths


OPACITY_MAX, OPACITY_MIN, OPACITY_LEVELS = 1.0, 0.1, 10


def member_order(dist: np.ndarray, labels: np.ndarray, medoid: int, cluster: int) -> np.ndarray:
    """Members of a cluster from the best representative (the medoid) to the worst
    (largest distance to the medoid)."""
    idx = np.flatnonzero(labels == cluster)
    return idx[np.lexsort((idx, dist[idx, medoid]))]


def member_opacity(n: int) -> np.ndarray:
    """Opacity by rank: 1 for the medoid, falling linearly to OPACITY_MIN for the last
    member, rounded to OPACITY_LEVELS steps (each step is one material in VMD)."""
    if n == 1:
        return np.array([OPACITY_MAX])
    raw = OPACITY_MAX - (OPACITY_MAX - OPACITY_MIN) * np.arange(n) / (n - 1)
    step = (OPACITY_MAX - OPACITY_MIN) / (OPACITY_LEVELS - 1)
    return np.round(OPACITY_MIN + np.round((raw - OPACITY_MIN) / step) * step, 2)


def write_cluster_views(run, out: Path, colors: list[str]) -> list[Path]:
    """Per cluster: all members in one multi-model PDB plus VMD and PyMOL scripts.

    Members are superposed on their cluster medoid as for the clustering distance
    (same atoms; least squares for RMSD, TM-score superposition for TM), which is
    itself superposed on the C1 medoid, so every cluster is in the same
    frame and several can be loaded together. Models are ordered from the medoid
    to the member farthest from it; the scripts draw every member as a cartoon in
    the cluster colour, with opacity falling from 1 (medoid) to 0.1 (farthest).
    """
    out.mkdir(exist_ok=True)
    ens, dist = run.ensemble, run.distances
    fit = superposition(run.config, ens)
    mask = fit.mask
    ref = ens.coords[run.medoids[0], mask]
    cap = run.config.figures.cluster_view_max_members
    rows, paths = [], []
    for c in range(run.n_clusters):
        name = _cluster_name(c)
        m = run.medoids[c]
        r, t = superpose(ens.coords[m, mask], ref, fit.metric, fit.d0)
        target = ens.coords[m, mask] @ r.T + t
        order = member_order(dist, run.labels, m, c)
        if cap is not None:
            order = order[:cap]
        opacity = member_opacity(len(order))
        models = []
        for k, i in enumerate(order):
            ri, ti = superpose(ens.coords[i, mask], target, fit.metric, fit.d0)
            models.append(ens.coords[i] @ ri.T + ti)
            rows.append({"cluster": c + 1, "model": k + 1, "structure": ens.names[i],
                         "distance_to_medoid": round(float(dist[i, m]), 4),
                         "opacity": float(opacity[k])})
        pdb = out / f"{name}_members.pdb"
        write_models_pdb(pdb, ens.atoms, models, [ens.names[i] for i in order])
        hexc = colors[c % len(colors)].lstrip("#")
        rgb = [int(hexc[j:j + 2], 16) / 255 for j in (0, 2, 4)]
        (out / f"{name}.tcl").write_text(_vmd_cluster_script(name, pdb.name, rgb, opacity, c))
        (out / f"{name}.py").write_text(_pymol_cluster_script(name, pdb.name, rgb, opacity))
        paths.append(pdb)
    pd.DataFrame(rows).to_csv(out / "members.csv", index=False)
    return paths


def _vmd_cluster_script(name: str, pdb: str, rgb: list[float], opacity: np.ndarray,
                        c: int) -> str:
    color_id = 17 + c % 16
    lines = [
        f"# VMD: vmd -e {name}.tcl   (run from this folder; several clusters can be loaded",
        "# together, they share one frame). Every member of the cluster is drawn as",
        "# NewCartoon in the cluster colour, from the medoid (opaque) to the member",
        "# farthest from it (most transparent); members.csv lists the order.",
        f"mol new {pdb} type pdb waitfor all",
        "set m [molinfo top]",
        f"mol rename $m {name}",
        "mol delrep 0 $m",
        f"color change rgb {color_id} {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}",
    ]
    for level in sorted(set(opacity.tolist())):
        mat = f"{name}_op{int(round(level * 100)):03d}"
        lines += [f"catch {{material add {mat} copy Transparent}}",
                  f"material change opacity {mat} {level:.2f}"]
    for frame, level in enumerate(opacity):
        mat = f"{name}_op{int(round(level * 100)):03d}"
        lines += ["mol representation NewCartoon", f"mol color ColorID {color_id}",
                  f"mol material {mat}", "mol selection all", "mol addrep $m",
                  f"mol drawframes $m {frame} {frame}"]
    lines.append("animate goto 0")
    return "\n".join(lines) + "\n"


def _pymol_cluster_script(name: str, pdb: str, rgb: list[float], opacity: np.ndarray) -> str:
    return f'''# PyMOL: pymol {name}.py   (or, inside PyMOL: run {name}.py)
# Every member of the cluster as a cartoon in the cluster colour, from the medoid
# (opaque) to the member farthest from it (most transparent); members.csv lists
# the order. Several cluster scripts can be run in one session (shared frame).
import os
from pymol import cmd

# the structure file is next to this script; PyMOL does not always set __file__
# to the script, so the working directory is tried as well
candidates = [os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "."))),
                           "{pdb}"), os.path.abspath("{pdb}")]
path = next((p for p in candidates if os.path.exists(p)), None)
if path is None:
    raise FileNotFoundError("{pdb} not found: run this script inside its cluster_views folder")
opacity = {[float(o) for o in opacity]}
cmd.load(path, "{name}_all")
cmd.set_color("{name}_colour", {[round(v, 3) for v in rgb]})
for state, value in enumerate(opacity, start=1):
    obj = "{name}_%04d" % state
    cmd.create(obj, "{name}_all", state, 1)
    cmd.set("cartoon_transparency", round(1 - value, 2), obj)
cmd.delete("{name}_all")
cmd.group("{name}", "{name}_*")
cmd.hide("everything", "{name}")
cmd.show("cartoon", "{name}")
cmd.color("{name}_colour", "{name}")
cmd.set("cartoon_ring_mode", 3)
cmd.orient("{name}")
'''


# ---------------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------------

def source_sha256() -> str:
    """SHA-256 of the package's Python files (names and contents, in name order): it
    identifies the code that wrote a result even when the version number is unchanged."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _versions() -> dict:
    import importlib.metadata as md
    out = {"python": platform.python_version(), "struct2clust": __version__,
           "struct2clust_source_sha256": source_sha256()}
    for pkg in ("numpy", "scipy", "scikit-learn", "umap-learn", "MDAnalysis", "diptest"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def residue_ranges(positions: list[int], residues: pd.DataFrame) -> str:
    """Residue positions (1-based) as ranges of the original numbering, e.g. "8, 72-76";
    with several chains each range carries its chain, e.g. "A:72-76"."""
    info = residues.set_index("residue")
    several = info["chain"].nunique() > 1
    out, run = [], []
    for p in sorted(positions):
        if run and (p != run[-1] + 1 or info.at[p, "chain"] != info.at[run[-1], "chain"]):
            out.append(run)
            run = []
        run.append(p)
    if run:
        out.append(run)
    text = []
    for r in out:
        first, last = info.loc[r[0]], info.loc[r[-1]]
        span = (f"{first['resid']}" if len(r) == 1 else f"{first['resid']}-{last['resid']}")
        text.append(f"{first['chain']}:{span}" if several else span)
    return ", ".join(text)


def difference_summary(run) -> Optional[dict]:
    """Where the cluster medoids differ: the fewest residues that carry half of the
    squared deviation, averaged over the pairs of clusters, and for every pair."""
    md = run.medoid_difference
    if md is None or md.empty:
        return None
    residues = residue_table(run.ensemble.atoms)
    mean = md.groupby("residue")["share"].mean()
    main, share = main_residues(mean)
    pairs = []
    for (a, b), grp in md.groupby(["cluster_a", "cluster_b"]):
        p_main, p_share = main_residues(grp.set_index("residue")["share"])
        pairs.append({"a": int(a), "b": int(b), "residues": residue_ranges(p_main, residues),
                      "n_residues": len(p_main), "share": round(p_share, 3)})
    return {"residues": residue_ranges(main, residues), "n_residues": len(main),
            "n_residues_total": int(md["residue"].nunique()), "share": round(share, 3),
            "pairs": pairs}


# The report notes when at most this share of the residues carries half of the squared
# deviation between medoids (a reporting choice): the clusters then differ locally.
LOCAL_DIFFERENCE = 0.10

# Typical coordinate error of experimental structures, in Å (Cruickshank 1999). Clusters
# whose medoids are closer than this are flagged in the report; the decision is unchanged.
COORDINATE_ERROR = 0.5


def _ratio_text(p: dict, short: bool = False) -> str:
    """"0.80x the state scale (0.62-1.05 between subsamples)" or "" without a scale."""
    if not p.get("ratio_to_state_scale"):
        return ""
    text = f"{p['ratio_to_state_scale']:.2f}x" + ("" if short else " the state scale")
    spread = p.get("ratio_subsampling_range")
    if spread:
        text += f", {spread[0]:.2f}-{spread[1]:.2f} between subsamples"
    return text


def separation_notes(run) -> dict:
    """Facts the effect-size rule cannot judge by itself (METHODS.md, section 6).

    - ``no_reference``: with only two top-level clusters there is no other cluster to
      define the state scale, so the pair is not tested; its medoid distance and the
      clusters' own spread are given instead.
    - ``below_coordinate_error`` (RMSD only): pairs of final clusters whose medoids are
      closer than the typical coordinate error of experimental structures.
    """
    dist, notes = run.distances, {"no_reference": None, "below_coordinate_error": []}
    ratio = run.config.selection.refine_min_separation_ratio
    top = run.groups if run.groups is not None else run.labels
    if ratio and top.max() + 1 == 2:
        m = medoids(dist, top)
        under = clusters_under(run)
        notes["no_reference"] = {
            "medoid_distance": round(float(dist[m[0], m[1]]), 4),
            "top_level_clusters": [
                {"top_level": c + 1, "clusters": under(str(c + 1)),
                 "median_distance_to_medoid":
                     round(float(np.median(dist[np.flatnonzero(top == c), m[c]])), 4)}
                for c in (0, 1)],
        }
    if run.config.metric.name == "rmsd" and run.labels.max() >= 1:
        m = medoids(dist, run.labels)
        for a in m:
            for b in m:
                d = float(dist[m[a], m[b]])
                if a < b and d < COORDINATE_ERROR:
                    notes["below_coordinate_error"].append(
                        {"a": a + 1, "b": b + 1, "medoid_distance": round(d, 4)})
    return notes


def public_merges(run) -> list[dict]:
    """Top-level merges by the effect-size rule, reported by final cluster numbers."""
    under = clusters_under(run)
    return [{"clusters": under(m["into"]), "n_merged": len(m["merged_top_level_clusters"]),
             "medoid_distances": [{k: p[k] for k in ("distance", "state_scale",
                                                     "ratio_to_state_scale",
                                                     "ratio_subsampling_range")}
                                  for p in m["pairs"]]}
            for m in run.top_level_merges]


def public_attempts(run) -> list[dict]:
    """Refinement attempts, each node reported by its top-level cluster, its depth and the
    final clusters inside it."""
    under = clusters_under(run)
    out = []
    for e in run.refinement:
        e = dict(e)
        node = e.pop("cluster")
        entry = {"top_level": int(node.split(".")[0]), "level": node.count("."),
                 "clusters": under(node), **e}
        if "merged_subclusters" in e:
            entry["merged_subclusters"] = [{"clusters": under(m["into"]), "candidates": m["parts"]}
                                           for m in e["merged_subclusters"]]
        out.append(entry)
    return out


def run_summary(run, extra: dict) -> dict:
    cfg = run.config
    ca = run.clusterability
    sel = run.selection
    found = sel.found_clusters
    picked = sel.candidates.loc[sel.candidates["selected"], "min_jaccard_se"]
    return {
        "struct2clust_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "input": {
            "type": cfg.input.type,
            "n_structures": run.n_input,
            "n_duplicates_removed": len(run.duplicates),
            "duplicates": [{"kept": a, "removed": b, "distance": round(d, 6)}
                           for a, b, d in run.duplicates],
            "n_used": len(run.ensemble),
            "replicas": run.sampling or [
                {"name": r, "n_frames": int(sum(x == r for x in run.ensemble.replica))}
                for r in run.ensemble.replicas
            ],
            "molecule_type": run.ensemble.molecule_type,
            "n_residues": run.ensemble.n_residues,
            "resampling_groups": None if not cfg.grouped else {
                "file": str(cfg.input.groups), "n_groups": run.n_groups,
                "group_fraction": cfg.input.group_fraction,
                "absent_cluster": "Jaccard 0 (not reproduced from the groups drawn)"},
        },
        "distance": {"metric": cfg.metric.name, "label": run.metric_label,
                     "unit": run.distance_unit, "atoms": cfg.metric.atoms,
                     "equivalent_atoms_matched": cfg.metric.name == "rmsd"
                     and cfg.metric.equivalent_atoms,
                     "compared_atoms": run.compared_atoms},
        "clusterability": None if ca is None else {
            "verdict": ca.verdict, "dip_statistic": round(ca.dip_statistic, 5),
            "dip_pvalue": ca.dip_pvalue, "spectral_k": ca.spectral_k,
        },
        "selection": {
            "rule": "finest partition whose clusters all have mean Jaccard >= "
                    f"{cfg.stability.high}",
            "reproducible_partition_found": sel.stable,
            "min_cluster_size": sel.min_cluster_size if found else None,
            "min_samples": sel.min_samples if found else None,
            "cluster_selection_method": sel.method or None,
            "weakest_cluster_jaccard": round(sel.min_jaccard, 4) if found else None,
            "weakest_cluster_jaccard_se": round(float(picked.iloc[0]), 4) if found else None,
            "n_candidates": len(sel.candidates),
            "resampling_risk": cfg.selection.resampling_risk,
            "borderline": sel.borderline,
            "n_borderline_candidates": int(sel.candidates["borderline"].sum()),
            "tie_partitions_differ": sel.tie_partitions_differ,
        },
        "effect_size": {
            "rule": "clusters / sub-clusters whose medoids are closer than "
                    f"{cfg.selection.refine_min_separation_ratio} x the state scale "
                    "(median member-to-medoid distance of the clusters not compared) "
                    + (f"or than {cfg.selection.refine_min_separation} " if
                       cfg.selection.refine_min_separation else "")
                    + "are one state"
            if cfg.selection.refine_min_separation_ratio or cfg.selection.refine_min_separation
            else "off: no clusters are merged; the effect size of every pair is reported",
            "n_selected_clusters": sel.n_clusters,
            "top_level_merges": public_merges(run),
            # every pair of final clusters (separation.csv); reported, not a decision
            "pairs": run.separation,
            **separation_notes(run),
        },
        "medoid_difference": difference_summary(run),
        "refinement": {
            "enabled": cfg.selection.refine,
            "n_top_level_clusters": int(run.groups.max() + 1) if run.groups is not None
            else sel.n_clusters,
            "attempts": public_attempts(run),
        },
        "result": {
            "conclusion": run_conclusion(run)[0],
            # diagnostic: the same rule without the final number of clusters
            "conclusion_before_effect_size": run_conclusion(run)[1],
            "n_clusters": run.n_clusters,
            "noise_fraction": round(float(np.mean(run.labels < 0)), 4),
            "top_level_noise_fraction": round(float(np.mean(run.groups < 0)), 4)
            if run.groups is not None else None,
            "stability_tiers": run.stability.clusters["tier"].value_counts().to_dict()
            if run.stability is not None else {},
            "populations": population_summary(run),
            # trajectories: can the blocks be treated as independent? (METHODS.md, section 9)
            "sampling_check": run.sampling_check,
            "conclusion_if_sampling_sufficient": conclusion_if_sampling_sufficient(run),
            # input.membership_blocks: the pass with geometric blocks that was replaced
            "first_pass": run.first_pass,
        },
        "figures": {"umap_trustworthiness": extra.get("umap_trustworthiness")},
        "config": cfg.model_dump(mode="json"),
        "software": _versions(),
        "seed": cfg.seed,
    }


def _input_lines(run, s: dict) -> list[str]:
    lines = [
        f"- Structures: {s['input']['n_structures']} read, "
        f"{s['input']['n_duplicates_removed']} near-duplicates removed"
        + (" (`duplicates.csv`)" if s["input"]["n_duplicates_removed"] else "")
        + f", {s['input']['n_used']} clustered ({s['input']['molecule_type']}, "
        f"{s['input']['n_residues']} residues).",
        f"- Distance: {run.metric_label}"
        + (f" ({run.distance_unit})" if run.distance_unit else "") + ".",
    ]
    for t in run.sampling:
        lines.append(
            f"- Trajectory {t['name']}: {t['n_frames']} frames, autocorrelation time "
            f"{t['autocorrelation_time']:.0f} frames, about {t['effective_frames']:.0f} "
            "independent frames (METHODS.md, section 9)"
            + "".join(f"; {note}" for note in sampling_notes(t)) + ".")
    ca = s["clusterability"]
    if ca:
        p = ca["dip_pvalue"]
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        lines.append(f"- Clusterability: **{ca['verdict']}** (dip test {p_txt}; "
                     f"spectral eigengap k = {ca['spectral_k']}).")
    cfg = run.config
    unit = ("groups (`input.groups`)" if cfg.grouped else
            "trajectory blocks" if cfg.by_unit else "structures")
    lines.append(
        f"- Resampling conventions: every subsample holds {100 * cfg.selection_fraction:.0f}% "
        f"of the {unit}; a cluster counts as reproducible at a mean Jaccard of at least "
        f"{cfg.stability.high:g}. Both are conventions, and a result near the threshold can "
        "change with them (METHODS.md, section 13, \"Resampling conventions\").")
    return lines


def _separation_lines(pairs: list[dict], unit: str) -> list[str]:
    """Descriptive effect size of the final clusters (METHODS.md, section 6). It states
    how far apart the clusters are compared with the width of the ensemble's states; it
    does not change the clusters or the conclusion."""
    if not pairs:
        return []
    if all(p["ratio_to_state_scale"] is None for p in pairs):
        p = pairs[0]
        return [f"- Separation: the medoids of clusters {p['a']} and {p['b']} are "
                f"{p['medoid_distance']:.2f}{unit} apart; their members lie a median "
                f"{p['scatter_a']:.2f}{unit} and {p['scatter_b']:.2f}{unit} from their own "
                "medoid. With two clusters there is no other cluster to define the state "
                "scale (`separation.csv`)."]
    scaled = [p for p in pairs if p["ratio_to_state_scale"] is not None]
    close = min(scaled, key=lambda p: p["ratio_to_state_scale"])
    text = (f"- Separation (descriptive, `separation.csv`): the closest clusters are "
            f"{close['a']} and {close['b']}, medoids {close['medoid_distance']:.2f}{unit} "
            f"apart ({_ratio_text(close)}).")
    below = [p for p in scaled if p["ratio_to_state_scale"] < 1]
    if below:
        text += (f" {len(below)} of {len(scaled)} pairs have medoids closer than the "
                 "ensemble's states are wide (ratio < 1): "
                 + ", ".join(f"{p['a']} and {p['b']} ({p['ratio_to_state_scale']:.2f})"
                             for p in below)
                 + ". The clusters are kept; the ratio describes how distinct they are.")
    return [text]


def empty_report(run, summary: dict) -> str:
    """Report of a run in which no candidate setting gave two or more clusters."""
    s = summary
    lines = [f"# Struct2Clust report: {run.out_dir.name}", "",
             f"**{CONCLUSIONS[s['result']['conclusion']]}**", ""]
    lines += _input_lines(run, s)
    lines.append(
        f"- Resolution: none of the {s['selection']['n_candidates']} candidate HDBSCAN "
        "settings (`selection.csv`) gives two or more clusters. The structures form one "
        "dense group without reproducible sub-groups, or no dense groups at all; the "
        "clusterability verdict above and the distance matrix (`distance_matrix.npy`) tell "
        "the two apart.")
    return "\n".join(lines) + "\n"


def report(run, summary: dict, cluster_table: pd.DataFrame) -> str:
    s = summary
    unit = f" {run.distance_unit}" if run.distance_unit else ""
    lines = [
        f"# Struct2Clust report: {run.out_dir.name}",
        "",
    ] + _input_lines(run, s)
    sel = s["selection"]
    found = "yes" if sel["reproducible_partition_found"] else "**no**"
    ref = s["refinement"]
    lines[2:2] = [f"**{CONCLUSIONS[s['result']['conclusion']]}**", ""]
    check = s["result"].get("sampling_check")
    if check and not check["sufficient"]:
        lines[4:4] = [
            "Sampling check failed: " + "; ".join(check["reasons"]) + ". With independent "
            f"blocks the resampling would give `{s['result']['conclusion_if_sampling_sufficient']}`"
            "; this cannot be assessed from the data, so the conclusion is "
            "`insufficient_sampling`.", ""]
    first = s["result"].get("first_pass")
    if first:
        blocks = ", ".join(f"{t['name']} {t['n_blocks']}" for t in run.sampling)
        lines[4:4] = [
            "Second pass (input.membership_blocks): with blocks of about one geometric "
            f"autocorrelation time the sampling check failed because cluster memberships "
            f"stay correlated for longer than a block (that pass: `{first['conclusion']}`, "
            f"{first['n_clusters']} clusters). The resampling was repeated with blocks of "
            f"about one membership autocorrelation time ({blocks} blocks); this report "
            "shows that pass.", ""]
    before = s["result"].get("conclusion_before_effect_size")
    if before and before != s["result"]["conclusion"]:
        lines[4:4] = [
            f"Before the effect-size rule, the clusters of the resolution selection gave "
            f"`{before}`. This is a diagnostic of an intermediate step, not the conclusion of "
            "the run: the final partition (below) has fewer than two clusters.", ""]
    floor = run.config.selection.refine_min_separation
    scale_text = ("the state scale" if not floor else
                  f"the separation threshold (the state scale or {floor:g}{unit}, "
                  "whichever is larger)")
    lines.append(
        f"- Resolution: {ref['n_top_level_clusters']} top-level clusters "
        f"(min_cluster_size {sel['min_cluster_size']}, min_samples {sel['min_samples']}; "
        f"weakest cluster Jaccard {sel['weakest_cluster_jaccard']:.2f}). "
        f"All clusters reproducible: {found}."
    )
    for m in s["effect_size"]["top_level_merges"]:
        pairs = ", ".join(f"{p['distance']:.2f}{unit}"
                          + (f" ({_ratio_text(p)})" if _ratio_text(p) else "")
                          for p in m["medoid_distances"])
        lines.append(
            f"- {clusters_text(m['clusters']).capitalize()}: {m['n_merged']} clusters of the "
            f"resolution selection are closer than {scale_text} (medoids {pairs}) and are "
            "reported as one state; its sub-groups are numbered in `clusters.csv`, column "
            "`substructure`.")
    lines += _separation_lines(s["effect_size"].get("pairs") or [], unit)
    nr = s["effect_size"]["no_reference"]
    if nr:
        t1, t2 = nr["top_level_clusters"]
        lines.append(
            f"- Two top-level clusters ({clusters_text(t1['clusters'])} and "
            f"{clusters_text(t2['clusters'])}): their medoids are "
            f"{nr['medoid_distance']:.2f}{unit} apart, and their members lie a median "
            f"{t1['median_distance_to_medoid']:.2f}{unit} and "
            f"{t2['median_distance_to_medoid']:.2f}{unit} from their own medoid. With two "
            "clusters there is no other cluster to define the state scale, so the effect-size "
            "rule does not apply to this pair; judge from these distances whether they are "
            "distinct states.")
    if sel["borderline"]:
        lines.append(
            "- **At the stability threshold:** a candidate with more clusters has a "
            "weakest-cluster Jaccard within Monte Carlo error of the threshold after the "
            "maximum number of subsamples. Its stability was not established, so it was not "
            "selected (see `selection.csv`, column `borderline`).")
    if sel.get("tie_partitions_differ"):
        lines.append(
            "- **Tied settings:** several qualifying settings have the selected number of "
            "clusters and noise fraction but give different partitions; the one with the "
            "highest estimated mean Jaccard was selected. This choice can change with the "
            "random seed (see `selection.csv`).")
    splits = [e for e in ref["attempts"] if e["split"]]
    for e in ref["attempts"]:
        if e["split"] and e["level"] == 0:
            lines.append(
                f"- Refinement: {clusters_text(e['clusters'])} are sub-states of one top-level "
                f"cluster (top level {e['top_level']}), separated by refinement; their stability "
                "is measured within that cluster.")
    within = [e for e in ref["attempts"] if e.get("substructure")]
    if not splits and ref["attempts"]:
        lines.append("- Refinement: no cluster contains distinct, reproducible sub-states.")
    for e in within:
        pairs = ", ".join(
            f"{p['a']} and {p['b']} {p['distance']:.2f}{unit}"
            + (f" ({_ratio_text(p, short=True)})" if _ratio_text(p) else "")
            for p in e["medoid_separation"])
        limit = (f"the state scale ({e['state_scale']:.2f}{unit})"
                 if e["state_scale"] and not floor
                 else f"the separation threshold ({e['separation_threshold']:.2f}{unit})")
        lines.append(
            f"- {clusters_text(e['clusters']).capitalize()} contains "
            f"{len(e['subcluster_sizes'])} reproducible sub-groups (sizes "
            f"{', '.join(map(str, e['subcluster_sizes']))}) whose medoids are closer than "
            f"{limit}: sub-groups {pairs}. They are reported as one state; the sub-groups are "
            "numbered in `clusters.csv`, column `substructure`.")
    for e in ref["attempts"]:
        for m in e.get("merged_subclusters") or []:
            lines.append(
                f"- {clusters_text(m['clusters']).capitalize()} is made of "
                f"{len(m['candidates'])} reproducible sub-groups closer than {scale_text}; "
                "they are numbered in `clusters.csv`, column `substructure`.")
    edge = sorted({c for e in ref["attempts"] if e.get("borderline") for c in e["clusters"]})
    if edge:
        lines.append("- Not split because the sub-partition is at the stability threshold "
                     "(reproducibility not established within Monte Carlo error): "
                     + clusters_text(edge) + ".")
    close = s["effect_size"]["below_coordinate_error"]
    if close:
        pairs = ", ".join(f"{p['a']} and {p['b']} ({p['medoid_distance']:.2f}{unit})"
                          for p in close)
        lines.append(
            f"- Clusters closer than {COORDINATE_ERROR}{unit} (medoid RMSD): {pairs}. For "
            "experimental structures this is within the typical coordinate error (Cruickshank "
            "1999); set `selection.refine_min_separation` to the coordinate error of the data to "
            "report such groups as one state.")
    res = s["result"]
    noise = f"{100 * res['noise_fraction']:.0f}% noise"
    if splits:
        noise += (f" ({100 * (res['noise_fraction'] - res['top_level_noise_fraction']):.0f}% "
                  "belong to a top-level cluster but to none of its sub-clusters)")
    lines.append(f"- Result: {res['n_clusters']} clusters, {noise}.")
    diff = s["medoid_difference"]
    if diff:
        lines.append(
            f"- Where the clusters differ: residues {diff['residues']} ({diff['n_residues']} of "
            f"{diff['n_residues_total']}) carry {100 * diff['share']:.0f}% of the squared "
            "deviation between the cluster medoids, averaged over the pairs of clusters (on "
            "the atoms used for clustering; per residue and pair in `medoid_difference.csv`).")
        if diff["n_residues"] <= LOCAL_DIFFERENCE * diff["n_residues_total"]:
            lines.append(
                f"- **The clusters differ mainly in {diff['n_residues']} residues** (at most "
                f"{100 * LOCAL_DIFFERENCE:.0f}% of the residues carry at least half of the "
                "squared deviation). RMSD squares deviations, so a few strongly displaced "
                "residues, such as a flexible terminus or loop, can decide the clusters while "
                "the rest of the molecule is alike. If these residues are not of interest, "
                "run again without them (`input.residues`) and compare the two runs.")
    nested = bool(splits)
    lines += [
        "",
        "## Clusters",
        "",
        "| Cluster |" + (" Top level |" if nested else "")
        + " Size | Selection | Independent Jaccard (2.5-97.5%) | Tier "
        "| Median intra-distance | Medoid |",
        "|---|" + ("---|" if nested else "") + "---|---|---|---|---|---|",
    ]
    for r in cluster_table.itertuples():
        lines.append(
            f"| C{r.cluster} |" + (f" {r.top_level} |" if nested else "")
            + f" {r.size} ({100 * r.fraction:.0f}%) | "
            f"{r.selection_decision.replace('_', ' ')} ({r.selection_estimate:.2f}) | "
            f"{r.independent_stability_estimate:.2f} ({r.independent_stability_low:.2f}-"
            f"{r.independent_stability_high:.2f}) | {r.stability_tier} | "
            f"{r.median_intra_distance:.2f}{unit} | {r.medoid} |"
        )
    hi = run.config.stability.high
    differ = [int(r.cluster) for r in cluster_table.itertuples()
              if r.selection_decision == "established" and r.independent_stability_estimate < hi]
    if differ:
        lines += ["", f"Reproducibility of {clusters_text(differ)} was established in the "
                  f"selection, but the independent estimate is below {hi:.2f}: the true mean "
                  "Jaccard is close to the threshold, and the two estimates come from "
                  "different subsamples."]
    if diff and len(diff["pairs"]) <= 15:
        lines += ["", "## Where the clusters differ", "",
                  "Fewest residues that carry at least half of the squared deviation between "
                  "the two medoids.", "",
                  "| Clusters | Residues | Share |", "|---|---|---|"]
        for p in diff["pairs"]:
            lines.append(f"| C{p['a']}-C{p['b']} | {p['residues']} | {100 * p['share']:.0f}% |")
    if run.populations:
        lines += _population_lines(run)
    hi, mo = run.config.stability.high, run.config.stability.moderate
    lines += [
        "",
        "Selection: whether the resolution selection established the cluster's "
        f"reproducibility (mean Jaccard ≥ {hi:.2f}) at the stated Monte Carlo risk, with its "
        "estimate. Independent Jaccard: a separate estimate from fresh subsamples. "
        f"Tier, from the independent estimate (Hennig 2007): HIGH ≥ {hi:.2f}; MODERATE, "
        f"{mo:.2f}-{hi:.2f}; LOW, < {mo:.2f} (capped by the clusterability verdict)."
        + (" Top level: clusters with the same number are sub-states of one top-level cluster, "
           "separated by refinement." if nested else "")
        + " Biological comparisons in `biology/` are "
        "descriptive; no p-values are given because the clusters were derived from the same data.",
    ]
    if run.biology is not None and not run.biology.torsion_spread.empty:
        res = run.biology.residues.set_index("residue")
        top = run.biology.torsion_spread.head(5)
        lines += ["", "## Torsions that best separate the clusters", ""]
        for t in top.itertuples():
            r = res.loc[t.residue]
            lines.append(f"- {r['resname']}{r['resid']} {t.torsion}: separation "
                         f"{t.separation:.2f}")
    return "\n".join(lines) + "\n"


def _population_lines(run) -> list[str]:
    """Report section: populations with bootstrap intervals, overall and per part."""
    pop = run.populations
    names = list(pop["parts"])
    halves = len(run.ensemble.replicas) == 1

    def cell(p, lo, hi):
        return f"{100 * p:.0f}% ({100 * lo:.0f}-{100 * hi:.0f})"

    lines = ["", "## Cluster populations" + (" by half of the trajectory" if halves
                                             else " by trajectory"), "",
             "Fraction of frames, with an approximate 95% block-bootstrap interval (the "
             "partition held fixed). In simulated two-state trajectories these intervals "
             "contained the true population in 92-99.5% of samples (97% on average); "
             "with few transitions between the clusters they are less reliable "
             "(METHODS.md, section 9).", "",
             "| Cluster | All | " + " | ".join(n.replace("_", " ") for n in names) + " |",
             "|---|---|" + "---|" * len(names)]
    for c in range(run.n_clusters):
        lines.append(f"| C{c + 1} | " + cell(np.mean(run.labels == c), pop["fraction_low"][c],
                                              pop["fraction_high"][c]) + " | "
                     + " | ".join(cell(pop["parts"][n]["population"][c], pop["parts"][n]["low"][c],
                                       pop["parts"][n]["high"][c]) for n in names) + " |")
    lines.append("| noise | " + f"{100 * np.mean(run.labels < 0):.0f}% | " + " | ".join(
        f"{100 * (1 - sum(pop['parts'][n]['population'])):.0f}%" for n in names) + " |")
    differ = population_summary(run)["clusters_differing"]
    if differ:
        lines += ["", "Populations differ between the parts by more than their approximate "
                  "sampling uncertainty (95% family-wise intervals of the difference; "
                  "`population_differences.csv`): "
                  + "; ".join(f"{k.replace('_', ' ')}: {clusters_text(v)}"
                              for k, v in differ.items())
                  + ". The parts may not have sampled the same distribution; the populations "
                  "then describe these simulations, not an equilibrium. The flag is approximate, "
                  "not a test at the 5% level: in simulated stationary trajectories it was "
                  "raised in 8-20% of samples (METHODS.md, section 9)."]
    else:
        lines += ["", "The populations of the parts agree within their sampling uncertainty "
                  "(95% family-wise intervals of the difference). Agreement does not "
                  "prove convergence: states that no part has visited are not seen."]
    return lines


def write_separation(run, out: Path) -> None:
    """separation.csv: the effect size of every pair of final clusters (METHODS.md,
    section 6)."""
    if not run.separation:
        return
    pd.DataFrame([{**{k: v for k, v in p.items() if k != "ratio_subsampling_range"},
                   "ratio_q025": (p["ratio_subsampling_range"] or [None, None])[0],
                   "ratio_q975": (p["ratio_subsampling_range"] or [None, None])[1]}
                  for p in run.separation]).to_csv(out / "separation.csv", index=False)


def write_populations(run, out: Path) -> None:
    """Trajectory mode: populations.csv (every part, with bootstrap intervals) and
    population_differences.csv (every pair of parts; METHODS.md, section 9)."""
    pop = run.populations
    if not pop:
        return
    rows = [{"part": "all", "cluster": c + 1, "n_frames": len(run.labels),
             "population": round(float(np.mean(run.labels == c)), 4),
             "low": round(pop["fraction_low"][c], 4), "high": round(pop["fraction_high"][c], 4)}
            for c in range(run.n_clusters)]
    for name, part in pop["parts"].items():
        rows += [{"part": name, "cluster": c + 1, "n_frames": part["n_frames"],
                  "population": round(part["population"][c], 4),
                  "low": round(part["low"][c], 4), "high": round(part["high"][c], 4)}
                 for c in range(run.n_clusters)]
    pd.DataFrame(rows).to_csv(out / "populations.csv", index=False)
    pd.DataFrame(pop["comparisons"]).to_csv(out / "population_differences.csv", index=False)


def population_summary(run) -> Optional[dict]:
    """run_summary.json entry: the parts compared and the clusters whose populations
    differ between them."""
    pop = run.populations
    if not pop:
        return None
    differ: dict = {}
    for c in pop["comparisons"]:
        if c["differs"]:
            differ.setdefault(f"{c['part_a']} vs {c['part_b']}", []).append(c["cluster"])
    return {"interval_level": pop["interval_level"],
            "flag_level": pop.get("flag_level"),
            "blocks": pop.get("blocks"),
            "parts": {k: {"n_frames": v["n_frames"], "n_blocks": v["n_blocks"]}
                      for k, v in pop["parts"].items()},
            "parts_are": "trajectories" if len(run.ensemble.replicas) > 1 else "halves",
            "max_abs_difference": round(max((abs(c["difference"]) for c in pop["comparisons"]),
                                            default=0.0), 4),
            "clusters_differing": differ,
            "parts_agree": not differ}


def write_results(run) -> dict:
    """Write every output of a finished run; returns the run summary."""
    from struct2clust.figures import SERIES, all_figures

    out = run.out_dir
    clusters = clusters_table(run)
    clusters.to_csv(out / "clusters.csv", index=False)
    if run.duplicates:
        # removed before clustering; listed for completeness, used nowhere else
        pd.DataFrame(run.duplicates, columns=["duplicate_of", "structure", "distance"])[
            ["structure", "duplicate_of", "distance"]].round(6).to_csv(
            out / "duplicates.csv", index=False)
    run.selection.candidates.round(4).to_csv(out / "selection.csv", index=False)

    extra: dict = {}
    table = pd.DataFrame()
    if run.n_clusters:
        table = cluster_summary(run)
        table.to_csv(out / "cluster_summary.csv", index=False)
        write_separation(run, out)
        write_populations(run, out)
        if run.medoid_difference is not None:
            res = residue_table(run.ensemble.atoms)[["residue", "chain", "resid", "resname"]]
            run.medoid_difference.merge(res, on="residue").round(4).to_csv(
                out / "medoid_difference.csv", index=False)
        write_medoids(run, out / "medoids", SERIES)
        if run.config.figures.cluster_views:
            write_cluster_views(run, out / "cluster_views", SERIES)
        if run.biology is not None:
            _write_biology(run, out / "biology")
        if run.config.figures.enabled:
            extra = all_figures(run, out / "figures")

    summary = run_summary(run, extra)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    text = report(run, summary, table) if run.n_clusters else empty_report(run, summary)
    (out / "report.md").write_text(text, encoding="utf-8")
    logger.info("Results written to %s", out)
    return summary
