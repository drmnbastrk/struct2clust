"""Does the naming of chemically equivalent atoms change heavy-atom RMSD and the clusters?

    python validation/symmetry_check.py example/data/eralpha --out validation/results/symmetry
    python validation/symmetry_check.py -c example/1rkj_rna_bound_vs_free.yaml --out ...

Heavy-atom RMSD pairs atoms by name. Atoms that the heavy-atom structure of a
residue cannot tell apart (struct2clust.metrics.EQUIVALENT_ATOMS and the
terminal and phosphate groups; checked by validation/equivalent_atoms.py) can be
named either way round, and a different naming in two structures counts as a
deviation although the heavy atoms are in the same places.

Three distance matrices are computed with struct2clust.metrics.rmsd_matrix:
- usual: atoms paired by name (the default of Struct2Clust);
- aromatic_charged: equivalent atoms matched for Asp, Glu, Phe, Tyr, Arg and the
  phosphate oxygens only (the subset examined first);
- all_equivalent: every group of the table (adds Leu, Val and the C-terminal
  carboxylate) -- the option metric.equivalent_atoms.
Each matrix is clustered as `struct2clust run` clusters it, with the settings of the
configuration file if given (near-duplicates removed from ensembles, not from
trajectories), and the results are written side by side.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np

FIRST_SUBSET = {"ASP", "GLU", "PHE", "TYR", "ARG"}


def groups_for(atoms, variant: str):
    from struct2clust import metrics

    if variant == "usual":
        return None
    saved = (dict(metrics.EQUIVALENT_ATOMS), metrics.C_TERMINUS)
    try:
        if variant == "aromatic_charged":
            for name in list(metrics.EQUIVALENT_ATOMS):
                if name not in FIRST_SUBSET:
                    del metrics.EQUIVALENT_ATOMS[name]
            metrics.C_TERMINUS = ("", "")          # never present
        return metrics.equivalent_atom_groups(atoms)
    finally:
        metrics.EQUIVALENT_ATOMS.clear()
        metrics.EQUIVALENT_ATOMS.update(saved[0])
        metrics.C_TERMINUS = saved[1]


def cluster(ens, dist, cfg):
    from struct2clust.pipeline import Run, near_duplicates, run_conclusion
    from struct2clust.pipeline import cluster as run_cluster

    kept, _, _ = near_duplicates(cfg, dist, ens.names)
    sub = ens.subset(kept)
    run = Run(config=cfg, out_dir=None, ensemble=sub, distances=dist[np.ix_(kept, kept)],
              metric_label="", distance_unit="Å", n_input=len(ens))
    t0 = time.time()
    run_cluster(run)
    ca = run.clusterability
    tab = run.stability.clusters if run.stability is not None else None
    return {"n_used": len(kept), "clusterability": ca.verdict if ca else None,
            "conclusion": run_conclusion(run)[0],
            "n_clusters": int(run.n_clusters), "noise": round(float(np.mean(run.labels < 0)), 4),
            "cluster_sizes": [int(v) for v in tab["size"]] if tab is not None else [],
            "selection_estimates": run.selection.cluster_estimates,
            "independent_estimates": [round(v, 3) for v in tab["jaccard_mean"]]
            if tab is not None else [],
            "seconds": round(time.time() - t0, 1)}, dict(zip(sub.names, run.labels.tolist()))


def main() -> None:
    from sklearn.metrics import adjusted_rand_score

    from struct2clust.cli import build_config
    from struct2clust.metrics import rmsd_matrix
    from struct2clust.pipeline import load_input
    from struct2clust.structures import atom_mask

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", type=Path)
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--name", help="name of the result file (default: the input's name)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    cfg = build_config(args.config, n_jobs=args.n_jobs)
    cfg.biology.enabled = False
    ens = load_input(cfg, args.input)
    mask = atom_mask(ens.atoms, "heavy-atoms")
    atoms, coords = ens.atoms.subset(mask), ens.coords[:, mask]
    iu = np.triu_indices(len(ens), 1)
    result = {"input": str(args.input or args.config), "n_structures": len(ens),
              "atoms": int(mask.sum()), "variants": {}}
    labels, plain = {}, None
    for variant in ("usual", "aromatic_charged", "all_equivalent"):
        groups = groups_for(atoms, variant)
        t0 = time.time()
        d = rmsd_matrix(coords, args.n_jobs, groups)
        entry = {"groups_per_structure": int(sum(len(i) for i, _ in groups)) if groups else 0,
                 "seconds_for_matrix": round(time.time() - t0, 1)}
        if plain is None:
            plain = d
        else:
            drop = plain[iu] - d[iu]
            entry["rmsd_decrease"] = {
                "pairs_with_a_decrease": round(float(np.mean(drop > 1e-6)), 4),
                "median": round(float(np.median(drop)), 4),
                "p95": round(float(np.quantile(drop, 0.95)), 4),
                "max": round(float(drop.max()), 4),
                "share_above_0.1_A": round(float(np.mean(drop > 0.1)), 4),
                "never_larger": bool(np.all(drop > -1e-6))}
        entry["clustering"], labels[variant] = cluster(ens, d, cfg)
        result["variants"][variant] = entry
    for variant in ("aromatic_charged", "all_equivalent"):
        shared = sorted(set(labels["usual"]) & set(labels[variant]))
        a = [labels["usual"][s] for s in shared]
        b = [labels[variant][s] for s in shared]
        result["variants"][variant]["ari_with_usual"] = round(float(adjusted_rand_score(a, b)), 4)
        result["variants"][variant]["structures_in_both"] = len(shared)
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or (args.input or args.config).stem
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
