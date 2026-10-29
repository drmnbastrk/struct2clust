"""Where the members of each cluster come from: clusters of a finished run against a
label of every structure (prediction method, PDB entry, ligand, simulation, ...).

The comparison is descriptive. Agreement between clusters and sources can mean that
the sources sample different conformations, or that the clusters reflect how the
structures were made (docs/METHODS.md, section 10); it does not validate the clusters.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

logger = logging.getLogger(__name__)

NOISE = "noise"
UNKNOWN = "unknown"


def cramers_v(table: pd.DataFrame) -> float:
    """Cramér's V of a contingency table (Cramér 1946): 0 = no association; 1 = the
    classification with fewer classes is fully determined by the other."""
    obs = table.to_numpy(float)
    n = obs.sum()
    k = min(obs.shape) - 1
    if n == 0 or k == 0:
        return float("nan")
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    chi2 = float(np.sum(np.where(exp > 0, (obs - exp) ** 2 / np.where(exp > 0, exp, 1), 0)))
    return float(np.sqrt(chi2 / (n * k)))


def cramers_v_corrected(table: pd.DataFrame) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013). The classical estimate is biased upwards
    in small samples and large tables; the correction subtracts the expected value of
    phi^2 under independence and adjusts the table dimensions accordingly."""
    obs = table.to_numpy(float)
    n = obs.sum()
    r, k = obs.shape
    if n <= 1 or min(r, k) < 2:
        return float("nan")
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / n
    phi2 = float(np.sum(np.where(exp > 0, (obs - exp) ** 2 / np.where(exp > 0, exp, 1), 0))) / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def sparse_share(table: pd.DataFrame) -> float:
    """Share of cells whose expected count under independence is below 5."""
    obs = table.to_numpy(float)
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / obs.sum()
    return float(np.mean(exp < 5))


def _agreement(clusters: pd.Series, sources: pd.Series) -> dict:
    """Cramér's V, ARI and AMI between clusters and sources (noise and structures
    without a source excluded); None when fewer than two clusters or sources remain."""
    keep = (clusters > 0) & (sources != UNKNOWN)
    c, s = clusters[keep], sources[keep]
    if c.nunique() < 2 or s.nunique() < 2:
        return {"cramers_v": None, "cramers_v_corrected": None, "adjusted_rand_index": None,
                "adjusted_mutual_information": None, "sparse_cells": None}
    table = pd.crosstab(c, s)
    return {"cramers_v": round(cramers_v(table), 4),
            "cramers_v_corrected": round(cramers_v_corrected(table), 4),
            "sparse_cells": round(sparse_share(table), 3),
            "adjusted_rand_index": round(float(adjusted_rand_score(s, c)), 4),
            "adjusted_mutual_information": round(float(adjusted_mutual_info_score(s, c)), 4)}


def composition(run_dir: Path, sources: Path, out_dir: Path, label: str = "Source") -> dict:
    """Cluster composition by source; writes tables, a figure and composition.json."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cl = pd.read_csv(run_dir / "clusters.csv")
    src = pd.read_csv(sources, dtype=str)
    if not {"structure", "source"} <= set(src.columns):
        raise ValueError(f"{sources}: the columns 'structure' and 'source' are required")
    df = cl.merge(src[["structure", "source"]], on="structure", how="left")
    df["source"] = df["source"].fillna(UNKNOWN)
    if (df["source"] == UNKNOWN).all():
        raise ValueError(f"{sources}: no structure name matches the run; the column "
                         "'structure' must use the names of clusters.csv")
    dup_file = run_dir / "duplicates.csv"
    duplicates = set(pd.read_csv(dup_file)["structure"]) if dup_file.exists() else set()
    missing = src.loc[~src["structure"].isin(cl["structure"]), "structure"]
    n_duplicates = int(missing.isin(duplicates).sum())
    n_unused = len(missing) - n_duplicates
    if n_duplicates:
        logger.info("%d structure(s) of %s are near-duplicates removed before clustering "
                    "(duplicates.csv) and are not counted", n_duplicates, Path(sources).name)
    if n_unused:
        logger.warning("%d structure(s) of %s are not in the run", n_unused, Path(sources).name)
    n_unknown = int((df["source"] == UNKNOWN).sum())
    if n_unknown:
        logger.warning("%d structure(s) of the run have no source", n_unknown)

    k = int(df["cluster"].max())
    rows = [f"C{c}" for c in range(1, k + 1)] + ([NOISE] if (df["cluster"] == 0).any() else [])
    df["row"] = np.where(df["cluster"] > 0, "C" + df["cluster"].astype(str), NOISE)
    cols = df["source"].value_counts().index.tolist()
    counts = pd.crosstab(df["row"], df["source"]).reindex(index=rows, columns=cols, fill_value=0)
    counts.index.name, counts.columns.name = "cluster", "source"
    fractions = counts.div(counts.sum(axis=1), axis=0)
    by_source = counts.div(counts.sum(axis=0), axis=1).T

    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "counts.csv")
    fractions.round(4).to_csv(out_dir / "fractions.csv")
    by_source.round(4).to_csv(out_dir / "by_source.csv")

    top = cl.groupby("cluster")["top_level"].first() if "top_level" in cl else None
    per_cluster = []
    for r in rows:
        c = counts.loc[r]
        entry = {"cluster": r, "size": int(c.sum()), "main_source": c.idxmax(),
                 "purity": round(float(c.max() / c.sum()), 4), "n_sources": int((c > 0).sum())}
        if top is not None and r != NOISE:
            entry["top_level"] = int(top[int(r[1:])])
        per_cluster.append(entry)
    result = {
        "run": run_dir.name, "sources": Path(sources).name, "label": label,
        "n_structures": len(df), "n_clusters": k, "n_sources": len(cols),
        "n_noise": int((df["cluster"] == 0).sum()), "n_without_source": n_unknown,
        "clusters": _agreement(df["cluster"], df["source"]),
        "clusters_table": per_cluster,
    }
    if "top_level" in df and df.groupby("top_level")["cluster"].nunique().max() > 1:
        result["top_level_clusters"] = _agreement(df["top_level"], df["source"])
    (out_dir / "composition.json").write_text(json.dumps(result, indent=2))
    _figure(counts, by_source, label, run_dir.name, out_dir / "composition")
    logger.info("Composition of %s by %s -> %s", run_dir.name, label.lower(), out_dir)
    return result


def _figure(counts: pd.DataFrame, by_source: pd.DataFrame, label: str, title: str,
            out_base: Path) -> None:
    """(a) members of each cluster by source; (b) clusters of each source's structures."""
    import matplotlib.pyplot as plt

    from struct2clust.figures import DOUBLE, INK, INK2, MM, STYLE, cluster_color

    rows, cols = list(counts.index), list(counts.columns)
    # sources in light tones, so they are not mistaken for the cluster colours
    tones = list(plt.get_cmap("Set3").colors) + list(plt.get_cmap("Pastel2").colors)
    colour = {s: ("#d9d9d9" if s == UNKNOWN else tones[i % len(tones)])
              for i, s in enumerate(cols)}

    def cluster_colour(r):
        return "#ffffff" if r == NOISE else cluster_color(int(r[1:]) - 1)

    n = max(len(rows), len(cols))
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(DOUBLE, (30 + 5 * n + 3 * ((len(cols) + 1) // 2)) * MM),
            constrained_layout=True)
        left = np.zeros(len(rows))
        for s in cols:
            v = counts[s].to_numpy()
            ax1.barh(range(len(rows)), v, left=left, color=colour[s], label=s, height=0.75,
                     edgecolor=INK2, linewidth=0.3)
            left += v
        for i, total in enumerate(left):
            ax1.text(total, i, f" {int(total)}", va="center", fontsize=6, color=INK2)
        ax1.set_yticks(range(len(rows)), rows)
        for t, r in zip(ax1.get_yticklabels(), rows):
            t.set_color(INK if r == NOISE else cluster_colour(r))
        ax1.invert_yaxis()
        ax1.set_xlabel("Structures")
        ax1.set_title(f"(a) Members of each cluster by {label.lower()}", loc="left", color=INK)
        ax1.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=2, frameon=False,
                   title=label, alignment="left")

        left = np.zeros(len(cols))
        for r in rows:
            v = by_source[r].to_numpy()
            ax2.barh(range(len(cols)), v, left=left, color=cluster_colour(r), edgecolor=INK2,
                     linewidth=0.3, height=0.75)
            for i, x in enumerate(v):
                if x >= 0.08:
                    ax2.text(left[i] + x / 2, i, r, ha="center", va="center", fontsize=5.5,
                             color=INK if r == NOISE else "white")
            left += v
        ax2.set_yticks(range(len(cols)), [f"{s} (n={int(counts[s].sum())})" for s in cols])
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Fraction of the structures of the source")
        ax2.set_title("(b) Clusters of each source", loc="left", color=INK)
        fig.suptitle(title, x=0.01, ha="left", fontsize=8, fontweight="bold", color=INK)
        for fmt in ("pdf", "png"):
            fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
