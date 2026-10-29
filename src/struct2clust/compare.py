"""Comparing two runs on the same structures (e.g. RMSD vs TM-score, backbone vs heavy atoms)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

logger = logging.getLogger(__name__)


def _ordered(names) -> list[str]:
    """C1, C2, ..., noise last."""
    return sorted(names, key=lambda s: (s == "noise", int(s[1:]) if s != "noise" else 0))


def _label(run_dir: Path) -> str:
    summary = run_dir / "run_summary.json"
    if summary.exists():
        return f"{run_dir.name} ({json.loads(summary.read_text())['distance']['label']})"
    return run_dir.name


def compare_runs(run_a: Path, run_b: Path, out_dir: Path) -> dict:
    """Agreement between the clusterings of two finished runs.

    Structures present in both runs are compared; noise is treated as its own
    group. Writes comparison.json, contingency.csv and a contingency figure.
    """
    run_a, run_b, out_dir = Path(run_a), Path(run_b), Path(out_dir)
    a = pd.read_csv(run_a / "clusters.csv", usecols=["structure", "cluster"])
    b = pd.read_csv(run_b / "clusters.csv", usecols=["structure", "cluster"])
    both = a.merge(b, on="structure", suffixes=("_a", "_b"))
    if both.empty:
        raise ValueError("The two runs share no structures.")
    la, lb = both["cluster_a"].to_numpy(), both["cluster_b"].to_numpy()
    clustered = (la > 0) & (lb > 0)
    table = pd.crosstab(
        both["cluster_a"].map(lambda c: f"C{c}" if c else "noise"),
        both["cluster_b"].map(lambda c: f"C{c}" if c else "noise"),
    )
    table = table.loc[_ordered(table.index), _ordered(table.columns)]

    result = {
        "run_a": _label(run_a),
        "run_b": _label(run_b),
        "n_shared_structures": int(len(both)),
        "n_clusters": {"a": int(la.max()), "b": int(lb.max())},
        "adjusted_rand_index": round(float(adjusted_rand_score(la, lb)), 4),
        "adjusted_mutual_information": round(float(adjusted_mutual_info_score(la, lb)), 4),
        "adjusted_rand_index_clustered_in_both": round(
            float(adjusted_rand_score(la[clustered], lb[clustered])), 4
        ) if clustered.sum() > 1 else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    table.to_csv(out_dir / "contingency.csv")
    _contingency_figure(table, result, out_dir / "contingency")
    logger.info("ARI %.3f, AMI %.3f over %d shared structures",
                result["adjusted_rand_index"], result["adjusted_mutual_information"], len(both))
    return result


def _contingency_figure(table: pd.DataFrame, result: dict, out_base: Path) -> None:
    import matplotlib.pyplot as plt

    from struct2clust.figures import INK, MM, SEQUENTIAL, STYLE

    frac = table.to_numpy() / table.to_numpy().sum(axis=1, keepdims=True)
    with plt.rc_context(STYLE):
        w = min(183, 30 + 7 * table.shape[1]) * MM
        h = (25 + 6 * table.shape[0]) * MM
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
        im = ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
        for (i, j), n in np.ndenumerate(table.to_numpy()):
            if n:
                ax.text(j, i, str(n), ha="center", va="center", fontsize=5.5,
                        color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(table.shape[1]), table.columns)
        ax.set_yticks(range(table.shape[0]), table.index)
        ax.set_xlabel(result["run_b"])
        ax.set_ylabel(result["run_a"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("Fraction of row cluster")
        ax.set_title(f"ARI {result['adjusted_rand_index']:.2f} · "
                     f"AMI {result['adjusted_mutual_information']:.2f}", loc="left", color=INK)
        for fmt in ("pdf", "png"):
            fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
