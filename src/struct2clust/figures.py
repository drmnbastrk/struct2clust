"""Figures.

Widths are 89 mm (single column) or 183 mm (double column), text is 7 pt
sans-serif and embedded as TrueType, axes and grids are hairlines.

Cluster identity never relies on colour alone: in the UMAP each cluster carries
its number, elsewhere clusters are identified by position on an axis. The
categorical palette (8 hues, validated for colour-vision deficiency on adjacent
use) is only an aid; magnitudes use a single-hue light-to-dark ramp.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

logger = logging.getLogger(__name__)

MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NOISE = "#e1e0d9", "#c3c2b7", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = SERIES[0]
SEQUENTIAL = LinearSegmentedColormap.from_list("s2c_blue", [
    "#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
])
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
         "zeta": "ζ", "chi": "χ", "phi": "φ", "psi": "ψ"}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}


def cluster_color(c: int) -> str:
    """Colour of 0-based cluster c (noise: -1). Past 8, hues repeat but every
    cluster is also labelled by number, so identity never rests on colour."""
    return NOISE if c < 0 else SERIES[c % len(SERIES)]


def _save(fig, out_base: Path, formats, dpi) -> list[Path]:
    paths = []
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def _panel_letter(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=INK, ha="right", va="bottom")


def _unit(unit: str) -> str:
    return f" ({unit})" if unit else ""


def member_order(dist: np.ndarray, labels: np.ndarray, medoids: dict) -> np.ndarray:
    """Members grouped by cluster (noise last), each group sorted by distance to its medoid."""
    order = []
    for c in range(labels.max() + 1):
        idx = np.flatnonzero(labels == c)
        order.extend(idx[np.argsort(dist[medoids[c], idx])])
    order.extend(np.flatnonzero(labels < 0))
    return np.asarray(order)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _distance_heatmap(ax, run):
    dist, labels = run.distances, run.labels
    order = member_order(dist, labels, run.medoids)
    im = ax.imshow(dist[np.ix_(order, order)], cmap=SEQUENTIAL, interpolation="nearest",
                   aspect="equal")
    sizes = [int((labels == c).sum()) for c in range(run.n_clusters)]
    bounds = np.cumsum(sizes)
    has_noise = bool((labels < 0).any())
    for b in (bounds if has_noise else bounds[:-1]):  # white gap between blocks
        ax.axhline(b - 0.5, color="white", lw=0.8)
        ax.axvline(b - 0.5, color="white", lw=0.8)
    starts = np.r_[0, bounds[:-1]]
    ticks = [(s + e) / 2 - 0.5 for s, e in zip(starts, bounds)]
    names = [f"C{c + 1}" for c in range(run.n_clusters)]
    if has_noise:
        ticks.append((bounds[-1] + len(labels)) / 2 - 0.5)
        names.append("noise")
    # label only clusters large enough for the text to fit
    keep = [i for i, t in enumerate(ticks) if i == len(sizes) or sizes[i] >= len(labels) / 40]
    ax.set_xticks([ticks[i] for i in keep], [names[i] for i in keep], rotation=90)
    ax.set_yticks([ticks[i] for i in keep], [names[i] for i in keep])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label(run.metric_label + _unit(run.distance_unit), color=INK2)


def _core(pts):
    """The points of a cluster inside its 5-95% box, so that a few stray points
    do not pull the label away from the cluster."""
    if len(pts) < 20:
        return pts
    lo, hi = np.percentile(pts, [5, 95], axis=0)
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[inside] if inside.any() else pts


def _label_positions(embedding, labels, k, box=0.035):
    """A spot next to each cluster that overlaps neither points nor other labels.

    Candidates are placed around each cluster's bounding box; the first one far
    enough (in units of the plot span) from all points and earlier labels wins.
    """
    span = np.ptp(embedding, axis=0).max() or 1.0
    gap = box * span
    placed = []
    order = np.argsort([-(labels == c).sum() for c in range(k)])  # big clusters first
    out = [None] * k
    for c in order:
        pts = _core(embedding[labels == c])
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        candidates = [(x1 + gap, y1 + gap), (x0 - gap, y1 + gap), (x1 + gap, y0 - gap),
                      (x0 - gap, y0 - gap), (xm, y1 + gap), (xm, y0 - gap), (x1 + gap, ym),
                      (x0 - gap, ym)]
        for r in (1.0, 1.8, 2.6):
            best = None
            for cx, cy in candidates:
                cx, cy = xm + (cx - xm) * r, ym + (cy - ym) * r
                d_pts = np.min(np.abs(embedding - (cx, cy)).max(axis=1))
                d_lab = min((max(abs(cx - px), abs(cy - py)) for px, py in placed), default=np.inf)
                if d_pts > 0.6 * gap and d_lab > 1.2 * gap:
                    best = (cx, cy)
                    break
            if best:
                break
        out[c] = best or candidates[0]
        placed.append(out[c])
    return out


def _umap_panel(ax, run, embedding):
    labels = run.labels
    noise = labels < 0
    ax.scatter(*embedding[noise].T, s=4, c=NOISE, lw=0, label="noise", zorder=1)
    for c in range(run.n_clusters):
        pts = embedding[labels == c]
        ax.scatter(*pts.T, s=7, color=cluster_color(c), lw=0.25, edgecolor="white", zorder=2)
        med = embedding[run.medoids[c]]
        ax.scatter(*med, s=16, facecolor="none", edgecolor=INK, lw=0.6, zorder=3)
    for c, (x, y) in enumerate(_label_positions(embedding, labels, run.n_clusters)):
        pts = _core(embedding[labels == c])
        near = pts[np.argmin(((pts - (x, y)) ** 2).sum(axis=1))]
        ax.annotate(str(c + 1), near, xytext=(x, y), fontsize=6.5, fontweight="bold", color=INK,
                    ha="center", va="center", zorder=4,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.4,
                                "shrinkA": 3, "shrinkB": 1})
    ax.scatter([], [], s=16, facecolor="none", edgecolor=INK, lw=0.6, label="medoid")
    ax.legend(loc="best", handletextpad=0.2, borderaxespad=0.2)
    ax.margins(0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def _selection_panel(ax, run, threshold):
    t = run.selection.candidates.dropna(subset=["min_jaccard"])
    ok = t["qualifies"].astype(bool)
    rng = np.random.default_rng(0)
    x = t["n_clusters"] + rng.uniform(-0.18, 0.18, len(t))
    ax.scatter(x[~ok], t.loc[~ok, "min_jaccard"], s=8, color=AXIS, lw=0, label="not all stable")
    ax.scatter(x[ok], t.loc[ok, "min_jaccard"], s=8, color=ACCENT, lw=0,
               label="all clusters stable")
    sel = t["selected"].to_numpy()
    ax.scatter(x[sel], t.loc[sel, "min_jaccard"], s=40, facecolor="none", edgecolor=INK,
               lw=0.8, label="selected", zorder=3)
    ax.axhline(threshold, color=MUTED, lw=0.5)
    ax.annotate(f"stable ({threshold:.2f})", (1.0, threshold), xycoords=("axes fraction", "data"),
                xytext=(0, 2), textcoords="offset points", color=MUTED, ha="right", va="bottom",
                fontsize=6)
    refined = run.n_clusters != run.selection.n_clusters
    ax.set_xlabel("Number of top-level clusters" if refined else "Number of clusters")
    ax.set_ylabel("Weakest-cluster Jaccard")
    ax.set_ylim(-0.02, 1.02)
    k_lo, k_hi = int(t["n_clusters"].min()), int(t["n_clusters"].max())
    k_lo, k_hi = min(k_lo, max(2, k_hi - 2)), max(k_hi, k_lo + 2)   # show at least 3 values
    if k_hi > 30:   # a few very fine candidates would squeeze the informative range
        ax.set_xscale("log")
        ax.set_xlim(k_lo * 0.85, k_hi * 1.2)
        nice = [2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(
            [v for v in nice if k_lo * 0.85 <= v <= k_hi * 1.2]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        ax.set_xlim(k_lo - 0.6, k_hi + 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=3))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", handletextpad=0.2, borderaxespad=0.2)


def _stability_panel(ax, run, high, moderate):
    tab = run.stability.clusters
    y = np.arange(len(tab))[::-1]
    # the mean can lie outside a percentile interval of a skewed distribution
    lo = (tab["jaccard_mean"] - tab["jaccard_q025"]).clip(lower=0)
    hi = (tab["jaccard_q975"] - tab["jaccard_mean"]).clip(lower=0)
    ax.axvspan(high, 1.02, color="#f0f5fc", lw=0, zorder=0)
    for v in (moderate, high):
        ax.axvline(v, color=MUTED, lw=0.5, zorder=1)
    ax.errorbar(tab["jaccard_mean"], y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.6,
                capsize=0, zorder=2)
    ax.scatter(tab["jaccard_mean"], y, s=18, color=ACCENT, edgecolor="white", lw=0.5, zorder=3)
    ax.set_yticks(y, [f"C{r.cluster} (n={r.size})  {r.tier}" for r in tab.itertuples()])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, len(tab) - 0.3)
    ax.set_xlabel("Mean Jaccard (bar: 2.5-97.5% of subsamples)")
    ax.text(high + 0.01, len(tab) - 0.45, "stable", color=MUTED, fontsize=6, va="bottom")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def overview(run, embedding, out_dir: Path, formats, dpi) -> list[Path]:
    st = run.config.stability
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(DOUBLE, DOUBLE * 0.82), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        _distance_heatmap(axes[0], run)
        _umap_panel(axes[1], run, embedding)
        _selection_panel(axes[2], run, st.high)
        _stability_panel(axes[3], run, st.high, st.moderate)
        for ax, letter in zip(axes, "abcd"):
            _panel_letter(ax, letter)
        return _save(fig, out_dir / "overview", formats, dpi)


def distances(run, out_dir: Path, formats, dpi) -> list[Path]:
    dist, labels = run.distances, run.labels
    groups, names = [], []
    for c in range(run.n_clusters):
        idx = np.flatnonzero(labels == c)
        groups.append(dist[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)])
        names.append(f"C{c + 1}")
    clustered = np.flatnonzero(labels >= 0)
    between = dist[np.ix_(clustered, clustered)][
        (labels[clustered][:, None] != labels[clustered][None, :])
    ]
    groups.append(between)
    names.append("between")
    with plt.rc_context(STYLE):
        width = min(DOUBLE, max(SINGLE, (len(groups) + 1) * 7 * MM))
        fig, ax = plt.subplots(figsize=(width, 55 * MM), constrained_layout=True)
        bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 0.8},
                        whiskerprops={"color": INK2, "lw": 0.5},
                        capprops={"color": INK2, "lw": 0.5})
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor="#cde2fb" if i < len(groups) - 1 else "#f0efec",
                    edgecolor=INK2, lw=0.5)
        ax.set_xticks(range(1, len(names) + 1), names)
        ax.set_ylabel(run.metric_label + _unit(run.distance_unit))
        ax.set_xlabel("Pairs within each cluster, and between clusters")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        return _save(fig, out_dir / "distances", formats, dpi)


def _residue_labels(residues):
    return [f"{rn}{rid}" for rn, rid in zip(residues["resname"], residues["resid"])]


def rmsf(run, out_dir: Path, formats, dpi) -> list[Path]:
    bio = run.biology
    table = bio.rmsf.pivot(index="cluster", columns="residue", values="rmsf")
    n_res = table.shape[1]
    with plt.rc_context(STYLE):
        width = DOUBLE if n_res > 30 else SINGLE
        fig, ax = plt.subplots(figsize=(width, (18 + 4.5 * len(table)) * MM),
                               constrained_layout=True)
        im = ax.imshow(table.to_numpy(), cmap=SEQUENTIAL, aspect="auto", interpolation="nearest")
        labels = _residue_labels(bio.residues)
        step = 1 if n_res <= 60 else int(np.ceil(n_res / 60))
        ax.set_xticks(range(0, n_res, step), labels[::step], rotation=90, fontsize=5)
        ax.set_yticks(range(len(table)), [f"C{c}" for c in table.index])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.outline.set_visible(False)
        cb.set_label("RMSF (Å)", color=INK2)
        ax.set_xlabel("Residue")
        return _save(fig, out_dir / "rmsf", formats, dpi)


def contacts(run, out_dir: Path, formats, dpi) -> list[Path]:
    bio = run.biology
    k = len(bio.contacts)
    ncol = min(4, k)
    nrow = int(np.ceil(k / ncol))
    with plt.rc_context(STYLE):
        panel = DOUBLE / 4.3
        fig, axes = plt.subplots(nrow, ncol, figsize=(panel * ncol + 12 * MM, panel * nrow),
                                 constrained_layout=True, squeeze=False)
        n_res = len(bio.residues)
        for ax in axes.flat[k:]:
            ax.set_visible(False)
        for ax, (c, freq) in zip(axes.flat, bio.contacts.items()):
            im = ax.imshow(freq, cmap=SEQUENTIAL, vmin=0, vmax=1, origin="lower",
                           interpolation="nearest")
            size = int((run.labels == c - 1).sum())
            ax.set_title(f"C{c} (n={size})", color=INK, loc="left", pad=2)
            ticks = np.linspace(0, n_res - 1, min(5, n_res)).round().astype(int)
            ax.set_xticks(ticks, bio.residues["resid"].to_numpy()[ticks])
            ax.set_yticks(ticks, bio.residues["resid"].to_numpy()[ticks])
            for s in ax.spines.values():
                s.set_visible(False)
        cb = fig.colorbar(im, ax=axes[:, -1].tolist(), fraction=0.05, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("Contact frequency", color=INK2)
        return _save(fig, out_dir / "contacts", formats, dpi)


def torsions(run, out_dir: Path, formats, dpi, n_top: int = 6) -> list[Path]:
    from struct2clust.biology import torsion_angles
    bio = run.biology
    top = bio.torsion_spread.head(n_top)
    if top.empty:
        return []
    angles = torsion_angles(run.ensemble)
    angles["cluster"] = run.labels[angles["member"].to_numpy()] + 1
    angles = angles[angles["cluster"] > 0]
    res = bio.residues.set_index("residue")
    ncol = min(3, len(top))
    nrow = int(np.ceil(len(top) / ncol))
    rng = np.random.default_rng(0)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE, 42 * MM * nrow),
                                 constrained_layout=True, squeeze=False, sharey=True)
        for ax in axes.flat[len(top):]:
            ax.set_visible(False)
        for ax, row in zip(axes.flat, top.itertuples()):
            sub = angles[(angles["residue"] == row.residue) & (angles["torsion"] == row.torsion)]
            x = sub["cluster"] + rng.uniform(-0.22, 0.22, len(sub))
            ax.scatter(x, sub["angle"], s=3, color=ACCENT, alpha=0.6, lw=0)
            r = res.loc[row.residue]
            ax.set_title(
                f"{r['resname']}{r['resid']} {GREEK.get(row.torsion, row.torsion)}  "
                f"(separation {row.separation:.2f})", loc="left", color=INK, pad=2,
            )
            ax.set_yticks([-180, -90, 0, 90, 180])
            ax.set_ylim(-185, 185)
            ks = sorted(sub["cluster"].unique())
            ax.set_xticks(ks, [f"C{c}" for c in ks])
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Angle (°)")
        return _save(fig, out_dir / "torsions", formats, dpi)


MAX_TIMELINE_PANELS = 4


def populations(run, out_dir: Path, formats, dpi) -> list[Path]:
    """Several trajectories: fraction of each trajectory's frames in each cluster."""
    ens = run.ensemble
    replica = np.asarray(ens.replica)
    reps, k = ens.replicas, run.n_clusters
    frac = np.array([[np.mean(run.labels[replica == r] == c) for c in [*range(k), -1]]
                     for r in reps])
    cols = [f"C{c + 1}" for c in range(k)] + ["noise"]
    with plt.rc_context(STYLE):
        width = min(DOUBLE, (30 + 5.5 * len(cols)) * MM)
        height = (22 + 4.2 * len(reps)) * MM
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
        im = ax.imshow(frac, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto",
                       interpolation="nearest")
        for (i, j), v in np.ndenumerate(frac):
            if v >= 0.05:   # label only substantial populations
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(cols)), cols, rotation=90)
        ax.set_yticks(range(len(reps)), reps)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Cluster")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        cb.set_label("Frames of the trajectory in the cluster", color=INK2)
        return _save(fig, out_dir / "populations", formats, dpi)


def timeline(run, out_dir: Path, formats, dpi) -> list[Path]:
    """Trajectory mode: cluster of every frame over time, one panel per replica."""
    ens = run.ensemble
    replica, frame = np.asarray(ens.replica), np.asarray(ens.frame)
    k = run.n_clusters
    names = [f"C{c + 1}" for c in range(k)] + ["noise"]
    reps = ens.replicas
    with plt.rc_context(STYLE):
        height = (14 + (6 + 3.2 * (k + 1)) * len(reps)) * MM
        fig, axes = plt.subplots(len(reps), 1, figsize=(DOUBLE, height),
                                 constrained_layout=True, squeeze=False, sharey=True)
        for ax, label in zip(axes[:, 0], reps):
            m = replica == label
            lab = run.labels[m]
            y = np.where(lab >= 0, lab, k)
            clustered = lab >= 0
            ax.scatter(frame[m][clustered], y[clustered], s=3, color=ACCENT, lw=0)
            ax.scatter(frame[m][~clustered], y[~clustered], s=3, color=NOISE, lw=0)
            ax.set_yticks(range(k + 1), names)
            ax.set_ylim(k + 0.6, -0.6)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            if len(reps) > 1:
                ax.set_title(label, loc="left", color=INK, pad=2)
        axes[-1, 0].set_xlabel("Frame")
        return _save(fig, out_dir / "timeline", formats, dpi)


# trustworthiness of the map is computed on at most this many structures
TRUST_SAMPLE = 3000


def umap_embedding(dist: np.ndarray, n_neighbors: int, min_dist: float, seed: int):
    """2-D UMAP of the precomputed distances and its trustworthiness (k = 10), computed on
    TRUST_SAMPLE structures drawn at random when there are more (trustworthiness ranks all
    distances and needs several N x N arrays)."""
    import umap
    from sklearn.manifold import trustworthiness

    n_neighbors = min(n_neighbors, len(dist) - 1)
    emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="precomputed",
                    random_state=seed).fit_transform(dist)
    idx = np.arange(len(dist))
    if len(dist) > TRUST_SAMPLE:
        idx = np.sort(np.random.default_rng(seed).choice(len(dist), TRUST_SAMPLE,
                                                         replace=False))
    trust = trustworthiness(dist[np.ix_(idx, idx)], emb[idx],
                            n_neighbors=min(10, len(idx) // 2 - 1), metric="precomputed")
    return emb, float(trust)


def all_figures(run, out_dir: Path) -> dict:
    """Write every figure; returns {"files": [...], "umap_trustworthiness": float}."""
    f = run.config.figures
    out_dir.mkdir(parents=True, exist_ok=True)
    emb, trust = umap_embedding(run.distances, f.umap_n_neighbors, f.umap_min_dist,
                                run.config.seed)
    files = overview(run, emb, out_dir, f.formats, f.dpi)
    files += distances(run, out_dir, f.formats, f.dpi)
    reps = run.ensemble.replicas
    if reps and len(reps) <= MAX_TIMELINE_PANELS:
        files += timeline(run, out_dir, f.formats, f.dpi)
    if len(reps) > 1:
        files += populations(run, out_dir, f.formats, f.dpi)
    if run.biology is not None:
        files += rmsf(run, out_dir, f.formats, f.dpi)
        files += contacts(run, out_dir, f.formats, f.dpi)
        files += torsions(run, out_dir, f.formats, f.dpi)
    logger.info("Figures written to %s (UMAP trustworthiness %.2f)", out_dir, trust)
    return {"files": files, "umap_trustworthiness": trust, "embedding": emb}
