"""Synthetic data sets with a known answer, for the validation study.

Two families:

top_*     ONE data set per scenario, clustered as it is.
          top_null_*  structureless (a single group of some shape): no reliable
                      cluster structure should be reported.
          top_pos_*   several groups: they should be found.
refine_*  TWO or more groups that are well separated at the top level; what
          happens INSIDE the groups tests the hierarchical refinement.
          refine_null_*  no structure inside the groups: nothing may split.
          refine_pos_*   three sub-groups hidden by a far-away group: they
                         should be recovered.
          refine_scale_* two tight, reproducible sub-groups inside one group,
                         whose medoids are f times the state scale apart (the
                         median member-to-medoid distance of the other group).
                         They characterise the optional effect-size rule (run
                         with selection.refine_min_separation_ratio = 1): by
                         design f < 1 is reported as one state, f > 1 as two.
top_null_copies  40 "entries" drawn from one structureless Gaussian group, each
                 present as 6 near-identical copies (SD 0.05): no distinct
                 states. Measures how far dependent copies (chain copies, models
                 of one entry) can create reproducible clusters by themselves.
top_pos_2groups_6sigma_<a>_<b>  two groups 6 sigma apart, unequal (300 points split
                 a:b): the smaller group should be found.
top_pos_<k>equidistant_6sigma   k = 4 or 6 groups of 50 points, every pair 6 sigma
                 apart: all k should be found.
traj_*           trajectories (replica labels), run as struct2clust runs trajectories:
                 traj_null_replicas_meet  two replicas of the same structureless
                                          drift (AR(1)), which overlap: no states
                 traj_pos_markov_<speed>  two states 6 sigma apart with AR(1) noise
                                          (phi 0.9), 1000 frames, switching with
                                          probability 0.005 (slow, about 5
                                          transitions), 0.02 (about 20) or 0.1 (fast,
                                          about 100) per frame: two states
                 traj_pos_markov_unequal  as above, 0.01 out of the large state and
                                          0.04 out of the small one (populations
                                          80:20): two states
                 traj_pos_markov_replicas two replicas of 500 frames, switching 0.02,
                                          starting in different states: two states
                 The structureless drift of one trajectory and of two replicas that
                 never meet are top_null_md_drift and refine_null_md_drift.

Every generator returns (points, truth labels, replica labels or None).
"""
from __future__ import annotations

import numpy as np

D = 6          # dimensions
SEP = 40.0     # top-level separation between groups


def shift(x, k):
    x = x.copy()
    x[:, 0] += k * SEP
    return x


# --- one structureless group of a given shape ----------------------------------

def gauss(rng, n):
    return rng.normal(size=(n, D))


def uniform(rng, n):
    return rng.uniform(-2.5, 2.5, size=(n, D))


def elongated(rng, n):
    return rng.normal(size=(n, D)) * np.r_[5.0, np.ones(D - 1)]


def banana(rng, n):
    t = rng.uniform(-np.pi / 2, np.pi / 2, n)
    x = rng.normal(scale=0.6, size=(n, D))
    x[:, 0] += 8 * np.sin(t)
    x[:, 1] += 8 * np.cos(t)
    return x


def gradient(rng, n):
    x = rng.normal(size=(n, D))
    x[:, 0] = rng.exponential(2.0, n)      # dense at one end, sparse at the other
    return x


def ar1(rng, n, phi=0.99):
    """Continuous drift: an AR(1) walk (correlation time ~100 frames), no states."""
    x = np.zeros((n, D))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(scale=0.5, size=D)
    return x


SHAPES = {"gauss": gauss, "uniform": uniform, "elongated": elongated, "banana": banana,
          "gradient": gradient}


def blobs(rng, k, sep, n=100, scale=(1.0,)):
    s = np.r_[scale, np.ones(D - len(scale))]
    return np.vstack([rng.normal(size=(n, D)) * s + np.r_[i * sep, np.zeros(D - 1)]
                      for i in range(k)])


# --- scenarios -------------------------------------------------------------------

def top_null(kind):
    def make(rng):
        if kind == "md_drift":
            return ar1(rng, 500), np.zeros(500, int), ["A"] * 500
        return SHAPES[kind](rng, 300), np.zeros(300, int), None
    return make


def top_pos(k, sep, aniso=False):
    def make(rng):
        x = blobs(rng, k, sep, scale=(1.0, 5.0) if aniso else (1.0,))
        return x, np.repeat(np.arange(k), 100), None
    return make


def refine_null(kind):
    def make(rng):
        if kind == "dominant":
            a = rng.normal(scale=0.5, size=(80, D))
            b = shift(rng.normal(scale=2.0, size=(300, D)), 1)
            return np.vstack([a, b]), np.r_[np.zeros(80, int), np.ones(300, int)], None
        if kind == "md_drift":
            a, b = ar1(rng, 500), shift(ar1(rng, 500), 1)
            return np.vstack([a, b]), np.repeat([0, 1], 500), ["A"] * 500 + ["B"] * 500
        f = SHAPES[kind]
        x = np.vstack([f(rng, 150), shift(f(rng, 150), 1)])
        return x, np.repeat([0, 1], 150), None
    return make


def refine_pos(sub_sep):
    def make(rng):
        inner = np.vstack([rng.normal(size=(60, D)) + np.r_[s, np.zeros(D - 1)]
                           for s in (0, sub_sep, 2 * sub_sep)])
        far = shift(rng.normal(size=(150, D)), 2)
        return np.vstack([inner, far]), np.r_[np.repeat([0, 1, 2], 60), np.full(150, 3)], None
    return make


# median distance of a standard-normal point in D dimensions to the centre:
# the state scale of a unit Gaussian group (~2.35 for D = 6)
STATE_SCALE = float(np.sqrt(D - 2 / 3))


def refine_scale(f, sub_sd=0.25, n=100):
    def make(rng):
        sep = f * STATE_SCALE
        inner = np.vstack([rng.normal(scale=sub_sd, size=(n, D)) + np.r_[s, np.zeros(D - 1)]
                           for s in (0.0, sep)])
        other = shift(rng.normal(size=(150, D)), 1)
        return (np.vstack([inner, other]), np.r_[np.repeat([0, 1], n), np.full(150, 2)], None)
    return make


SCENARIOS = {}
for _k in ("gauss", "uniform", "elongated", "banana", "gradient", "md_drift"):
    SCENARIOS[f"top_null_{_k}"] = top_null(_k)
for _k, _s in ((2, 6), (3, 4), (3, 5), (3, 6)):
    SCENARIOS[f"top_pos_{_k}groups_{_s}sigma"] = top_pos(_k, _s)
SCENARIOS["top_pos_3elongated_6sigma"] = top_pos(3, 6, aniso=True)
for _k in ("gauss", "uniform", "elongated", "banana", "gradient", "dominant", "md_drift"):
    SCENARIOS[f"refine_null_{_k}"] = refine_null(_k)
for _s in (4, 5, 6, 8):
    SCENARIOS[f"refine_pos_{_s}sigma"] = refine_pos(_s)
for _f in ("0.5", "1.5", "3"):
    SCENARIOS[f"refine_scale_{_f}x"] = refine_scale(float(_f))


def top_null_copies(n_entries=40, copies=6, copy_sd=0.05):
    def make(rng):
        centres = rng.normal(size=(n_entries, D))
        x = np.repeat(centres, copies, axis=0) + rng.normal(scale=copy_sd,
                                                            size=(n_entries * copies, D))
        return x, np.zeros(n_entries * copies, int), None
    return make


SCENARIOS["top_null_copies"] = top_null_copies()


def top_pos_unequal(sizes, sep=6.0):
    def make(rng):
        x = np.vstack([rng.normal(size=(n, D)) + np.r_[i * sep, np.zeros(D - 1)]
                       for i, n in enumerate(sizes)])
        return x, np.repeat(np.arange(len(sizes)), sizes), None
    return make


def top_pos_equidistant(k, sep=6.0, n=50):
    """k groups (k <= D) on the axes, sep / sqrt(2) from the origin: every pair sep apart."""
    def make(rng):
        centres = np.eye(D)[:k] * sep / np.sqrt(2)
        x = np.vstack([rng.normal(size=(n, D)) + c for c in centres])
        return x, np.repeat(np.arange(k), n), None
    return make


def markov(rng, n=1000, p_out=(0.02, 0.02), sep=6.0, start=None, phi=0.9):
    """Two states sep apart along the first axis, leaving state s with probability
    p_out[s] per frame, plus AR(1) noise (unit variance, correlation phi per frame)."""
    s = np.empty(n, int)
    s[0] = (rng.random() < 0.5) if start is None else start
    u = rng.random(n)
    for t in range(1, n):
        s[t] = s[t - 1] ^ int(u[t] < p_out[s[t - 1]])
    e = np.zeros((n, D))
    z = rng.normal(size=(n, D))
    e[0] = z[0]
    for t in range(1, n):
        e[t] = phi * e[t - 1] + np.sqrt(1 - phi ** 2) * z[t]
    x = e.copy()
    x[:, 0] += sep * s
    return x, s


def traj_markov(p_out):
    def make(rng):
        x, s = markov(rng, p_out=p_out)
        return x, s, ["A"] * len(x)
    return make


def traj_markov_replicas(rng):
    a, sa = markov(rng, 500, start=0)
    b, sb = markov(rng, 500, start=1)
    return np.vstack([a, b]), np.r_[sa, sb], ["A"] * 500 + ["B"] * 500


def traj_null_replicas_meet(rng):
    a, b = ar1(rng, 500), ar1(rng, 500)
    return np.vstack([a, b]), np.zeros(1000, int), ["A"] * 500 + ["B"] * 500


for _a, _b in ((240, 60), (270, 30)):
    SCENARIOS[f"top_pos_2groups_6sigma_{_a}_{_b}"] = top_pos_unequal((_a, _b))
for _k in (4, 6):
    SCENARIOS[f"top_pos_{_k}equidistant_6sigma"] = top_pos_equidistant(_k)
SCENARIOS["traj_null_replicas_meet"] = traj_null_replicas_meet
for _name, _p in (("slow", 0.005), ("medium", 0.02), ("fast", 0.1)):
    SCENARIOS[f"traj_pos_markov_{_name}"] = traj_markov((_p, _p))
SCENARIOS["traj_pos_markov_unequal"] = traj_markov((0.01, 0.04))
SCENARIOS["traj_pos_markov_replicas"] = traj_markov_replicas

# scenarios with replica labels: run as trajectories, and also with input.membership_blocks off
TRAJECTORY_SCENARIOS = ["top_null_md_drift", "refine_null_md_drift", "traj_null_replicas_meet",
                        "traj_pos_markov_slow", "traj_pos_markov_medium",
                        "traj_pos_markov_fast", "traj_pos_markov_unequal",
                        "traj_pos_markov_replicas"]


# --- held-out trajectory scenarios --------------------------------------------------
# Not used while the second pass (input.membership_blocks) was designed; they test it on
# new kinds of drift and of states. Acceptance criteria were written down before the runs.

def ar1_axes(rng, n, phis):
    """AR(1) walk with a correlation phi per axis (unit stationary variance)."""
    phis = np.asarray(phis, float)
    x = np.zeros((n, D))
    x[0] = rng.normal(size=D)
    scale = np.sqrt(1 - phis ** 2)
    for i in range(1, n):
        x[i] = phis * x[i - 1] + scale * rng.normal(size=D)
    return x


def heldout_drift(phi, n):
    def make(rng):
        return ar1_axes(rng, n, [phi] * D) * 3.0, np.zeros(n, int), ["A"] * n
    return make


def heldout_anisotropic(rng, n=1000):
    x = ar1_axes(rng, n, [0.999] + [0.9] * (D - 1))
    x[:, 0] *= 5.0
    return x, np.zeros(n, int), ["A"] * n


def heldout_curved(rng, n=1000):
    """A slow walk along a curved path (an arc of radius 8) with noise: one curved
    continuum, no states."""
    t = np.zeros(n)
    for i in range(1, n):
        t[i] = np.clip(t[i - 1] + rng.normal(scale=0.03), -np.pi / 2, np.pi / 2)
    x = ar1_axes(rng, n, [0.9] * D) * 0.6
    x[:, 0] += 8 * np.sin(t)
    x[:, 1] += 8 * np.cos(t)
    return x, np.zeros(n, int), ["A"] * n


def heldout_markov3(rng, n=1500, p=0.02, sep=6.0):
    s = np.empty(n, int)
    s[0] = rng.integers(3)
    for t in range(1, n):
        s[t] = (s[t - 1] + rng.integers(1, 3)) % 3 if rng.random() < p else s[t - 1]
    centres = np.eye(D)[:3] * sep / np.sqrt(2)
    return ar1_axes(rng, n, [0.9] * D) + centres[s], s, ["A"] * n


def heldout_markov_intermediate(rng, n=1000, p=0.02, sep=6.0, path=10):
    """Two states; every switch passes through `path` frames on the line between them
    (labelled by the nearer state)."""
    pos = np.empty(n)
    state, i = int(rng.integers(2)), 0
    while i < n:
        if i and rng.random() < p:
            target = 1 - state
            for j in range(1, path + 1):
                if i >= n:
                    break
                pos[i] = state + (target - state) * j / (path + 1)
                i += 1
            state = target
            continue
        pos[i] = state
        i += 1
    x = ar1_axes(rng, n, [0.9] * D)
    x[:, 0] += sep * pos
    return x, (pos >= 0.5).astype(int), ["A"] * n


def heldout_markov_close(rng):
    x, s = markov(rng, sep=4.0)
    return x, s, ["A"] * len(x)


HELDOUT = {
    "traj_heldout_null_drift_fast": heldout_drift(0.95, 1000),
    "traj_heldout_null_drift_slow": heldout_drift(0.995, 2000),
    "traj_heldout_null_anisotropic": heldout_anisotropic,
    "traj_heldout_null_curved": heldout_curved,
    "traj_heldout_pos_markov3": heldout_markov3,
    "traj_heldout_pos_markov_intermediate": heldout_markov_intermediate,
    "traj_heldout_pos_markov_close": heldout_markov_close,
}
SCENARIOS.update(HELDOUT)
HELDOUT_SCENARIOS = list(HELDOUT)
