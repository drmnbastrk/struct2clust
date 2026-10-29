"""Benchmarks on real molecules with a known answer.

    python validation/real_benchmark.py fetch  --data DIR
    python validation/real_benchmark.py alanine --data DIR --jobs 4 [--frames 100 300]
    python validation/real_benchmark.py adk    --data DIR --seeds 20 --jobs 4 [--scenarios ...]
    python validation/real_benchmark.py table

fetch    Downloads the data (checked against SHA-256):
         - alanine dipeptide in water, three 250 ns trajectories saved every 1 ps,
           without water, with their backbone dihedrals (mdshare, FU Berlin;
           Wehmeyer et al. 2019);
         - adenylate kinase of E. coli, open (PDB 4AKE) and closed (PDB 1AKE).
alanine  Clusters the three trajectories together, thinned to 100, 300, 1000 and 2000
         frames each, with default settings. The true state of every frame comes
         from its dihedrals (Ramachandran regions): alpha_R (phi < 0, -120 < psi <
         50), beta / C7eq (phi < 0, other psi), alpha_L (phi > 0). Struct2Clust leaves
         the ACE and NME caps out, so only the five heavy atoms of the alanine are
         compared and phi is not seen; the answer to recover is alpha_R against beta.
adk      Ensembles of 200 structures made from the real heavy-atom structures of the
         open and the closed form (chain A, atoms common to both), each structure
         displaced along the normal modes of an anisotropic network model (Atilgan et
         al. 2001; cutoff 8 A between heavy atoms) with thermal-like amplitudes, scaled
         so that the mean RMSD to the starting structure is 1, 2 or 3 A. Two states:
         open:closed 50:50, 80:20, 90:10 (the forms are 7 A apart). One state: only
         the open or only the closed form. Clustered with default settings.

Results: validation/results/real_alanine_<frames>.csv, real_adk_<scenario>.csv (one row
per run: conclusion, clusters, ARI to the true states, noise); `table` summarises them.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import warnings  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MDSHARE = "https://ftp.imp.fu-berlin.de/pub/cmb-data/"
FILES = {
    "alanine-dipeptide-nowater.pdb":
        "20d4ff8f96df706d42f2efa334ab7facf256d16d5f8ba80175b407c3885bf9f5",
    "alanine-dipeptide-0-250ns-nowater.xtc":
        "ab8c91c8f572fddf4de3b9b78a6dfe56285f5f7dda30d2b124e261fa612e2b29",
    "alanine-dipeptide-1-250ns-nowater.xtc":
        "6a80b5f301ee8c301ef7a7e19559411ca623a9af9a416d8fb70e0ff8f6b94134",
    "alanine-dipeptide-2-250ns-nowater.xtc":
        "635813174dc3bdebe2ee65b99a91830ff54d37c24c0235054476f522244bceb5",
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz":
        "f5ad30768a7cf3451b3061cb2ecb7f7d1de8c13044534376d26a6653d4cd5717",
}
PDB = {"4AKE": "ff798ee8791878eb58bac1c6bed32042f51b455512ac93b50bda8fa0ff0e7f78",
       "1AKE": "651e952f55f1317f50f4e82b7f0e99053397032cd8da5b2896f79d0af9f619a9"}
ALANINE_FRAMES = [100, 300, 1000, 2000]
ADK_SCENARIOS = {          # name: (fraction open, amplitude in A); one form only if 1 or 0
    "two_states_50_50_1A": (0.5, 1.0), "two_states_80_20_1A": (0.8, 1.0),
    "two_states_90_10_1A": (0.9, 1.0), "two_states_50_50_2A": (0.5, 2.0),
    "two_states_80_20_2A": (0.8, 2.0), "two_states_90_10_2A": (0.9, 2.0),
    "open_only_1A": (1.0, 1.0), "open_only_2A": (1.0, 2.0), "open_only_3A": (1.0, 3.0),
    "closed_only_1A": (0.0, 1.0), "closed_only_2A": (0.0, 2.0),
    "closed_only_3A": (0.0, 3.0),
}
N_ADK = 200
ANM_CUTOFF = 8.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    urls = {f: MDSHARE + f for f in FILES}
    urls.update({f"{p}.pdb": f"https://files.rcsb.org/download/{p}.pdb" for p in PDB})
    sums = {**FILES, **{f"{p}.pdb": s for p, s in PDB.items()}}
    for name, url in urls.items():
        path = data / name
        if not path.exists():
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, path)
        if sha256(path) != sums[name]:
            raise SystemExit(f"{path}: SHA-256 does not match; delete it and fetch again")
    print(f"all files present and checked in {data}")


# --- alanine dipeptide -----------------------------------------------------------------

def ramachandran(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """0 = beta / C7eq, 1 = alpha_R, 2 = alpha_L (degrees)."""
    state = np.where((psi > -120) & (psi < 50), 1, 0)
    return np.where(phi > 0, 2, state)


def run_struct2clust(raw: dict, input_path=None):
    from struct2clust.config import Config
    from struct2clust.pipeline import run_conclusion, run_pipeline

    cfg = Config.model_validate(raw)
    with tempfile.TemporaryDirectory() as tmp:
        run = run_pipeline(cfg, input_path, Path(tmp) / "run", write=False)
    return run, run_conclusion(run)[0]


def alanine(data: Path, jobs: int, frames: list[int]) -> None:
    from sklearn.metrics import adjusted_rand_score

    dih = np.load(data / "alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    truth = [ramachandran(*np.degrees(dih[k]).T) for k in sorted(dih.files)]
    for n in frames:
        rows = []
        raw = {"seed": 42, "n_jobs": jobs, "biology": {"enabled": False},
               "figures": {"enabled": False},
               "input": {"type": "trajectory", "max_frames": n,
                         "topology": [str(data / "alanine-dipeptide-nowater.pdb")],
                         "trajectory": [str(data / f"alanine-dipeptide-{i}-250ns-nowater.xtc")
                                        for i in range(3)]}}
        t0 = time.time()
        run, conclusion = run_struct2clust(raw)
        reps = sorted(set(run.ensemble.replica))
        true = np.array([truth[reps.index(r)][f]
                         for r, f in zip(run.ensemble.replica, run.ensemble.frame)])
        ab = true < 2                                   # alpha_R / beta frames
        lab = run.labels
        rows.append({
            "frames_per_trajectory": n, "conclusion": conclusion,
            "n_clusters": int(run.n_clusters), "noise": round(float(np.mean(lab < 0)), 3),
            "ari_alphaR_beta": round(float(adjusted_rand_score(true[ab], lab[ab])), 3),
            "ari_three_states": round(float(adjusted_rand_score(true, lab)), 3),
            "fraction_alphaR": round(float(np.mean(true == 1)), 3),
            "fraction_alphaL": round(float(np.mean(true == 2)), 3),
            "sampling_sufficient": (run.sampling_check or {}).get("sufficient"),
            "second_pass": run.first_pass is not None,
            "seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        RESULTS.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(RESULTS / f"real_alanine_{n}.csv", index=False)


# --- adenylate kinase ------------------------------------------------------------------

def read_chain_a(path: Path) -> dict:
    """{(resid, atom name): (line, xyz)} for the heavy atoms of chain A (first altloc)."""
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[21] != "A" or line[16] not in " A":
            continue
        name = line[12:16].strip()
        if name.startswith("H") or line[76:78].strip() == "H":
            continue
        key = (int(line[22:26]), name)
        if key not in out:
            out[key] = (line, np.array([float(line[30:38]), float(line[38:46]),
                                        float(line[46:54])]))
    return out


def kabsch_fit(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    a, b = mobile - mobile.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return a @ r + target.mean(0)


def anm_modes(xyz: np.ndarray, cutoff: float = ANM_CUTOFF):
    """Eigenvalues and eigenvectors of the anisotropic network model Hessian (unit spring
    constant), without the six rigid-body modes."""
    n = len(xyz)
    diff = xyz[:, None, :] - xyz[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    contact = (d2 < cutoff ** 2) & ~np.eye(n, dtype=bool)
    h = np.zeros((3 * n, 3 * n))
    ii, jj = np.nonzero(contact)
    for i, j in zip(ii, jj):
        block = -np.outer(diff[i, j], diff[i, j]) / d2[i, j]
        h[3 * i:3 * i + 3, 3 * j:3 * j + 3] = block
    for i in range(n):
        h[3 * i:3 * i + 3, 3 * i:3 * i + 3] = -sum(
            h[3 * i:3 * i + 3, 3 * j:3 * j + 3] for j in np.flatnonzero(contact[i]))
    val, vec = np.linalg.eigh(h)
    return val[6:], vec[:, 6:]


def sample(rng, xyz, modes, amplitude: float, n: int) -> np.ndarray:
    """n structures displaced along the modes with variance 1 / eigenvalue each (thermal
    in the network model), scaled to a mean RMSD of `amplitude` to xyz."""
    if n == 0:
        return np.empty((0, *xyz.shape))
    val, vec = modes
    z = rng.normal(size=(n, len(val))) / np.sqrt(val)
    disp = (z @ vec.T).reshape(n, -1, 3)
    rms = np.sqrt(np.mean(np.sum(disp ** 2, axis=2), axis=1)).mean()
    return xyz + disp * (amplitude / rms)


def write_models(path: Path, lines: list[str], models: np.ndarray) -> None:
    with open(path, "w") as fh:
        for k, xyz in enumerate(models, start=1):
            fh.write(f"MODEL     {k:4d}\n")
            for line, (x, y, z) in zip(lines, xyz):
                fh.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}\n")
            fh.write("ENDMDL\n")
        fh.write("END\n")


def adk(data: Path, seeds: int, jobs: int, scenarios: list[str]) -> None:
    from sklearn.metrics import adjusted_rand_score

    open_, closed = read_chain_a(data / "4AKE.pdb"), read_chain_a(data / "1AKE.pdb")
    keys = [k for k in open_ if k in closed]
    lines = [open_[k][0] for k in keys]
    xyz_open = np.array([open_[k][1] for k in keys])
    xyz_closed = kabsch_fit(np.array([closed[k][1] for k in keys]), xyz_open)
    gap = float(np.sqrt(np.mean(np.sum((xyz_open - xyz_closed) ** 2, axis=1))))
    print(f"{len(keys)} common heavy atoms; open-closed RMSD {gap:.2f} A", flush=True)
    modes = {"open": anm_modes(xyz_open), "closed": anm_modes(xyz_closed)}
    for name in scenarios:
        f_open, amp = ADK_SCENARIOS[name]
        out = RESULTS / f"real_adk_{name}.csv"
        rows = pd.read_csv(out).to_dict("records") if out.exists() else []
        done = {(r["scenario"], r["seed"]) for r in rows}
        for seed in range(seeds):
            if (name, seed) in done:
                continue
            rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
            n_open = int(round(N_ADK * f_open))
            models = np.concatenate([
                sample(rng, xyz_open, modes["open"], amp, n_open),
                sample(rng, xyz_closed, modes["closed"], amp, N_ADK - n_open)])
            truth = np.r_[np.zeros(n_open, int), np.ones(N_ADK - n_open, int)]
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                pdb = Path(tmp) / "adk.pdb"
                write_models(pdb, lines, models)
                run, conclusion = run_struct2clust(
                    {"seed": 42, "n_jobs": jobs, "biology": {"enabled": False},
                     "figures": {"enabled": False}}, pdb)
            kept = np.array([int(s.rsplit("_m", 1)[1]) - 1 for s in run.ensemble.names])
            rows.append({
                "scenario": name, "seed": seed, "open_fraction": f_open, "amplitude": amp,
                "conclusion": conclusion, "n_clusters": int(run.n_clusters),
                "noise": round(float(np.mean(run.labels < 0)), 3),
                "ari": round(float(adjusted_rand_score(truth[kept], run.labels)), 3),
                "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
            RESULTS.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)


def table() -> None:
    for name in ("real_alanine", "real_adk"):
        parts = sorted(RESULTS.glob(f"{name}_*.csv"))
        if not parts:
            continue
        df = pd.concat([pd.read_csv(p) for p in parts])
        df.to_csv(RESULTS / f"{name}.csv", index=False)
        print(f"== {name}")
        if "scenario" in df:
            print(pd.crosstab(df["scenario"], df["conclusion"]).to_string())
            print(df.groupby("scenario")[["n_clusters", "ari", "noise"]].mean().round(2)
                  .to_string())
        else:
            print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["fetch", "alanine", "adk", "table"])
    ap.add_argument("--data", type=Path, default=HERE / "real_data")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--frames", type=int, nargs="*", default=ALANINE_FRAMES)
    ap.add_argument("--scenarios", nargs="*", default=list(ADK_SCENARIOS))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    if args.what == "fetch":
        fetch(args.data)
    elif args.what == "alanine":
        alanine(args.data, args.jobs, args.frames)
    elif args.what == "adk":
        adk(args.data, args.seeds, args.jobs, args.scenarios)
    else:
        table()


if __name__ == "__main__":
    main()
