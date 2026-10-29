"""Memory needed by a run, and how many parallel workers fit into it.

The distance matrix of N structures holds N^2 float64 values (8 N^2 bytes). The main
process holds about eight such matrices at its peak (the matrix, clusterability, the
map and its trustworthiness), and every resampling worker about two more (a subsample
of the matrix and HDBSCAN's copy of it), measured at N = 3000. The peak is therefore
about (8 + 2 n_jobs) * 8 N^2 bytes: for 10,000 structures about 6.4 GB plus 1.6 GB per
worker. Results do not depend on the number of workers.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

MAIN_MATRICES = 8      # N x N float64 matrices held by the main process at its peak
WORKER_MATRICES = 2    # ... and by every resampling worker


def cpu_count() -> int:
    """Cores this process may use: the CPU affinity where the system reports it (e.g. the
    cores a cluster job was given), else every core of the machine."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:          # not available on Windows and macOS
        return os.cpu_count() or 1


def peak_memory(n: int, n_jobs: int) -> int:
    """Approximate peak memory of a run in bytes."""
    return (MAIN_MATRICES + WORKER_MATRICES * n_jobs) * 8 * n * n


def _cgroup_room() -> Optional[int]:
    """Bytes left below the memory limit of this process's control group (v2 or v1), or
    None without a limit."""
    for limit_file, usage_file in (("memory.max", "memory.current"),
                                   ("memory/memory.limit_in_bytes",
                                    "memory/memory.usage_in_bytes")):
        try:
            with open(f"/sys/fs/cgroup/{limit_file}") as fh:
                limit = fh.read().strip()
            with open(f"/sys/fs/cgroup/{usage_file}") as fh:
                usage = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if limit.isdigit() and int(limit) < 1 << 60:      # "max" or a huge value: no limit
            return max(0, int(limit) - usage)
    return None


def available_memory() -> Optional[int]:
    """Memory available to new processes in bytes (Linux, Windows), or None if unknown.
    On Linux, the limit of the process's control group counts too (e.g. the memory a
    cluster job was given)."""
    try:
        if sys.platform.startswith("linux"):
            free = None
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        free = int(line.split()[1]) * 1024
            limit = _cgroup_room()
            return limit if free is None else free if limit is None else min(free, limit)
        elif sys.platform == "win32":
            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "total_phys", "avail_phys", "total_page", "avail_page",
                        "total_virtual", "avail_virtual", "avail_extended")]
            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.avail_phys)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def worker_count(n_jobs: Optional[int], n: int, available: Optional[int]) -> int:
    """Resampling workers: ``n_jobs`` as given (-1 or 0: all cores), or for None all cores
    but no more than ``available`` bytes hold beside the main process (at least one)."""
    cpus = cpu_count()
    if n_jobs is not None:
        return cpus if n_jobs in (-1, 0) else max(1, min(n_jobs, cpus))
    if available is None:
        return cpus
    fit = (available - peak_memory(n, 0)) // (WORKER_MATRICES * 8 * n * n)
    return int(max(1, min(cpus, fit)))
