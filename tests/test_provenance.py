"""Provenance of a run: the source hash in run_summary.json identifies the code."""
import hashlib
from pathlib import Path

from struct2clust import output


def test_source_hash_covers_every_python_file_of_the_package():
    pkg = Path(output.__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(pkg.glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    assert output.source_sha256() == h.hexdigest()
    assert len(output.source_sha256()) == 64


def test_versions_report_the_source_hash():
    v = output._versions()
    assert v["struct2clust_source_sha256"] == output.source_sha256()
