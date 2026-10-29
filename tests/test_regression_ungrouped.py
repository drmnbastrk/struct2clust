"""Without input.groups, results are those of the version before resampling by group.

The reference (data/regression_ungrouped.json) was written by running this file as a
script with the version before input.groups was added (source SHA-256 f5a6271a…):

    python tests/test_regression_ungrouped.py

It holds the selected parameters, the labels before and after refinement, the
stability estimates, the refinement log and the conclusion of a full clustering of a
small synthetic ensemble (a far-away group and three sub-groups hidden by it), so the
selection, stability and refinement paths are all compared.
"""
import json
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).parent / "data" / "regression_ungrouped.json"
REFINEMENT_KEYS = ("cluster", "size", "clusterability", "split", "n_subclusters", "reason")


def _points():
    rng = np.random.default_rng(11)
    inner = [rng.normal(np.r_[6.0 * i, np.zeros(4)], 1.0, (40, 5)) for i in range(3)]
    far = rng.normal(np.r_[80.0, np.zeros(4)], 1.0, (60, 5))
    return np.vstack(inner + [far])


def _result() -> dict:
    import logging
    import warnings

    from scipy.spatial.distance import cdist

    from struct2clust.config import Config
    from struct2clust.pipeline import Run, cluster, run_conclusion
    from struct2clust.structures import Ensemble

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    x = _points()
    n = len(x)
    ens = Ensemble(names=[str(i) for i in range(n)], sources=[None] * n, atoms=None,
                   coords=np.zeros((n, 1, 3)), replica=None, frame=None)
    cfg = Config.model_validate({"seed": 7, "n_jobs": 1, "biology": {"enabled": False},
                                 "selection": {"max_resamples": 300},
                                 "stability": {"n_resamples": 50}})
    run = Run(config=cfg, out_dir=None, ensemble=ens, distances=cdist(x, x),
              metric_label="", distance_unit="", n_input=n)
    cluster(run)
    return {
        "params": [int(v) if not isinstance(v, str) else v for v in run.selection.params],
        "top_level": run.groups.tolist(),
        "labels": run.labels.tolist(),
        "stability": json.loads(run.stability.clusters.round(10).to_json(orient="records")),
        "refinement": [{k: e.get(k) for k in REFINEMENT_KEYS} for e in run.refinement],
        "conclusion": run_conclusion(run)[0],
    }


def test_without_groups_results_equal_the_version_before_groups():
    reference = json.loads(REFERENCE.read_text())
    assert _result() == reference


if __name__ == "__main__":
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(_result(), indent=1))
    print(f"written {REFERENCE}")
