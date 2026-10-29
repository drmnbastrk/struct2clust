"""Write the source of every structure of the example runs, for struct2clust composition.

    python example/source_tables.py        (after the runs, see README.md)

  tar             PDB entry of each NMR model
  eralpha         partner bound in the PDB entry: coactivator peptide (possible only
                  with helix 12 in the agonist position), antagonist (4-hydroxytamoxifen
                  OHT or raloxifene RAL; helix 12 in the antagonist position), or neither
  endothiapepsin  space group of the crystal
  1RKJ_rna        simulation: RNA in the complex, or free RNA

Writes data/<run>_sources.csv with the columns structure, source.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
COACTIVATOR = (r"coactivator|NCOA|GRIP|glucocorticoid receptor.interacting"
               r"|intermediary factor|SRC|receptor-interacting|stapled")
ANTAGONISTS = {"OHT", "RAL"}


def structures(run: str) -> pd.DataFrame:
    """Every input structure of a run: the clustered ones and the near-duplicates
    removed before clustering, so the table does not depend on which copy was kept."""
    names = [pd.read_csv(RESULTS / run / "clusters.csv", usecols=["structure"])]
    if (RESULTS / run / "duplicates.csv").exists():
        names.append(pd.read_csv(RESULTS / run / "duplicates.csv", usecols=["structure"]))
    return pd.concat(names, ignore_index=True)


def main() -> None:
    tar = structures("tar")
    tar["source"] = tar["structure"].str[:4]

    meta = pd.read_csv(DATA / "eralpha_meta.csv", index_col=0)
    era = structures("eralpha")
    m = meta.reindex(era["structure"])
    coact = m["other_polymers"].fillna("").str.contains(COACTIVATOR, case=False).to_numpy()
    anta = m["ligands"].fillna("").str.split().apply(lambda x: bool(ANTAGONISTS & set(x)))
    era["source"] = "neither"
    era.loc[anta.to_numpy(), "source"] = "antagonist (OHT, RAL)"
    era.loc[coact, "source"] = "coactivator peptide"

    meta = pd.read_csv(DATA / "endothiapepsin_meta.csv", index_col=0)
    endo = structures("endothiapepsin")
    endo["source"] = meta.reindex(endo["structure"])["space_group"].to_numpy()

    rkj = structures("1RKJ_rna")
    rkj["source"] = rkj["structure"].str.split(":").str[0].map(
        {"complex": "RNA in the complex", "rna": "free RNA"})

    for run, table in [("tar", tar), ("eralpha", era), ("endothiapepsin", endo),
                       ("1RKJ_rna", rkj)]:
        table.to_csv(DATA / f"{run}_sources.csv", index=False)
        print(f"{run}: {len(table)} structures, {table['source'].nunique()} sources")


if __name__ == "__main__":
    main()
