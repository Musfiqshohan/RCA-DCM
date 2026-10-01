"""Data loading + real-graph construction for lt_interventions_standard_v1
(the "light tunnel, standard configuration" causal chamber dataset).

The case list lives in causal_chamber_datasets.yaml next to this file.
CSV files are not shipped with the code. Set CAUSAL_CHAMBER_DATA to the
dataset's data directory (the folder that contains the intervention CSVs).
"""
import os
import yaml
import pandas as pd

import ground_truth as gt

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_YAML = os.path.join(_HERE, "causal_chamber_datasets.yaml")
DATA_DIR = os.environ.get(
    "CAUSAL_CHAMBER_DATA",
    os.path.join(_HERE, "data", "lt_interventions_standard_v1"),
)
DATASET_KEY = "lt_interventions_standard_v1"


def _load_dataset_entry():
    with open(DATASETS_YAML) as f:
        cfg = yaml.safe_load(f)
    return cfg["datasets"][DATASET_KEY]


def cases():
    """[(anomalous_filename, true_root_cause_column)], one per intervention
    experiment. Root cause resolved from the yaml's LaTeX label back to the
    raw column name via ground_truth.COLUMN_FROM_LATEX."""
    entry = _load_dataset_entry()
    out = []
    for fname, latex_label in entry["root_causes"].items():
        col = gt.COLUMN_FROM_LATEX.get(latex_label)
        if col is None:
            raise KeyError(f"{fname}: no column found for latex label {latex_label!r}")
        out.append((fname, col))
    return sorted(out)


def normal_file():
    entry = _load_dataset_entry()
    assert len(entry["normal_files"]) == 1, entry["normal_files"]
    return entry["normal_files"][0]


def _read(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    return df[gt.VARIABLES]  # drop admin columns, fix column order


def load_case(anomalous_fname):
    """(normal_df, anomalous_df), both restricted to the 38 real variables,
    admin columns dropped. No time-window split needed -- normal/anomalous
    are already two separate files."""
    nrm = _read(normal_file())
    anm = _read(anomalous_fname)
    return nrm, anm


def acc_at_k(ranked, true_col, k):
    """Top-k accuracy: 1 if true_col in ranked[:k], else 0."""
    return int(true_col in ranked[:k])


def AC_at_k(ranked, true_causes, k):
    """ASE'24-style AC@k (per instance, real-valued, partial credit):
        AC@k = sum_{i<=k} 1[R[i] in V_rc] / min(k, |V_rc|)
    true_causes: a single node name or a collection (V_rc). Here (single
    accepted answer) this is mathematically identical to acc_at_k -- both
    reduce to "is the one true cause in the top-k" -- kept as a separate
    function so the formula is explicit and this file's Exp-1 usage lines
    up with the multi-label Exp-2 version in hidden_config.py."""
    V_rc = {true_causes} if isinstance(true_causes, str) else set(true_causes)
    if not V_rc:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in V_rc)
    return hits / min(k, len(V_rc))


def Avg_at_k(ranked, true_causes, k):
    """Avg@k = (1/k) * sum_{j=1..k} AC@j, per instance."""
    return sum(AC_at_k(ranked, true_causes, j) for j in range(1, k + 1)) / k


def exact_match_prr(ranked, true_causes):
    """PRR per the paper's definition (same as run_sink_graph.exact_match_prr,
    reimplemented locally to keep this folder self-contained rather than
    reaching across a path that isn't guaranteed set up when this module is
    imported standalone): 1 if the top-|V_rc| predictions equal V_rc
    exactly (as a set), else 0. NOT swept over k=1/3/5 -- k is fixed to
    |V_rc| by the formula itself, one value per instance."""
    V_rc = {true_causes} if isinstance(true_causes, str) else set(true_causes)
    k = len(V_rc)
    if k == 0:
        return 0.0
    return 1.0 if set(ranked[:k]) == V_rc else 0.0


def write_causal_graph(path, name="lt_standard_real"):
    """Write the real 38-node/57-edge ADMG to a graphs.yaml-shaped file, same
    convention run_sink_graph.write_node_graph() uses for the (edge-free)
    sink graph -- here with the actual ground-truth parent lists instead."""
    entry = {"type": "admg", "dag": gt.dag(), "confounders": {},
             "dim_dict": {n: 1 for n in gt.VARIABLES}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"graphs": {name: entry}}, f, default_flow_style=False, sort_keys=False)
    return name
