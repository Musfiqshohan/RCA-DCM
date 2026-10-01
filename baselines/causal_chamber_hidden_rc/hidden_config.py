"""Experiment 2: the true intervened variable is treated as UNOBSERVED.

For each case, the true root-cause column is dropped entirely from the data
AND from the graph (it's not just unlabeled -- it's not there to be ranked
at all). Since the true cause is invisible, "correct" is redefined as: did
the method's top-k point at one of the true cause's DIRECT CHILDREN in the
real causal graph (ground_truth.py) -- the observable symptoms closest to
the hidden cause. This is a genuinely harder, differently-shaped task, not
a relabeling of Experiment 1.

Reuses baselines/causal_chamber/{ground_truth,config}.py for the base 38-node
graph and raw data loading (single source of truth for the real DAG) --
imported by explicit file path, not by module name, to avoid colliding with
this experiment's own hidden_config.py / ground_truth-derived logic.
"""
import os
import sys
import yaml
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CC_DIR = os.path.abspath(os.path.join(HERE, "..", "causal_chamber"))
sys.path.insert(0, CC_DIR)
import ground_truth as gt  # noqa: E402  (baselines/causal_chamber/ground_truth.py)
import config as base_cc  # noqa: E402  (baselines/causal_chamber/config.py)

DATA_DIR = os.path.join(HERE, "data")

_CHILDREN = {}
for _f, _t in gt.EDGES:
    _CHILDREN.setdefault(_f, set()).add(_t)


def accepted_root_causes(true_col):
    """The evaluation target for a case whose true cause is true_col: its
    direct children in the real graph -- the observable nodes one hop
    downstream of the (now-hidden) true cause. Every source node has >=1
    child (verified 2026-08-19: zero source nodes have an empty child set),
    so this is never empty."""
    return sorted(_CHILDREN.get(true_col, set()))


def hidden_dag(true_col):
    """The 37-node dag with true_col removed as a node AND as a parent
    reference everywhere it appears (its children lose that one parent
    entry; they keep whatever other parents they have)."""
    full = gt.dag()
    return {n: [p for p in parents if p != true_col]
            for n, parents in full.items() if n != true_col}


def hidden_confounders(true_col):
    """true_col is not just dropped -- it becomes an unobserved confounder
    (ADMG bidirected-edge group) among its own former children, keyed by its
    own name. This is what "hidden" means here: DCM should see these
    children as mutually confounded by a latent it never observes, not as
    independent. Requires dcm/functions/models.py's N-way confounder-group
    fix (2026-08-19 -- the code previously only supported exactly 2 members
    per group via a hardcoded v1,v2=pair unpack)."""
    kids = accepted_root_causes(true_col)
    return {true_col: kids} if kids else {}


def save_hidden_dataset(true_col):
    """Write the reduced (true_col dropped) normal file + graph.yaml for
    this hidden column, once per unique column (shared across every case
    with this true cause, e.g. blue_mid/blue_strong both hide "blue").
    Returns (case_dir, graph_name, graph_yml_path)."""
    case_dir = os.path.join(DATA_DIR, true_col)
    os.makedirs(case_dir, exist_ok=True)

    keep = [v for v in gt.VARIABLES if v != true_col]
    nrm_full = pd.read_csv(os.path.join(base_cc.DATA_DIR, base_cc.normal_file()))[keep]
    nrm_path = os.path.join(case_dir, "reference_hidden.csv")
    nrm_full.to_csv(nrm_path, index=False)

    graph_name = f"lt_standard_hidden_{true_col}"
    graph_yml = os.path.join(case_dir, "graph.yaml")
    entry = {"type": "admg", "dag": hidden_dag(true_col),
             "confounders": hidden_confounders(true_col),
             "dim_dict": {n: 1 for n in keep}}
    with open(graph_yml, "w") as f:
        yaml.dump({"graphs": {graph_name: entry}}, f, default_flow_style=False, sort_keys=False)

    return case_dir, graph_name, graph_yml, nrm_path


def save_hidden_anomalous(fname, true_col):
    """Write the reduced (true_col dropped) anomalous file for one case,
    alongside its column's shared reference_hidden.csv/graph.yaml."""
    case_dir = os.path.join(DATA_DIR, true_col)
    os.makedirs(case_dir, exist_ok=True)
    keep = [v for v in gt.VARIABLES if v != true_col]
    anm_full = pd.read_csv(os.path.join(base_cc.DATA_DIR, fname))[keep]
    anm_path = os.path.join(case_dir, fname)
    anm_full.to_csv(anm_path, index=False)
    return anm_path


def cases():
    """[(anomalous_filename, true_col, accepted_root_causes_list)]."""
    return [(fname, true_col, accepted_root_causes(true_col))
            for fname, true_col in base_cc.cases()]


def build_all_hidden_datasets():
    """Materialize every hidden dataset + graph up front (one per unique
    true_col, ~26 of them) -- call once before running baselines/DCM."""
    done = {}
    for fname, true_col, _ in cases():
        if true_col not in done:
            done[true_col] = save_hidden_dataset(true_col)
        save_hidden_anomalous(fname, true_col)
    return done


def load_hidden_case(fname, true_col):
    """(normal_df, anomalous_df) for one case, columns = the 37 kept vars,
    read back from the saved files in data/."""
    case_dir = os.path.join(DATA_DIR, true_col)
    keep = [v for v in gt.VARIABLES if v != true_col]
    nrm = pd.read_csv(os.path.join(case_dir, "reference_hidden.csv"))[keep]
    anm = pd.read_csv(os.path.join(case_dir, fname))[keep]
    return nrm, anm


def acc_at_k(ranked, accepted_set, k):
    """Multi-label top-k accuracy: 1 if ANY accepted root cause appears in
    the top-k of `ranked`, else 0. Generalizes this project's existing
    single-label acc@k (int(true in ranks[:k])) to a set of acceptable
    answers -- same "at least one hit in top-k" semantics, just checked
    against a set instead of one value. NOT ASE'24's partial-credit AC@k
    (which divides by min(k,|V_rc|) and counts every hit, not just whether
    there was one) -- documented explicitly since the task requires stating
    the convention, not just picking one silently."""
    if not accepted_set:
        return 0
    return int(any(r in accepted_set for r in ranked[:k]))


def AC_at_k(ranked, accepted_set, k):
    """ASE'24-style AC@k (per instance, real-valued, partial credit) --
    supersedes the earlier prr_at_k (subset-containment, binary, swept over
    k=1/3/5) per the exact formula given for the multi-root-cause setup:
        AC@k = sum_{i<=k} 1[R[i] in V_rc] / min(k, |V_rc|)
    Counts EVERY hit in the top-k (not just whether there was at least
    one, like acc_at_k), normalized by min(k, |V_rc|) so a method that
    finds all of a small accepted set early scores 1.0 exactly."""
    if not accepted_set:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in accepted_set)
    return hits / min(k, len(accepted_set))


def Avg_at_k(ranked, accepted_set, k):
    """Avg@k = (1/k) * sum_{j=1..k} AC@j, per instance."""
    return sum(AC_at_k(ranked, accepted_set, j) for j in range(1, k + 1)) / k


def exact_match_prr(ranked, accepted_set):
    """PRR per the paper's definition: 1 if the top-|accepted_set|
    predictions equal accepted_set exactly (as a set), else 0. Fixed
    k = |accepted_set| by the formula itself -- not swept over k=1/3/5."""
    k = len(accepted_set)
    if k == 0:
        return 0.0
    return 1.0 if set(ranked[:k]) == set(accepted_set) else 0.0
