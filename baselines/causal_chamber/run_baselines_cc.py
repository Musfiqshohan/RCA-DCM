"""Run the 5 kept RCAEval baseline methods (nsigma, baro, rcd, circa, rcg)
against lt_interventions_standard_v1 (light tunnel, standard configuration),
using the REAL published causal graph (ground_truth.py) for circa/rcg -- not
a sink graph, not PC-discovery.

Reuses RCAEval/run_baselines.py's to_rca_input()/run_one()/METHOD_SPECS/
GRAPH_METHODS directly (generic, not microservice-specific) rather than
reimplementing them.

No fault-type loop, no time-window split, no replicates: each case is just
(uniform_reference.csv, one uniform_X.csv) -- see config.py.
"""
import argparse
import os
import sys
import time

import pandas as pd
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines", "RCAEval"))

import config as cc  # noqa: E402
import ground_truth as gt  # noqa: E402
# This file is named run_baselines_cc.py (not run_baselines.py) specifically
# so this import can't collide with baselines/RCAEval/run_baselines.py --
# a same-named file here would shadow it and break both the import itself
# and, more subtly, multiprocessing's ability to pickle/re-import _mp_worker
# by module path in each spawned subprocess.
from run_baselines import to_rca_input, run_one, METHOD_SPECS, GRAPH_METHODS  # noqa: E402


def real_graph_nx():
    """The 38-node/57-edge ground-truth DAG as an nx.DiGraph, already in the
    causal (parent -> child) direction circa/rcg's `graph=` expects -- NOT
    reversed like the microservice call graph was (that graph was caller ->
    callee and needed inverting to fault-propagation direction; this one is
    already cause -> effect)."""
    G = nx.DiGraph()
    G.add_nodes_from(gt.VARIABLES)
    G.add_edges_from(gt.EDGES)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "baselines_results.csv"))
    args = ap.parse_args()

    graph = real_graph_nx()
    cases = cc.cases()
    print(f"Running {len(METHOD_SPECS)} methods on {len(cases)} causal-chamber cases: {list(METHOD_SPECS)}", flush=True)

    rows = []
    t_start = time.time()
    for fname, true_col in cases:
        nrm, anm = cc.load_case(fname)
        data, inject_time = to_rca_input(nrm, anm)
        num_node = len(gt.VARIABLES)

        row = {"case": fname, "true": true_col}
        case_summary = []
        for name, (modpath, fn) in METHOD_SPECS.items():
            g = graph if name in GRAPH_METHODS else None
            ranks, error, seconds = run_one(modpath, fn, data, inject_time, num_node,
                                              entry_node=None, timeout_s=args.timeout, graph=g)
            row[f"{name}_seconds"] = seconds
            row[f"{name}_error"] = error or ""
            for k in (1, 3, 5):
                row[f"{name}_acc@{k}"] = cc.acc_at_k(ranks, true_col, k) if ranks else 0
                row[f"{name}_AC@{k}"] = cc.AC_at_k(ranks, true_col, k) if ranks else 0.0
                row[f"{name}_Avg@{k}"] = cc.Avg_at_k(ranks, true_col, k) if ranks else 0.0
            row[f"{name}_PRR"] = cc.exact_match_prr(ranks, true_col) if ranks else 0.0
            row[f"{name}_top1"] = ranks[0] if ranks else ""
            row[f"{name}_top5"] = ",".join(ranks[:5]) if ranks else ""
            case_summary.append(f"{name}={'OK' if not error else 'ERR'}({row[f'{name}_acc@1']},{seconds:.0f}s)")
        rows.append(row)
        print(f"  {fname} (true={true_col}): {' '.join(case_summary)}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\n=== causal_chamber summary (n={len(df)} cases, {(time.time()-t_start)/60:.1f} min) ===")
    print(f"{'method':<10}{'Acc@1':>8}{'Acc@3':>8}{'Acc@5':>8}{'AC@1':>8}{'AC@3':>8}{'AC@5':>8}"
          f"{'Avg@5':>8}{'PRR':>8}{'n_err':>8}")
    for name in METHOD_SPECS:
        n_err = (df[f"{name}_error"] != "").sum()
        print(f"{name:<10}{df[f'{name}_acc@1'].mean():>8.3f}{df[f'{name}_acc@3'].mean():>8.3f}"
              f"{df[f'{name}_acc@5'].mean():>8.3f}{df[f'{name}_AC@1'].mean():>8.3f}"
              f"{df[f'{name}_AC@3'].mean():>8.3f}{df[f'{name}_AC@5'].mean():>8.3f}"
              f"{df[f'{name}_Avg@5'].mean():>8.3f}{df[f'{name}_PRR'].mean():>8.3f}{n_err:>8}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
