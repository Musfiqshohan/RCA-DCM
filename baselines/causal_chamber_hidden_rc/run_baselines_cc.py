"""Experiment 2: 5 baselines (nsigma, baro, rcd, circa, rcg) against the
hidden-true-cause datasets (hidden_config.py) -- true cause column removed
from data and graph; success = top-k contains ANY of its direct children
(hidden_config.acc_at_k). Real graph (37 nodes, case-specific: whichever
column is hidden this case), not sink graph. See ../causal_chamber/
run_baselines_cc.py for the non-hidden counterpart this is adapted from.
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

import hidden_config as hc  # noqa: E402
# Named run_baselines.py in RCAEval/, deliberately not duplicated here
# (see that file + causal_chamber/run_baselines_cc.py for why this file is
# NOT itself called run_baselines.py -- same-basename collision risk).
from run_baselines import to_rca_input, run_one, METHOD_SPECS, GRAPH_METHODS  # noqa: E402


def graph_for(true_col):
    G = nx.DiGraph()
    dag = hc.hidden_dag(true_col)
    G.add_nodes_from(dag.keys())
    for node, parents in dag.items():
        for p in parents:
            G.add_edge(p, node)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "baselines_results.csv"))
    args = ap.parse_args()

    hc.build_all_hidden_datasets()
    cases = hc.cases()
    print(f"Running {len(METHOD_SPECS)} methods on {len(cases)} hidden-root-cause cases: {list(METHOD_SPECS)}", flush=True)

    rows = []
    t_start = time.time()
    for fname, true_col, accepted in cases:
        nrm, anm = hc.load_hidden_case(fname, true_col)
        data, inject_time = to_rca_input(nrm, anm)
        num_node = nrm.shape[1]
        graph = graph_for(true_col)

        row = {"case": fname, "true_hidden": true_col, "accepted_rc": ",".join(accepted)}
        case_summary = []
        for name, (modpath, fn) in METHOD_SPECS.items():
            g = graph if name in GRAPH_METHODS else None
            ranks, error, seconds = run_one(modpath, fn, data, inject_time, num_node,
                                              entry_node=None, timeout_s=args.timeout, graph=g)
            row[f"{name}_seconds"] = seconds
            row[f"{name}_error"] = error or ""
            for k in (1, 3, 5):
                row[f"{name}_acc@{k}"] = hc.acc_at_k(ranks, set(accepted), k) if ranks else 0
                row[f"{name}_AC@{k}"] = hc.AC_at_k(ranks, set(accepted), k) if ranks else 0.0
                row[f"{name}_Avg@{k}"] = hc.Avg_at_k(ranks, set(accepted), k) if ranks else 0.0
            row[f"{name}_PRR"] = hc.exact_match_prr(ranks, set(accepted)) if ranks else 0.0
            row[f"{name}_top1"] = ranks[0] if ranks else ""
            row[f"{name}_top5"] = ",".join(ranks[:5]) if ranks else ""
            case_summary.append(f"{name}={'OK' if not error else 'ERR'}({row[f'{name}_acc@1']},{seconds:.0f}s)")
        rows.append(row)
        print(f"  {fname} (hidden={true_col}, accepted={accepted}): {' '.join(case_summary)}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\n=== hidden_rc summary (n={len(df)} cases, {(time.time()-t_start)/60:.1f} min) ===")
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
