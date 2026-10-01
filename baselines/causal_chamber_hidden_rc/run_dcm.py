"""Experiment 2: DCM against the hidden-true-cause datasets/graphs
(hidden_config.py). engine='normal_once' with early stopping (standing
default). Multi-label accuracy: hidden_config.acc_at_k (top-k hits ANY
direct child of the hidden true cause).

Graph mode:
  default             real graph, case-specific (37 nodes, hidden node as an
                      ADMG confounder among its children) -- what Experiment 2's
                      headline results use. engine=normal_once, early_stop=True.
  --sink              sink graph instead: no real edges/confounders at all --
                      for whichever node is under test, every OTHER node
                      (of the same 37, hidden column still dropped from the
                      data) becomes its parent. DCM's own to_sink_graph()
                      builds this per-candidate at runtime from an edge-free
                      graph; see run_sink_graph.write_node_graph() for the same
                      convention used on Sock Shop/Online Boutique. engine=
                      normal_once, early_stop=True.
  --sink_confounded   "Sink Confounded Star": same directed edges as --sink
                      (every other node -> candidate Y), PLUS every pair of
                      the other nodes is bidirected-confounded (one shared
                      latent spanning all of them) -- see
                      CausalGraph.to_sink_confounded_graph()'s docstring.
                      Reuses the SAME edge-free base graph as --sink (only the
                      node set matters; to_sink_confounded_graph() overwrites
                      dag/confounders entirely, same as to_sink_graph()) --
                      just a different snk_*_grp flag + engine. REQUIRES
                      engine='original' -- see conf.yaml's snk_conf_grp
                      comment for why 'normal_once' can't be used here at
                      all, not just "isn't the default". engine='original'
                      now has its own early-stopping too (train_main.py,
                      ported from train_normal_once.py's Phase 2 -- safe
                      here since this engine has no cross-candidate
                      pretrained state to desynchronize), so this mode also
                      runs with early_stop=True, --max_epc (default 50).

--num_shards/--shard_id: split the 52 cases across N concurrent processes
(e.g. one per GPU) via interleaved slicing (cases[shard_id::num_shards]),
each shard writing its own --out file; merge afterward.
"""
import argparse
import os
import sys
import time

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dcm"))

import hidden_config as hc  # noqa: E402
from dcm_rca import DCM_RCA  # noqa: E402


def write_sink_graph_yaml(true_col, path):
    """Edge-free 37-node graph (hidden column excluded, same node set as
    the real-graph run) -- to_sink_graph() (triggered by snk_grp=True)
    builds the per-candidate star topology from this at runtime, same
    convention as run_sink_graph.write_node_graph()."""
    import ground_truth as gt  # noqa: E402  (baselines/causal_chamber/ground_truth.py, on sys.path via hidden_config's own import)
    nodes = [v for v in gt.VARIABLES if v != true_col]
    name = f"lt_standard_hidden_{true_col}_sink"
    entry = {"type": "admg", "dag": {n: [] for n in nodes}, "confounders": {},
             "dim_dict": {n: 1 for n in nodes}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"graphs": {name: entry}}, f, default_flow_style=False, sort_keys=False)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, required=True)
    ap.add_argument("--thr_num", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "dcm_results.csv"))
    ap.add_argument("--sink", action="store_true", help="use the sink graph instead of the real graph")
    ap.add_argument("--sink_confounded", action="store_true",
                     help="use the Sink Confounded Star graph (engine='original'; early stopping "
                          "now supported for this engine too, see train_main.py)")
    ap.add_argument("--max_epc", type=int, default=50,
                     help="early-stop ceiling, --sink_confounded only (real/--sink modes hardcode 50)")
    ap.add_argument("--no_early_stop", action="store_true",
                     help="disable early stopping and train a fixed --num_epc epochs. With early "
                          "stopping the score is a single eval snapshot at the stopping epoch; "
                          "without it, a trailing average over the last 10 eval points. On LIT "
                          "v10 that difference cost more than half the accuracy, so this is the "
                          "protocol used for the paper's LIT results.")
    ap.add_argument("--num_epc", type=int, default=50,
                     help="fixed epoch count when --no_early_stop is set")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()
    assert not (args.sink and args.sink_confounded), "--sink and --sink_confounded are mutually exclusive"

    hc.build_all_hidden_datasets()
    cases = hc.cases()
    if args.num_shards > 1:
        cases = cases[args.shard_id::args.num_shards]
    mode = "sink-confounded graph" if args.sink_confounded else ("sink graph" if args.sink else "real graph")
    engine = "original" if args.sink_confounded else "normal_once"
    es = "early_stop" if not args.no_early_stop else f"FIXED {args.num_epc} epochs (no early stop)"
    print(f"Running DCM (engine={engine}, {mode}, {es}) on {len(cases)} hidden-root-cause "
          f"cases (shard {args.shard_id}/{args.num_shards}), gpu={args.gpu_id}", flush=True)

    rows = []
    t_start = time.time()
    for fname, true_col, accepted in cases:
        nrm, anm = hc.load_hidden_case(fname, true_col)
        out_subdir = "dcm_runs_sink_confounded" if args.sink_confounded else ("dcm_runs_sink" if args.sink else "dcm_runs")
        case_out_dir = os.path.join(HERE, "out", out_subdir, fname.replace(".csv", ""))

        if args.sink or args.sink_confounded:
            graph_yml = os.path.join(HERE, "out", "sink_graphs", f"{true_col}.yaml")
            graph_name = write_sink_graph_yaml(true_col, graph_yml)
        else:
            graph_name = f"lt_standard_hidden_{true_col}"
            graph_yml = os.path.join(hc.DATA_DIR, true_col, "graph.yaml")

        overrides = {"pth.rot_dir": PROJECT_ROOT, "pth.out_dir": case_out_dir,
                     "trn.early_stop": not args.no_early_stop,
                     "trn.max_epc": args.max_epc if args.sink_confounded else 50}
        if args.no_early_stop:
            overrides["trn.num_epc"] = args.num_epc

        t0 = time.time()
        rca = DCM_RCA(
            config_path=os.path.join(PROJECT_ROOT, "dcm", "conf.yaml"),
            grp=graph_name, grp_yml=graph_yml,
            **overrides,
            dst=f"causal_chamber_hidden/{fname.replace('.csv', '')}", mdl="dcm_flow",
            data_type="cont", lat_dim=10, upd="shared",
            snk_grp=args.sink, snk_conf_grp=args.sink_confounded, cmpt_grp=False,
            gpu_id=args.gpu_id, thr_num=args.thr_num,
        )
        res = rca.run(normal_data=nrm, anomalous_data=anm, engine=engine)
        elapsed = time.time() - t0

        row = {"case": fname, "true_hidden": true_col, "accepted_rc": ",".join(accepted),
               "n_nodes": nrm.shape[1], "seconds": round(elapsed, 1)}
        if not res.scores.empty:
            ranked = res.ranking(metric="wasserstein", agg="add").index.tolist()
            for k in (1, 3, 5):
                row[f"acc@{k}"] = hc.acc_at_k(ranked, set(accepted), k)
                row[f"AC@{k}"] = hc.AC_at_k(ranked, set(accepted), k)
                row[f"Avg@{k}"] = hc.Avg_at_k(ranked, set(accepted), k)
            row["PRR"] = hc.exact_match_prr(ranked, set(accepted))
            row["top1"] = ranked[0]
            row["top5"] = ",".join(ranked[:5])
        else:
            for k in (1, 3, 5):
                row[f"acc@{k}"] = 0
                row[f"AC@{k}"] = 0.0
                row[f"Avg@{k}"] = 0.0
            row["PRR"] = 0.0
            row["top1"] = row["top5"] = ""
        rows.append(row)
        print(f"  {fname} (hidden={true_col}, accepted={accepted}): top1={row['top1']} acc@1={row['acc@1']} ({elapsed:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\n=== hidden_rc DCM summary (n={len(df)} cases, {(time.time()-t_start)/60:.1f} min) ===")
    for k in (1, 3, 5):
        print(f"  Acc@{k}: {df[f'acc@{k}'].mean():.3f}   AC@{k}: {df[f'AC@{k}'].mean():.3f}")
    print(f"  Avg@5: {df['Avg@5'].mean():.3f}   PRR: {df['PRR'].mean():.3f}")
    print(f"  mean seconds/case: {df['seconds'].mean():.1f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
