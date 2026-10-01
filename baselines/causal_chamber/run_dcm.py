"""Run DCM on lt_interventions_standard_v1 using the REAL published causal
graph (ground_truth.py / config.write_causal_graph) -- not a sink graph.

engine='normal_once' with early stopping is the standing default now (not
'original') -- see dcm/train_normal_once.py + EFFICIENCY_PLAN.md's history
for why. GPU is a required flag (--gpu_id), not defaulted, since this shares
the machine with other concurrent experiments.

--sink_confounded: use the "Sink Confounded Star" graph instead of the real
graph -- every other node -> candidate Y, plus every pair of the other nodes
bidirected-confounded (one shared latent spanning all of them); see
CausalGraph.to_sink_confounded_graph()'s docstring. REQUIRES engine='original'
(no early-stopping path exists for engine='normal_once' on this graph -- a
node's confounder-partner set changes depending on which node is currently the
candidate, so normal_once's Phase-1 pretrain-once-and-reuse shortcut would be
silently wrong; see dcm/conf.yaml's snk_conf_grp comment). engine='original'
has its own early-stopping now too (train_main.py, ported from
train_normal_once.py's Phase 2) -- safe here since this engine has no
cross-candidate pretrained state to desynchronize.
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

import config as cc  # noqa: E402
import ground_truth as gt  # noqa: E402
from dcm_rca import DCM_RCA  # noqa: E402

GRAPH_NAME = "lt_standard_real"
GRAPH_YML = os.path.join(HERE, "out", GRAPH_NAME, "graph.yaml")


def write_sink_graph_yaml(path, name="lt_standard_sink_confounded"):
    """Edge-free 38-node graph (same node set as the real graph) --
    to_sink_confounded_graph() (triggered by snk_conf_grp=True) builds the
    per-candidate star+confounder structure from this at runtime; only the
    node SET matters, same convention as causal_chamber_hidden_rc/run_dcm.py's
    write_sink_graph_yaml()."""
    entry = {"type": "admg", "dag": {n: [] for n in gt.VARIABLES},
             "confounders": {}, "dim_dict": {n: 1 for n in gt.VARIABLES}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"graphs": {name: entry}}, f, default_flow_style=False, sort_keys=False)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_id", type=int, required=True)
    ap.add_argument("--thr_num", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "dcm_results.csv"))
    ap.add_argument("--sink_confounded", action="store_true",
                     help="use the Sink Confounded Star graph instead of the real graph "
                          "(requires engine='original', enforced below)")
    ap.add_argument("--max_epc", type=int, default=50,
                     help="early-stop ceiling for both engines (real graph: normal_once; "
                          "--sink_confounded: original)")
    args = ap.parse_args()

    engine = "original" if args.sink_confounded else "normal_once"
    if args.sink_confounded:
        graph_name = write_sink_graph_yaml(GRAPH_YML.replace(GRAPH_NAME, "lt_standard_sink_confounded"))
        graph_yml = GRAPH_YML.replace(GRAPH_NAME, "lt_standard_sink_confounded")
    else:
        graph_name = cc.write_causal_graph(GRAPH_YML, GRAPH_NAME)
        graph_yml = GRAPH_YML

    cases = cc.cases()
    mode = "sink-confounded graph" if args.sink_confounded else "real graph"
    print(f"Running DCM (engine={engine}, {mode}, early_stop=True, max_epc={args.max_epc}) "
          f"on {len(cases)} causal-chamber cases, gpu={args.gpu_id}", flush=True)

    rows = []
    t_start = time.time()
    for fname, true_col in cases:
        nrm, anm = cc.load_case(fname)
        exp_dst = f"causal_chamber/{fname.replace('.csv', '')}"

        # out_dir overridden directly (full dotted path, bypasses the
        # ${pth.rot_dir}/dcm/out/... template) so results land under this
        # experiment's own out/ folder, not dcm/out/.
        out_subdir = "dcm_runs_sink_confounded" if args.sink_confounded else "dcm_runs"
        case_out_dir = os.path.join(HERE, "out", out_subdir, fname.replace(".csv", ""))
        t0 = time.time()
        rca = DCM_RCA(
            config_path=os.path.join(PROJECT_ROOT, "dcm", "conf.yaml"),
            grp=graph_name, grp_yml=graph_yml,
            **{"pth.rot_dir": PROJECT_ROOT, "pth.out_dir": case_out_dir,
               "trn.early_stop": True, "trn.max_epc": args.max_epc},
            dst=exp_dst, mdl="dcm_flow", data_type="cont", lat_dim=10,
            upd="shared", snk_grp=False, snk_conf_grp=args.sink_confounded,
            cmpt_grp=False, gpu_id=args.gpu_id,
            thr_num=args.thr_num,
        )
        res = rca.run(normal_data=nrm, anomalous_data=anm, engine=engine)
        elapsed = time.time() - t0

        row = {"case": fname, "true": true_col, "n_nodes": len(gt.VARIABLES), "seconds": round(elapsed, 1)}
        if not res.scores.empty:
            ranked = res.ranking(metric="wasserstein", agg="add").index.tolist()
            for k in (1, 3, 5):
                row[f"wasserstein_add_acc@{k}"] = cc.acc_at_k(ranked, true_col, k)
                row[f"wasserstein_add_AC@{k}"] = cc.AC_at_k(ranked, true_col, k)
                row[f"wasserstein_add_Avg@{k}"] = cc.Avg_at_k(ranked, true_col, k)
            row["wasserstein_add_PRR"] = cc.exact_match_prr(ranked, true_col)
            row["top1"] = ranked[0]
            row["top5"] = ",".join(ranked[:5])
        else:
            for k in (1, 3, 5):
                row[f"wasserstein_add_acc@{k}"] = 0
                row[f"wasserstein_add_AC@{k}"] = 0.0
                row[f"wasserstein_add_Avg@{k}"] = 0.0
            row["wasserstein_add_PRR"] = 0.0
            row["top1"] = row["top5"] = ""
        rows.append(row)
        print(f"  {fname} (true={true_col}): top1={row['top1']} acc@1={row['wasserstein_add_acc@1']} ({elapsed:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    print(f"\n=== causal_chamber DCM summary (n={len(df)} cases, {(time.time()-t_start)/60:.1f} min) ===")
    for k in (1, 3, 5):
        print(f"  Acc@{k}: {df[f'wasserstein_add_acc@{k}'].mean():.3f}   AC@{k}: {df[f'wasserstein_add_AC@{k}'].mean():.3f}")
    print(f"  Avg@5: {df['wasserstein_add_Avg@5'].mean():.3f}   PRR: {df['wasserstein_add_PRR'].mean():.3f}")
    print(f"  mean seconds/case: {df['seconds'].mean():.1f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
