"""
Run DCM_RCA with the SINK GRAPH on Sock Shop and Online Boutique (RE1 data,
i.e. the sock-shop-2 / online-boutique directories used by the earlier RCA/ runs).

SINK GRAPH (default): for the node under test, every other node is a parent;
all other nodes have no parents. Applied by CausalGraph.to_sink_graph(sink_node=
node), triggered by cfg.exp.snk_grp=True. Because the sink transform REPLACES
the dag entirely and keeps only the node set, the base graph written to YAML
supplies nodes only -- its edges are irrelevant. We therefore write an edge-free
entry and let the sink transform build the structure, rather than pretending a
discovered or topology graph was used.

SINK CONFOUNDED STAR (--sink_confounded): same directed edges as the sink graph,
PLUS every pair of the "other" nodes is bidirected-confounded (one shared latent
spanning all of them). Applied by CausalGraph.to_sink_confounded_graph(), same
edge-free base-yaml convention (only the node set matters). Requires
--engine original -- no early stopping exists for that engine, see conf.yaml's
snk_conf_grp comment.

Metric selection follows the baselines' own fault->metric map
(data_simulators/sock_shop.py:65-71): delay/disk/loss -> latency, cpu -> cpu,
mem -> mem. Online Boutique uses the same map against its own family names.

Scoring: Acc@k and PRR against the injected service, for all four DCM metrics
(mmd, flow, wasserstein, baro) x {add, sub}, from result.ranking().
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dcm"))
from dcm_rca import DCM_RCA  # noqa: E402

_REPO = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("RCAEVAL_DATA_ROOT", os.path.join(_REPO, "data", "rcaeval"))

DATASETS = {
    "sockshop": {
        "dir": f"{DATA_ROOT}/sock-shop-2",
        "csv": "simple_data.csv",
        # metric family per injected fault (baselines' issue_map)
        "metric_map": {"cpu": ["cpu"], "mem": ["mem"], "delay": ["latency-90"],
                       "disk": ["latency-90"], "loss": ["latency-90"]},
        "exclude": {"rabbitmq-exporter"},
    },
    "onlineboutique": {
        "dir": f"{DATA_ROOT}/online-boutique",
        "csv": "data.csv",
        # Column naming is NOT consistent across cases in this dataset: the
        # cpu/mem cases expose "<svc>_latency", while the delay/disk/loss cases
        # expose "<svc>_latency-50" and "<svc>_latency-90" instead (verified over
        # all 25 case dirs: 10 use the former, 15 the latter). Candidates are
        # tried in order, so latency-90 is preferred where present -- matching
        # what Sock Shop uses -- and plain "latency" is the fallback.
        "metric_map": {"cpu": ["cpu"], "mem": ["mem"],
                       "delay": ["latency-90", "latency"],
                       "disk": ["latency-90", "latency"],
                       "loss": ["latency-90", "latency"]},
        "exclude": {"PassthroughCluster", "frontend-external", "main", "time.1"},
    },
}
FAULTS = ["cpu", "mem", "delay", "disk", "loss"]
METRICS = ["mmd", "flow", "wasserstein", "baro"]
AGGS = ["add", "sub"]


def load_case(cfgd, case_dir, fault):
    """Return (normal_df, anomalous_df, nodes) for one case, metric-selected."""
    path = os.path.join(case_dir, cfgd["csv"])
    if not os.path.exists(path):
        return None
    inject = int(open(os.path.join(case_dir, "inject_time.txt")).read().strip())
    df = pd.read_csv(path)
    keep, fam = [], None
    for cand in cfgd["metric_map"][fault]:
        keep = [c for c in df.columns
                if c.rsplit("_", 1)[-1] == cand and c.rsplit("_", 1)[0] not in cfgd["exclude"]]
        if keep:
            fam = cand
            break
    if not keep:
        return None
    nrm = df[df["time"] < inject][keep].astype(float)
    anm = df[df["time"] >= inject][keep].astype(float)
    nrm = nrm.replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
    anm = anm.replace([float("inf"), float("-inf")], pd.NA).ffill().bfill()
    # drop constant-in-normal columns: zero variance breaks normalisation and
    # carries no signal (socket/error families are entirely constant, see
    # discovery/METRIC_SWEEP_RESULTS.md finding 4)
    keep2 = [c for c in keep if nrm[c].std() > 0 and anm[c].std() > 0]
    if len(keep2) < 3:
        return None
    nrm, anm = nrm[keep2], anm[keep2]
    nodes = [c.rsplit("_", 1)[0] for c in keep2]
    nrm.columns = nodes
    anm.columns = nodes
    return nrm.reset_index(drop=True), anm.reset_index(drop=True), nodes


def write_node_graph(nodes, path, name):
    """Edge-free graph entry: to_sink_graph() keeps only the node set."""
    entry = {"type": "admg", "dag": {n: [] for n in nodes},
             "confounders": {}, "dim_dict": {n: 1 for n in nodes}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"graphs": {name: entry}}, f, default_flow_style=False, sort_keys=False)


def prr(ranked, true_svc):
    """Rank-based PARTIAL-CREDIT score: 1.0 if true cause ranked first, decaying
    to 0 the further down the ranking it falls. NOT the paper's PRR definition
    (exact-match) -- this name collision was flagged 2026-08-17 (PRR provenance
    audit). Left unmodified: every existing DCM-RCA number in dcm/out/*_sink_
    results.csv and every other caller of this function depends on it computing
    exactly what it has always computed. Use exact_match_prr() below for the
    paper's definition instead of changing this."""
    if true_svc not in ranked:
        return 0.0
    return (len(ranked) - ranked.index(true_svc)) / len(ranked)


def exact_match_prr(ranked, true_causes):
    """PRR per the paper's stated definition (Reviewer tESe rebuttal,
    RCA/Rebuttal/Reviewer tESe.md): PRR = (1/N) sum_i 1[Rhat_i == R*_i], where
    Rhat is the algorithm's top-k prediction set with k = |R*_i| (the number of
    true root causes for that instance), checked for EXACT set equality -- not
    partial credit for a near-miss rank.

    `true_causes` may be a single service name (k=1, the common case in this
    repo's sink-graph sweeps) or a collection of names (k=|R*| > 1, the LIT
    synthetic-graph case). Returns 1 or 0 for one instance; average over
    instances yourself for the dataset-level PRR.

    Note: for every k=1 dataset in this repo (every sink-graph sweep --
    run_sink_graph.py, run_baselines.py), this is mathematically identical to
    top-1 accuracy (`acc@1`) -- both ask "is ranked[0] the single true cause".
    No separate computation was needed to audit those; the existing acc@1
    columns already are the exact-match PRR. This function exists for the
    general k>1 case and so callers don't have to reason about the equivalence
    themselves.
    """
    if isinstance(true_causes, str):
        true_causes = {true_causes}
    else:
        true_causes = set(true_causes)
    k = len(true_causes)
    if k == 0:
        return 0.0
    r_hat = set(ranked[:k])
    return 1.0 if r_hat == true_causes else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--faults", nargs="+", default=FAULTS)
    ap.add_argument("--replicates", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--num_epc", type=int, default=50)
    ap.add_argument("--thr_num", type=int, default=5)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out_root", default=os.path.join(PROJECT_ROOT, "dcm", "out"))
    ap.add_argument("--tag", default="sink")
    ap.add_argument("--sink_confounded", action="store_true",
                     help="use the 'Sink Confounded Star' graph instead of the plain sink "
                          "graph: same directed edges (every other node -> candidate Y), PLUS "
                          "every pair of the other nodes is bidirected-confounded (one shared "
                          "latent spanning all of them) -- see CausalGraph.to_sink_confounded_"
                          "graph()'s docstring. REQUIRES --engine original (no early stopping "
                          "exists for that engine; see conf.yaml's snk_conf_grp comment for why "
                          "'normal_once' can't be used here at all). Pick a distinct --tag "
                          "(e.g. sink_confounded) so results don't overwrite the plain-sink run.")
    ap.add_argument("--engine", choices=["original", "normal_once"], default="original",
                     help="'original': per-candidate O(N^2) path (unchanged default). "
                          "'normal_once': pretrain-once path, see train_normal_once.py.")
    ap.add_argument("--early_stop", action="store_true",
                     help="both engines. Train up to --max_epc, stopping each candidate "
                          "independently once its wasserstein tvd_diff stabilizes "
                          "(cfg.trn.early_stop_tol/early_stop_patience). --num_epc is ignored "
                          "when this is set. engine=original: safe with --sink_confounded too "
                          "(no cross-candidate pretrained state to desync, unlike normal_once's "
                          "Phase 1) -- see train_main.py's _train() docstring.")
    ap.add_argument("--max_epc", type=int, default=100,
                     help="ceiling used instead of --num_epc when --early_stop is set.")
    ap.add_argument("--early_stop_tol", type=float, default=None,
                     help="override cfg.trn.early_stop_tol (default from conf.yaml, currently 0.05).")
    ap.add_argument("--early_stop_patience", type=int, default=None,
                     help="override cfg.trn.early_stop_patience (default from conf.yaml, currently 1).")
    args = ap.parse_args()
    if args.sink_confounded and args.engine != "original":
        ap.error("--sink_confounded requires --engine original (no early stopping exists "
                  "for engine=normal_once with this graph -- see conf.yaml's snk_conf_grp comment)")

    cfgd = DATASETS[args.dataset]
    results = []
    t_start = time.time()

    for fault in args.faults:
        case_dirs = sorted(d for d in os.listdir(cfgd["dir"])
                           if d.endswith(f"_{fault}") and
                           os.path.isdir(os.path.join(cfgd["dir"], d)))
        for case in case_dirs:
            true_svc = case.rsplit("_", 1)[0]
            for rep in args.replicates:
                case_dir = os.path.join(cfgd["dir"], case, str(rep))
                if not os.path.isdir(case_dir):
                    continue
                loaded = load_case(cfgd, case_dir, fault)
                if loaded is None:
                    print(f"  SKIP {case}/{rep}: no usable columns", flush=True)
                    continue
                nrm, anm, nodes = loaded
                if true_svc not in nodes:
                    print(f"  SKIP {case}/{rep}: true cause {true_svc} not among nodes", flush=True)
                    continue

                exp_dst = f"{args.dataset}_{args.tag}/{fault}/{case}_r{rep}"
                out_dir = os.path.join(args.out_root, exp_dst)
                grp_name = f"{args.dataset}_{args.tag}"
                grp_yml = os.path.join(out_dir, "graph.yaml")
                write_node_graph(nodes, grp_yml, grp_name)

                extra_overrides = {}
                if args.early_stop_tol is not None:
                    extra_overrides["early_stop_tol"] = args.early_stop_tol
                if args.early_stop_patience is not None:
                    extra_overrides["early_stop_patience"] = args.early_stop_patience

                t0 = time.time()
                wall_clock_start = datetime.now(timezone.utc).isoformat()
                rca = DCM_RCA(
                    config_path=os.path.join(PROJECT_ROOT, "dcm", "conf.yaml"),
                    grp=grp_name, grp_yml=grp_yml,
                    **{"pth.rot_dir": PROJECT_ROOT},
                    dst=exp_dst,
                    num_epc=args.num_epc, thr_num=args.thr_num,
                    mdl="dcm_flow", data_type="cont", lat_dim=10,
                    upd="shared", snk_grp=not args.sink_confounded,
                    snk_conf_grp=args.sink_confounded, cmpt_grp=False,
                    gpu_id=args.gpu_id,
                    early_stop=args.early_stop, max_epc=args.max_epc,
                    **extra_overrides,
                )
                res = rca.run(normal_data=nrm, anomalous_data=anm, engine=args.engine)
                elapsed = time.time() - t0
                res.to_csv(out_dir)

                if res.scores.empty:
                    print(f"  FAIL {case}/{rep}: no nodes scored", flush=True)
                    continue
                row = {"dataset": args.dataset, "fault": fault, "case": case,
                       "replicate": rep, "true": true_svc, "n_nodes": len(nodes),
                       "wall_clock_start_utc": wall_clock_start,
                       "seconds": round(elapsed, 1),
                       "sec_per_node": round(elapsed / max(len(nodes), 1), 2)}
                if args.early_stop and res.stopping:
                    stopped = [v['stopped_epoch'] for v in res.stopping.values() if v['stopped_epoch'] is not None]
                    conv = [v['converged'] for v in res.stopping.values() if v['converged'] is not None]
                    row["mean_stopped_epoch"] = round(sum(stopped) / len(stopped), 1) if stopped else None
                    row["max_stopped_epoch"] = max(stopped) if stopped else None
                    row["n_converged"] = sum(1 for c in conv if c)
                    row["n_not_converged"] = sum(1 for c in conv if not c)
                if args.early_stop and res.pretrain_stopping:
                    row["pretrain_stopped_epoch"] = res.pretrain_stopping["stopped_epoch"]
                    row["pretrain_converged"] = res.pretrain_stopping["converged"]
                for m in METRICS:
                    for a in AGGS:
                        ranked = res.ranking(metric=m, agg=a).index.tolist()
                        for k in (1, 3, 5):
                            row[f"{m}_{a}_acc@{k}"] = int(true_svc in ranked[:k])
                        row[f"{m}_{a}_prr"] = round(prr(ranked, true_svc), 4)
                        row[f"{m}_{a}_top1"] = ranked[0]
                        row[f"{m}_{a}_top5"] = ",".join(ranked[:5])
                results.append(row)
                stop_msg = ""
                if args.early_stop and "mean_stopped_epoch" in row:
                    stop_msg = (f" mean_stopped_epoch={row['mean_stopped_epoch']} "
                                f"not_converged={row['n_not_converged']}/{len(nodes)}")
                if args.early_stop and "pretrain_stopped_epoch" in row:
                    stop_msg += (f" pretrain_epoch={row['pretrain_stopped_epoch']}"
                                 f"({'conv' if row['pretrain_converged'] else 'MAX'})")
                print(f"  {fault}/{case}/r{rep}: true={true_svc} "
                      f"w_add_top1={row['wasserstein_add_top1']} "
                      f"acc@1={row['wasserstein_add_acc@1']} ({elapsed:.0f}s){stop_msg}", flush=True)

                pd.DataFrame(results).to_csv(
                    os.path.join(args.out_root, f"{args.dataset}_{args.tag}_results.csv"),
                    index=False)

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_root, f"{args.dataset}_{args.tag}_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\ndone: {len(df)} runs in {(time.time()-t_start)/60:.1f} min -> {out_csv}")


if __name__ == "__main__":
    main()
