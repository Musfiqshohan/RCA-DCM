"""Run the 5 kept RCAEval baseline methods (nsigma, baro, rcd, circa, rcg)
against the same Sock Shop / Online Boutique sink-graph cases run_sink_graph.py
sweeps, for direct comparison against DCM_RCA's own numbers in
dcm/out/{sockshop,onlineboutique}_sink_results.csv.

This module docstring covers what the code below does mechanically.

Data: reuses run_sink_graph.py's load_case()/DATASETS/FAULTS/prr() directly --
same case-directory shape, same column-family selection/normalization, same
constant-column dropping. Do not reinvent preprocessing here.

RCAEval source: baselines/RCAEval. Imported via sys.path insertion, not a
pip install. This copy has local modifications and excludes datasets.

Call signature (confirmed from RCAEval/main.py's own invocation):
    func(data, inject_time, dataset=None, anomalies=None, dk_select_useful=False,
         sli=sli, verbose=False, n_iter=num_node, args=Namespace(...),
         graph=graph, target_node=entry_node)
`dataset=None` is deliberate: RCAEval.io.time_series.preprocess() is a no-op
when dataset is None, so it won't re-touch data load_case() already cleaned.
`graph`: the edge-inverted call graph (call_graphs.py) -- circa/rcg use it in
place of PC-algorithm discovery / a missing graph; every other method ignores
it via **kwargs.

Per-call timeout: each (method, case) call runs in its own spawned
subprocess (see run_one/_mp_worker), forcibly terminated/killed if it exceeds
--timeout -- a ThreadPoolExecutor's nominal timeout does NOT actually bound
wall-clock (confirmed empirically: a hung call once ran 3940s despite
timeout_s=90, since Python cannot forcibly kill a thread). Subprocesses can
be killed; that's the actual fix.
"""
import argparse
import importlib
import multiprocessing
import os
import queue
import sys
import time

import pandas as pd

# This file lives at baselines/RCAEval/run_baselines.py -- two
# levels below the actual repo root (PROJECT_ROOT), which is where
# run_sink_graph.py lives and where DCM_RCA's own config paths are anchored.
RCAEVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(RCAEVAL_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RCAEVAL_DIR)

from run_sink_graph import load_case, DATASETS, FAULTS, prr  # noqa: E402
from call_graphs import edge_inverted_causal_graph  # noqa: E402

ENTRY_NODE = {"sockshop": "front-end", "onlineboutique": "frontend"}

METHOD_SPECS = {
    "nsigma": ("RCAEval.e2e", "nsigma"),
    "baro": ("RCAEval.e2e.baro", "baro"),
    "rcd": ("RCAEval.e2e.rcd", "rcd"),
    "circa": ("RCAEval.e2e.circa", "circa"),
    "rcg": ("RCAEval.e2e.rcg", "rcg"),
}
GRAPH_METHODS = {"circa", "rcg"}  # get graph=edge_inverted_causal_graph(dataset); others ignore it


def to_rca_input(nrm, anm):
    """Bridge load_case()'s (nrm_df, anm_df) -- already split, bare service-
    name columns, no time column -- into the single time-indexed DataFrame +
    inject_time row-cutoff every RCAEval e2e method expects. "time" is a
    synthetic row index (0..len-1), not a real timestamp -- every method
    needs it present for its own pre/post split (data["time"] < inject_time),
    but it is stripped from the RETURNED ranking in _mp_worker below (see
    ../../doc/BASELINES.md "time column" section)."""
    merged = pd.concat([nrm, anm], ignore_index=True)
    merged["time"] = range(len(merged))
    inject_time = len(nrm)
    return merged, inject_time


def _mp_worker(modpath, fname, data, inject_time, num_node, target, out_queue, graph=None):
    """Runs in its own spawned process (see run_one) -- re-imports the method
    fresh here rather than pickling the function object across the process
    boundary. sys.path needs re-establishing too: spawn starts a fresh
    interpreter that hasn't run this module's top-level sys.path.insert
    calls."""
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines", "RCAEval"))
    try:
        mod = importlib.import_module(modpath)
        func = getattr(mod, fname)
        run_args = argparse.Namespace(root_path=os.getcwd(), data_path=None)
        out = func(
            data, inject_time, dataset=None, anomalies=None, dk_select_useful=False,
            sli=target, verbose=False, n_iter=num_node, args=run_args,
            graph=graph, target_node=target,
        )
        ranks = out.get("ranks", []) if isinstance(out, dict) else []
        # "time" is a real column in `data` (needed internally by every
        # method for its pre/post split) but never a candidate root-cause
        # node. nsigma/baro/rcd don't drop it before ranking when
        # dataset=None (confirmed empirically); circa/rcg already drop it
        # themselves. Stripped here once, unconditionally, for all methods.
        ranks = [r for r in ranks if r != "time"]
        out_queue.put(("ok", ranks))
    except Exception as e:
        out_queue.put(("err", f"{type(e).__name__}: {e}"))


def run_one(modpath, fname, data, inject_time, num_node, entry_node, timeout_s, graph=None):
    """Call one baseline method with a REAL, hard wall-clock timeout,
    isolating any failure (including a hang) to this single (method, case)
    cell -- never re-raises. Runs in its own spawned subprocess so it can be
    forcibly killed (a thread cannot be)."""
    target = entry_node if entry_node in data.columns else None
    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue()
    proc = ctx.Process(target=_mp_worker, args=(modpath, fname, data, inject_time, num_node, target, out_queue, graph))

    t0 = time.time()
    proc.start()
    proc.join(timeout_s)
    elapsed = round(time.time() - t0, 2)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():  # terminate() didn't land in time -- escalate
            proc.kill()
            proc.join(5)
        return [], f"TIMEOUT after {timeout_s}s", elapsed

    try:
        status, payload = out_queue.get_nowait()
    except queue.Empty:
        # Process exited (crash/segfault/killed by OOM) without ever putting
        # a result -- proc.exitcode is non-zero/None in this case.
        return [], f"process exited without result (exitcode={proc.exitcode})", elapsed

    if status == "ok":
        return payload, None, elapsed
    return [], payload, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--faults", nargs="+", default=FAULTS)
    ap.add_argument("--replicates", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--methods", nargs="+", default=list(METHOD_SPECS),
                     choices=list(METHOD_SPECS), help="subset of the 5 kept methods to run")
    ap.add_argument("--timeout", type=int, default=120, help="per (method, case) wall-clock timeout, seconds")
    ap.add_argument("--out_root", default=os.path.join(PROJECT_ROOT, "dcm", "out"))
    args = ap.parse_args()

    cfgd = DATASETS[args.dataset]
    entry_node = ENTRY_NODE[args.dataset]
    graph = edge_inverted_causal_graph(args.dataset)
    results = []
    t_start = time.time()

    print(f"Running {len(args.methods)} methods on {args.dataset}: {args.methods}", flush=True)

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

                data, inject_time = to_rca_input(nrm, anm)
                num_node = len(nodes)

                row = {"dataset": args.dataset, "fault": fault, "case": case,
                       "replicate": rep, "true": true_svc, "n_nodes": num_node}
                case_summary = []
                for name in args.methods:
                    modpath, fname = METHOD_SPECS[name]
                    g = graph if name in GRAPH_METHODS else None
                    ranks, error, seconds = run_one(modpath, fname, data, inject_time, num_node,
                                                      entry_node, args.timeout, graph=g)
                    row[f"{name}_seconds"] = seconds
                    row[f"{name}_error"] = error or ""
                    for k in (1, 3, 5):
                        row[f"{name}_acc@{k}"] = int(true_svc in ranks[:k]) if ranks else 0
                    row[f"{name}_prr"] = round(prr(ranks, true_svc), 4) if ranks else 0.0
                    row[f"{name}_top1"] = ranks[0] if ranks else ""
                    row[f"{name}_top5"] = ",".join(ranks[:5]) if ranks else ""
                    case_summary.append(f"{name}={'OK' if not error else 'ERR'}"
                                         f"({row[f'{name}_acc@1']},{seconds:.0f}s)")
                results.append(row)
                print(f"  {fault}/{case}/r{rep}: true={true_svc}  {' '.join(case_summary)}", flush=True)

                pd.DataFrame(results).to_csv(
                    os.path.join(args.out_root, f"{args.dataset}_baselines_results.csv"), index=False)

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_root, f"{args.dataset}_baselines_results.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n=== {args.dataset} summary (n={len(df)} cases, {(time.time()-t_start)/60:.1f} min) ===")
    print(f"{'method':<10}{'Acc@1':>8}{'Acc@3':>8}{'Acc@5':>8}{'PRR':>8}{'n_err':>8}")
    for name in args.methods:
        n_err = (df[f"{name}_error"] != "").sum()
        print(f"{name:<10}{df[f'{name}_acc@1'].mean():>8.3f}{df[f'{name}_acc@3'].mean():>8.3f}"
              f"{df[f'{name}_acc@5'].mean():>8.3f}{df[f'{name}_prr'].mean():>8.3f}{n_err:>8}")
    try:
        dcm = pd.read_csv(os.path.join(args.out_root, f"{args.dataset}_sink_results.csv"))
        print(f"{'DCM':<10}{dcm['wasserstein_add_acc@1'].mean():>8.3f}{dcm['wasserstein_add_acc@3'].mean():>8.3f}"
              f"{dcm['wasserstein_add_acc@5'].mean():>8.3f}{dcm['wasserstein_add_prr'].mean():>8.3f}{0:>8}")
    except FileNotFoundError:
        pass
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
