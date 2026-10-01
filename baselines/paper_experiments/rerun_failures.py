"""Paired re-run: take the iterations where DCM FAILED under early stopping and
re-score those exact datasets with early stopping OFF (fixed --num_epc).

Isolates the effect of the early-stop configuration, holding the data fixed --
far more informative than comparing two independent random samples, and much
cheaper since only the failures need re-running.

Turning early stopping off changes TWO things at once, which is the point:
  1. training length  (early stop averaged ~27 epochs vs a fixed num_epc)
  2. the SCORING RULE -- with early_stop the score is a single eval snapshot at
     the stopping epoch (_compute_scores_at_epoch); without it, it is a trailing
     average over the last 10 eval points (_compute_scores). The trailing average
     is far less noisy, which matters for exact-match over many root causes.
"""
import argparse, json, os, sys, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "dcm"))
from dcm_rca import DCM_RCA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="dir holding dataset*/ subdirs")
    ap.add_argument("--succ", required=True, help="succ.json from the early-stopped run")
    ap.add_argument("--graph_name", required=True, help="e.g. lit_random_v10_lat0")
    ap.add_argument("--num_epc", type=int, default=50)
    ap.add_argument("--metric", default="wasserstein")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--thr_num", type=int, default=6)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = json.load(open(a.succ))["dcm_flow"]["res"]
    failed = [i for i, x in enumerate(res) if x.split("/")[0] != x.split("/")[1]]
    print(f"{len(failed)} failed iterations under early stopping; re-running first {a.limit}", flush=True)

    rows = []
    for i in failed[:a.limit]:
        cand = glob.glob(os.path.join(a.data_root, f"dataset{i+1}_*"))
        if not cand:
            print(f"  iter {i}: dataset dir missing, skipped", flush=True); continue
        d = cand[0]
        nrm = pd.read_csv(os.path.join(d, "disc_nrm.csv"))
        anm = pd.read_csv(os.path.join(d, "disc_anm.csv"))
        true_rc = {f"X{j}" for j in json.load(open(os.path.join(d, "true_rc.json")))["true_rc"]}

        rca = DCM_RCA(
            config_path=os.path.join(ROOT, "dcm", "conf.yaml"),
            grp=a.graph_name, grp_yml=os.path.join(d, "graph.yaml"),
            **{"pth.rot_dir": ROOT,
               "pth.out_dir": os.path.join(a.out, "runs", f"iter{i}")},
            dst=f"rerun/iter{i}", num_epc=a.num_epc, thr_num=a.thr_num,
            mdl="dcm_flow", data_type="cont", lat_dim=10,
            upd="shared", snk_grp=False, gpu_id=a.gpu_id,
        )
        r = rca.run(normal_data=nrm, anomalous_data=anm)   # engine='original', NO early stop
        k = len(true_rc)
        got = {}
        for met in ["wasserstein", "mmd", "flow"]:
            pred = set(r.ranking(metric=met, agg="add").index.tolist()[:k])
            got[met] = int(pred == true_rc)
        rows.append({"iter": i, "k": k, "old_es_result": res[i], **{f"noES_{m}": v for m, v in got.items()}})
        print(f"  iter {i:3d}  was {res[i]:>5s}  ->  noES exact: "
              f"wass={got['wasserstein']} mmd={got['mmd']} flow={got['flow']}", flush=True)
        pd.DataFrame(rows).to_csv(a.out + "/rerun_failures.csv", index=False)

    df = pd.DataFrame(rows)
    print(f"\n=== {len(df)} previously-FAILED datasets, re-run without early stopping ===")
    for m in ["wasserstein", "mmd", "flow"]:
        print(f"  now correct with {m:12s}: {df[f'noES_{m}'].sum():3d}/{len(df)} "
              f"({df[f'noES_{m}'].mean():.2f})")
    print("\n(all of these scored 0 under early stopping, so any non-zero is a strict gain)")


if __name__ == "__main__":
    main()
