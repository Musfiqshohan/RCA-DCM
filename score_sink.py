"""Aggregate sink-graph run results into Acc@k / PRR tables per fault x metric x agg."""
import sys

import pandas as pd

METRICS = ["mmd", "flow", "wasserstein", "baro"]
AGGS = ["add", "sub"]


def table(df, dataset):
    print(f"\n{'='*88}\n{dataset}   (sink graph, n={len(df)} runs)\n{'='*88}")
    faults = sorted(df.fault.unique())
    for fault in faults:
        sub = df[df.fault == fault]
        print(f"\n-- fault={fault}  (n={len(sub)} runs, metric column used: "
              f"{'cpu' if fault=='cpu' else 'mem' if fault=='mem' else 'latency'}) --")
        print(f"{'metric':<18}{'Acc@1':>7}{'Acc@3':>7}{'Acc@5':>7}{'PRR':>7}")
        rows = []
        for m in METRICS:
            for a in AGGS:
                rows.append((f"{m}_{a}",
                             sub[f"{m}_{a}_acc@1"].mean(),
                             sub[f"{m}_{a}_acc@3"].mean(),
                             sub[f"{m}_{a}_acc@5"].mean(),
                             sub[f"{m}_{a}_prr"].mean()))
        for name, a1, a3, a5, p in sorted(rows, key=lambda r: -r[1]):
            print(f"{name:<18}{a1:>7.3f}{a3:>7.3f}{a5:>7.3f}{p:>7.3f}")
    # overall
    print(f"\n-- OVERALL (all faults, n={len(df)}) --")
    print(f"{'metric':<18}{'Acc@1':>7}{'Acc@3':>7}{'Acc@5':>7}{'PRR':>7}")
    rows = [(f"{m}_{a}", df[f"{m}_{a}_acc@1"].mean(), df[f"{m}_{a}_acc@3"].mean(),
             df[f"{m}_{a}_acc@5"].mean(), df[f"{m}_{a}_prr"].mean())
            for m in METRICS for a in AGGS]
    for name, a1, a3, a5, p in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<18}{a1:>7.3f}{a3:>7.3f}{a5:>7.3f}{p:>7.3f}")
    print(f"\nruntime: mean {df.seconds.mean():.0f}s/case, "
          f"{df.sec_per_node.mean():.1f}s/node, total {df.seconds.sum()/60:.0f} min")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["sockshop", "onlineboutique"]:
        path = f"dcm/out/{ds}_sink_results.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"(no results yet: {path})")
            continue
        table(df, ds)
