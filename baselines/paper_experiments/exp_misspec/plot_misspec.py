"""Graph-misspecification figure: same data, perturbed graph.

Bars are DCM's exact root-cause-set recovery under three graphs, on the SAME 100
Exp 1 (lambda=10) datasets. Style matches the other paper panels: 95% Wilson
intervals, rotated value labels, RCA-DCM green.

The control bar is Exp 1 lambda=10's own number (read from its summary.csv)
unless a `true` arm was run through this harness, in which case that is used --
the harness-run control is preferable because it shares every code path with
the perturbed arms.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)

ARMS = [("control",     "ls10_control",     "true\ngraph"),
        ("dense",       "ls10_dense50",     "$+$50%\ndir & bi"),
        ("bi_sparse",   "ls10_bisparse50",  "$-$50%\nbidir"),
        ("both_sparse", "ls10_bothsparse50","$-$50%\ndir & bi")]
COLORS = ["#2e7d32", "#7fb069", "#e8a33d", "#c0392b"]

def collect(metric="wasserstein"):
    rows = []
    for arm, sub, lab in ARMS:
        g = glob.glob(os.path.join(HERE, sub, "misspec_*.csv"))
        if not g: print(f"  !! no csv for {arm}"); continue
        df = pd.read_csv(g[0])
        k, n = int(df[metric].sum()), len(df)
        lo, hi = wilson(k, n)
        rows.append(dict(arm=arm, label=lab, k=k, n=n, rate=k/n,
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="wasserstein")
    ap.add_argument("--width", type=float, default=2.3)
    ap.add_argument("--height", type=float, default=1.35)
    a = ap.parse_args()
    df = collect(a.metric)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "out", "summary.csv"), index=False)
    print(df.to_string(index=False))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    pos = range(len(df))
    err = [df.rate - df.ci_lo, df.ci_hi - df.rate]
    ax.bar(pos, df.rate, 0.66, color=COLORS[:len(df)], edgecolor="#1a1a1a",
           linewidth=0.45, yerr=err,
           error_kw=dict(elinewidth=0.45, capsize=0.8, capthick=0.45, ecolor="#333333"))
    for p, r, hi in zip(pos, df.rate, df.ci_hi):
        ax.text(p, hi + 0.02, f"{r:.2f}", ha="center", va="bottom",
                fontsize=5.4, color="#1a1a1a", rotation=90)
    ax.set_xticks(list(pos)); ax.set_xticklabels(df.label, fontsize=5.4, linespacing=1.05)
    ax.set_ylabel("perfect recovery", fontsize=6.8)
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=5.6, pad=1.0); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.5, color="#999999")
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "out", f"misspec_{a.metric}.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("-> out/misspec_%s.pdf" % a.metric)

if __name__ == "__main__":
    main()
