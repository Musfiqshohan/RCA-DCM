"""Experiment 2 -- nonlinear model with heterogeneous anomaly.

Perfect-recovery rate vs number of observed variables, RCA-DCM against five
RCAEval baselines, with 95% Wilson confidence intervals.

Style matches exp1_confounding/plot_exp1.py: compact (~0.55 of an ICLR text
row), no title, baselines first and RCA-DCM last, RCG crimson (NOT green -- it
must not be confused with RCA-DCM), value labels, hairline grid, legend above
the axes.

Wilson rather than normal-approximation intervals: at n=100 with rates near
0.14 the normal interval would extend below zero.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "out", "summary.csv"))

PALETTES = {
    "forest": (["#5b8db8", "#e08214", "#9467bd", "#8c564b", "#c0392b", "#2e7d32"], "#1a1a1a"),
    "deep":   (["#1f4e79", "#d97706", "#7b5ea7", "#a0522d", "#a62c2c", "#1b6b3a"], "#111111"),
    "vivid":  (["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#d62728", "#2ca02c"], "#2f2f2f"),
}
ORDER = ["NSigma", "BARO", "CIRCA", "RCD", "RCG", "RCA-DCM"]

def make(pal="forest", errorbars=True, xs=None):
    colors, edge = PALETTES[pal]
    XS = xs if xs is not None else sorted(df.num_vars.unique())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.4,
                         "ytick.labelsize": 6.4, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(2.95, 1.25))
    W = 0.78 / len(ORDER)
    for i, (m, c) in enumerate(zip(ORDER, colors)):
        sub = [df[(df.num_vars == x) & (df.method == m)] for x in XS]
        vals = [float(s["perfect_recovery"].iloc[0]) if len(s) else float("nan") for s in sub]
        pos = [j + (i - (len(ORDER) - 1) / 2) * W for j in range(len(XS))]
        err = None
        if errorbars:
            lo = [v - float(s["ci_lo"].iloc[0]) if len(s) else 0 for v, s in zip(vals, sub)]
            hi = [float(s["ci_hi"].iloc[0]) - v if len(s) else 0 for v, s in zip(vals, sub)]
            err = [lo, hi]
        ax.bar(pos, vals, W, label=m, color=c, edgecolor=edge, linewidth=0.5,
               yerr=err, error_kw=dict(elinewidth=0.5, capsize=0.9, capthick=0.5,
                                       ecolor="#333333"))
        for p, v, s in zip(pos, vals, sub):
            top = float(s["ci_hi"].iloc[0]) if (errorbars and len(s)) else v
            ax.text(p, top + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=4.0, color="#222222")
    ax.set_xticks(range(len(XS)))
    ax.set_xticklabels([str(x) for x in XS], fontsize=7)
    ax.set_xlabel("number of observed variables", labelpad=1.5)
    ax.set_ylabel("perfect recovery")
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.5, color="#999999")
    ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    ax.legend(frameon=False, fontsize=5.4, ncol=len(ORDER), loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=0.7, columnspacing=0.55,
              handletextpad=0.3, borderaxespad=0.02)
    fig.tight_layout(pad=0.22)
    tag = f"exp2_heterogeneous_{pal}" + ("" if errorbars else "_noerr")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "out", f"{tag}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("->", f"out/{tag}.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="all")
    ap.add_argument("--no_errorbars", action="store_true")
    a = ap.parse_args()
    for p in (PALETTES if a.palette == "all" else [a.palette]):
        make(p, not a.no_errorbars)
