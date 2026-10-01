"""Experiment 1 -- nonlinear model with unobserved confounders.

Perfect-recovery rate vs confounding strength lambda, RCA-DCM against five
RCAEval baselines. n=100 per setting.

Style matches the lit_iv_weakC figures: compact (~0.55 of an ICLR text row),
no title, RCA-DCM green / baselines in a fixed colour order, value labels,
hairline grid, legend above the axes.

RCA-DCM is reported with `wasserstein` (the repository's configured metric).
The original RCA runs read `mmd` instead; that series is available in
out/summary.csv and can be overlaid with --show_mmd.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "out", "summary.csv"))

# RCA-DCM green; baselines ordered worst->best by convention, distinct hues
# Baselines first, RCA-DCM last so the proposed method reads as the endpoint.
# RCG is crimson, NOT the olive green it had before -- at this size it was hard to
# tell apart from RCA-DCM's green, which is the one bar that must stand out.
PALETTES = {
    "forest": (["#5b8db8", "#e08214", "#9467bd", "#8c564b", "#c0392b", "#2e7d32"], "#1a1a1a"),
    "deep":   (["#1f4e79", "#d97706", "#7b5ea7", "#a0522d", "#a62c2c", "#1b6b3a"], "#111111"),
    "vivid":  (["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#d62728", "#2ca02c"], "#2f2f2f"),
}
ORDER = ["NSigma", "BARO", "CIRCA", "RCD", "RCG", "RCA-DCM"]
LAMS  = [3, 10]

def make(pal="forest", show_mmd=False, errorbars=True):
    colors, edge = PALETTES[pal]
    methods = list(ORDER)
    cols = list(colors)
    if show_mmd:
        methods.append("RCA-DCM (mmd)"); cols.append("#9ccc9e")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 6.4,
                         "ytick.labelsize": 6.4, "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(2.95, 1.25))
    W = 0.78 / len(methods)
    for i, (m, c) in enumerate(zip(methods, cols)):
        vals = [float(df[(df.lam == l) & (df.method == m)]["perfect_recovery"].iloc[0])
                for l in LAMS]
        pos = [j + (i - (len(methods) - 1) / 2) * W for j in range(len(LAMS))]
        sub = [df[(df.lam == l) & (df.method == m)] for l in LAMS]
        err = None
        if errorbars and "ci_lo" in df.columns:
            err = [[v - float(s_["ci_lo"].iloc[0]) for v, s_ in zip(vals, sub)],
                   [float(s_["ci_hi"].iloc[0]) - v for v, s_ in zip(vals, sub)]]
        ax.bar(pos, vals, W, label=m, color=c, edgecolor=edge, linewidth=0.5,
               yerr=err, error_kw=dict(elinewidth=0.5, capsize=0.9, capthick=0.5,
                                       ecolor="#333333"))
        for p_, v, s_ in zip(pos, vals, sub):
            top = float(s_["ci_hi"].iloc[0]) if (errorbars and "ci_lo" in df.columns) else v
            ax.text(p_, top + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=4.0, color="#222222")
    ax.set_xticks(range(len(LAMS)))
    ax.set_xticklabels([str(l) for l in LAMS], fontsize=7)
    ax.set_xlabel("confounding strength $\\lambda$", labelpad=1.5)
    ax.set_ylabel("perfect recovery")
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.5, color="#999999")
    ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    ax.legend(frameon=False, fontsize=5.4, ncol=len(methods), loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=0.7, columnspacing=0.55,
              handletextpad=0.3, borderaxespad=0.02)
    fig.tight_layout(pad=0.22)
    tag = f"exp1_confounding_{pal}" + ("_mmd" if show_mmd else "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "out", f"{tag}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("->", f"out/{tag}.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="all")
    ap.add_argument("--show_mmd", action="store_true")
    ap.add_argument("--no_errorbars", action="store_true")
    a = ap.parse_args()
    for p in (PALETTES if a.palette == "all" else [a.palette]):
        make(p, a.show_mmd, not a.no_errorbars)
