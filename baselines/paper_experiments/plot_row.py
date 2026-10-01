"""All three paper experiments as ONE ICLR-width row.

  panel 1  Exp 1 -- unobserved confounders, vs confounding strength lambda
  panel 2  Exp 2 -- heterogeneous anomaly, vs number of observed variables
  panel 3  Causal chamber -- real data, with vs without latent confounding

All three report the SAME metric: exact recovery of the root-cause set
(= "perfect recovery" in LIT, = PRR on the chamber), so one shared y-axis is
meaningful. One legend for the row; y-label on the leftmost panel only.

Value labels are ROTATED 90 degrees: at ~1.8in per panel with 12 bars a
horizontal label collides with its neighbour ("0.060.06"); vertical fits AND
can be larger.

For the variant that folds the graph-misspecification ablation into panel (a),
see plot_row_misspec.py -- it writes row_misspec_*, not row_all3_*.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PALETTES = {
    "forest": (["#5b8db8", "#e08214", "#9467bd", "#8c564b", "#c0392b", "#2e7d32"], "#1a1a1a"),
    "deep":   (["#1f4e79", "#d97706", "#7b5ea7", "#a0522d", "#a62c2c", "#1b6b3a"], "#111111"),
    "vivid":  (["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#d62728", "#2ca02c"], "#2f2f2f"),
}
ORDER = ["NSigma", "BARO", "CIRCA", "RCD", "RCG", "RCA-DCM"]

PANELS = [
    dict(csv="exp1_confounding/out/summary.csv", xcol="lam",
         ticks=lambda v: ["$\\lambda=3$", "$\\lambda=10$"],
         xlabel="confounding strength", title="(a) latent confounders"),
    dict(csv="exp2_heterogeneous/out/summary.csv", xcol="num_vars",
         ticks=lambda v: ["$n=5$", "$n=10$"],
         xlabel="number of variables", title="(b) heterogeneous anomaly"),
    dict(csv="causal_chamber/out/summary.csv", xcol="condition",
         ticks=lambda v: ["no confounder", "w/ confounders"], tick_fs=5.9,
         sublabels=["$|R^*|\\!=\\!1$", "$|R^*|\\!\\in\\!\\{1,2,3,7\\}$"],
         xlabel=None, title="(c) causal chamber"),
]

def make(pal="forest", width=5.5, height=1.30, titles=True, errorbars=True, val_fs=5.0,
         hatch_dcm=False):
    colors, edge = PALETTES[pal]
    plt.rcParams.update({"hatch.linewidth": 0.35, "hatch.color": "#ffffff",
                         "font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 6.8, "xtick.labelsize": 6.2,
                         "ytick.labelsize": 6.2, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(width, height), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, P in zip(axes, PANELS):
        f = os.path.join(HERE, P["csv"])
        if not os.path.exists(f):
            ax.text(.5, .5, "no data", ha="center", transform=ax.transAxes); continue
        df = pd.read_csv(f)
        xs = list(dict.fromkeys(df[P["xcol"]].tolist()))
        W = 0.92 / len(ORDER)
        for i, (m, c) in enumerate(zip(ORDER, colors)):
            sub = [df[(df[P["xcol"]] == x) & (df.method == m)] for x in xs]
            vals = [float(s["perfect_recovery"].iloc[0]) if len(s) else float("nan") for s in sub]
            pos = [j + (i - (len(ORDER) - 1) / 2) * W for j in range(len(xs))]
            err = None
            if errorbars:
                err = [[v - float(s["ci_lo"].iloc[0]) if len(s) else 0 for v, s in zip(vals, sub)],
                       [float(s["ci_hi"].iloc[0]) - v if len(s) else 0 for v, s in zip(vals, sub)]]
            hk = dict(hatch="///") if (hatch_dcm and m == "RCA-DCM") else {}
            ax.bar(pos, vals, W, label=m, color=c, edgecolor=edge, linewidth=0.45,
                   yerr=err, **hk,
                   error_kw=dict(elinewidth=0.45, capsize=0.6, capthick=0.45, ecolor="#333333"))
            for p_, v, s_ in zip(pos, vals, sub):
                top = float(s_["ci_hi"].iloc[0]) if (errorbars and len(s_)) else v
                ax.text(p_, top + 0.02, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=val_fs, color="#1a1a1a", rotation=90)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(P["ticks"](xs), fontsize=P.get("tick_fs", 6.8), linespacing=1.05)
        if P.get("xlabel"):
            ax.set_xlabel(P["xlabel"], labelpad=1.6, fontsize=6.6)
        for j, sl in enumerate(P.get("sublabels") or []):
            ax.annotate(sl, xy=(j, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -12), textcoords="offset points",
                        ha="center", va="top", fontsize=5.3, annotation_clip=False)
        if titles:
            ax.set_title(P["title"], fontsize=6.6, pad=2.0)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.5, color="#999999")
        ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    axes[0].set_ylabel("perfect recovery", fontsize=6.8)
    axes[0].set_ylim(0, 1.30); axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    for ax in axes: ax.tick_params(axis="y", labelsize=5.6, pad=1.0)
    axes[1].legend(frameon=False, fontsize=5.8, ncol=len(ORDER), loc="lower center",
                   bbox_to_anchor=(0.5, 1.14 if titles else 1.0), handlelength=0.7,
                   columnspacing=0.8, handletextpad=0.32, borderaxespad=0.02)
    fig.tight_layout(pad=0.20)
    tag = f"row_all3_{pal}" + ("" if titles else "_notitle") + ("_hatch" if hatch_dcm else "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "out", f"{tag}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("->", f"out/{tag}.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="all")
    ap.add_argument("--width", type=float, default=5.5)
    ap.add_argument("--height", type=float, default=1.30)
    ap.add_argument("--no_titles", action="store_true")
    ap.add_argument("--val_fs", type=float, default=5.0)
    ap.add_argument("--hatch_dcm", action="store_true")
    a = ap.parse_args()
    for p in (PALETTES if a.palette == "all" else [a.palette]):
        make(p, a.width, a.height, not a.no_titles, val_fs=a.val_fs, hatch_dcm=a.hatch_dcm)
