"""The three paper figures as separate one-third-width ICLR panels.

Each panel is ~1.85in so three sit in one 5.5in text row. One shared plotting
routine keeps geometry, palette and bar order identical across them.

Layout decisions forced by the width:
  * x-axis title AND self-describing tick labels ("$\\lambda=3$", "$n=5$"):
    the ticks give the values, the title names the quantity.
  * Value labels are ROTATED 90 degrees. At 1.85in with 12 bars there is about
    0.1in per bar; horizontal labels overlapped illegibly ("0.060.06"), and
    shrinking them defeats the purpose. Rotated, they can be larger AND clear.
  * Legend ONLY on the confounding panel (exp1) -- colours are identical across
    all three, so one legend serves the row. That panel is correspondingly
    taller; pass --legend_height to match heights if your LaTeX needs it.
  * y tick labels are smaller than the value labels: the per-bar numbers carry
    the result, the axis only provides scale.
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

PANELS = {
    "exp1": dict(csv="exp1_confounding/out/summary.csv", xcol="lam",
                 ticks=["$\\lambda=3$", "$\\lambda=10$"], legend=True,
                 xlabel="confounding strength", tag="panel_exp1_confounding"),
    "exp2": dict(csv="exp2_heterogeneous/out/summary.csv", xcol="num_vars",
                 ticks=["$n=5$", "$n=10$"], legend=False,
                 xlabel="number of variables", tag="panel_exp2_heterogeneous"),
    "cc":   dict(csv="causal_chamber/out/summary.csv", xcol="condition",
                 ticks=["no confounder", "w/ confounders"], tick_fs=5.9,
                 sublabels=["$|R^*|\\!=\\!1$", "$|R^*|\\!\\in\\!\\{1,2,3,7\\}$"],
                 legend=False, xlabel=None, tag="panel_causal_chamber"),
}

def make(key, pal="forest", width=1.85, height=1.35, ymax=1.30,
         val_fs=5.2, tick_fs=6.8, ytick_fs=5.4, hatch_dcm=False):
    P = PANELS[key]
    colors, edge = PALETTES[pal]
    f = os.path.join(HERE, P["csv"])
    if not os.path.exists(f):
        print(f"  !! missing {P['csv']}"); return
    df = pd.read_csv(f)
    xs = list(dict.fromkeys(df[P["xcol"]].tolist()))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7,
                         "hatch.linewidth": 0.35, "hatch.color": "#ffffff"})
    h = height + (0.26 if P["legend"] else 0.0)
    fig, ax = plt.subplots(figsize=(width, h))
    W = 0.92 / len(ORDER)                       # wide bars; rotated labels keep them legible
    for i, (m, c) in enumerate(zip(ORDER, colors)):
        sub = [df[(df[P["xcol"]] == x) & (df.method == m)] for x in xs]
        vals = [float(s["perfect_recovery"].iloc[0]) if len(s) else float("nan") for s in sub]
        pos = [j + (i - (len(ORDER) - 1) / 2) * W for j in range(len(xs))]
        err = [[v - float(s["ci_lo"].iloc[0]) if len(s) else 0 for v, s in zip(vals, sub)],
               [float(s["ci_hi"].iloc[0]) - v if len(s) else 0 for v, s in zip(vals, sub)]]
        hk = dict(hatch="///") if (hatch_dcm and m == "RCA-DCM") else {}
        ax.bar(pos, vals, W, label=m, color=c, edgecolor=edge, linewidth=0.45, yerr=err, **hk,
               error_kw=dict(elinewidth=0.45, capsize=0.6, capthick=0.45, ecolor="#333333"))
        for p_, v, s in zip(pos, vals, sub):
            top = float(s["ci_hi"].iloc[0]) if len(s) else v
            ax.text(p_, top + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=val_fs, color="#1a1a1a", rotation=90)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(P["ticks"], fontsize=P.get("tick_fs", tick_fs), linespacing=1.05)
    if P.get("xlabel"):
        ax.set_xlabel(P["xlabel"], labelpad=1.6, fontsize=6.6)
    for j, sl in enumerate(P.get("sublabels") or []):
        ax.annotate(sl, xy=(j, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=5.3, annotation_clip=False)
    ax.set_ylabel("perfect recovery", fontsize=6.8, labelpad=1.5)
    ax.set_ylim(0, ymax); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=ytick_fs, pad=1.0)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.5, color="#999999")
    ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555"); ax.spines["bottom"].set_color("#555555")
    if P["legend"]:
        ax.legend(frameon=False, fontsize=5.0, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), handlelength=0.7, columnspacing=0.6,
                  handletextpad=0.3, borderaxespad=0.02, labelspacing=0.25)
    fig.tight_layout(pad=0.18)
    tag = f"{P['tag']}_{pal}" + ("_hatch" if hatch_dcm else "")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "out", f"{tag}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("->", f"out/{tag}.pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", default="forest")
    ap.add_argument("--panels", nargs="+", default=["exp1", "exp2", "cc"])
    ap.add_argument("--width", type=float, default=1.85)
    ap.add_argument("--height", type=float, default=1.35)
    ap.add_argument("--val_fs", type=float, default=5.2)
    ap.add_argument("--hatch_dcm", action="store_true")
    a = ap.parse_args()
    pals = list(PALETTES) if a.palette == "all" else [a.palette]
    for p in pals:
        for k in a.panels:
            make(k, p, a.width, a.height, val_fs=a.val_fs, hatch_dcm=a.hatch_dcm)
