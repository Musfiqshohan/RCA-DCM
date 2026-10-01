"""Build summary.csv for the causal-chamber figure.

Two conditions, side by side:
  "no confounder"  -- baselines/causal_chamber          (root cause OBSERVED,
                      true DAG, |R*| = 1 always)
  "with confounder"-- baselines/causal_chamber_hidden_rc(root cause HIDDEN, true
                      graph + the hidden node as an ADMG bidirected confounder
                      group among its former children; |R*| = 1..7)

Metric is PRR = exact set match at k = |R*| (config.exact_match_prr /
hidden_config.exact_match_prr) -- the SAME all-or-nothing criterion the LIT
experiments call "perfect recovery", so all three figures share a y-axis.

Note PRR == acc@1 identically in the no-confounder condition, because k=1 there.
"""
import os, sys, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASE = ["nsigma", "baro", "circa", "rcd", "rcg"]
DISP = {"nsigma": "NSigma", "baro": "BARO", "circa": "CIRCA",
        "rcd": "RCD", "rcg": "RCG"}

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0.0, c-h), min(1.0, c+h))

def row(cond, name, vec):
    vec = np.asarray(vec, dtype=float)
    k, n = int(vec.sum()), len(vec)
    lo, hi = wilson(k, n)
    return {"condition": cond, "method": name, "k": k, "n": n,
            "perfect_recovery": round(k/n, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "outcomes": "".join(str(int(x)) for x in vec)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcm_hidden", default="dcm_results.csv",
                    help="which hidden-RC DCM csv to use (dcm_results.csv or dcm_results_noES.csv)")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "summary.csv"))
    a = ap.parse_args()
    rows = []

    # ---- no confounder (observed root cause) ----
    d = os.path.join(ROOT, "baselines", "causal_chamber", "out")
    b = pd.read_csv(os.path.join(d, "baselines_results.csv"))
    for m in BASE:
        rows.append(row("no confounder", DISP[m], b[f"{m}_PRR"]))
    dc = pd.read_csv(os.path.join(d, "dcm_results.csv"))
    rows.append(row("no confounder", "RCA-DCM", dc["wasserstein_add_PRR"]))

    # ---- with confounder (hidden root cause) ----
    d = os.path.join(ROOT, "baselines", "causal_chamber_hidden_rc", "out")
    b = pd.read_csv(os.path.join(d, "baselines_results.csv"))
    for m in BASE:
        rows.append(row("with confounder", DISP[m], b[f"{m}_PRR"]))
    f = os.path.join(d, a.dcm_hidden)
    if os.path.exists(f):
        rows.append(row("with confounder", "RCA-DCM", pd.read_csv(f)["PRR"]))
    else:
        print(f"  !! {a.dcm_hidden} not found -- RCA-DCM missing for 'with confounder'")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.drop(columns=["outcomes"]).to_string(index=False))

if __name__ == "__main__":
    main()
