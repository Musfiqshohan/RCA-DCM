"""Build summary.csv for a paper experiment, including the per-iteration counts
needed for Wilson confidence intervals and paired McNemar tests.

Stores k (successes) and n alongside the rate so error bars can be recomputed
without re-reading succ.json, and dumps the per-iteration 0/1 vector so paired
tests between methods remain possible after the fact.
"""
import json, glob, os, sys, argparse
import numpy as np, pandas as pd

NAMES = [('dcm_flow', 'RCA-DCM'), ('nsigma', 'NSigma'), ('baro', 'BARO'),
         ('circa', 'CIRCA'), ('rcd', 'RCD'), ('rcg', 'RCG')]
EXTRA = [('dcm_flow_mmd_add', 'RCA-DCM (mmd)'), ('dcm_flow_flow_add', 'RCA-DCM (flow)')]

def outcomes(d, m):
    r = d.get(m, {}).get('res', [])
    return np.array([1 if x.split('/')[0] == x.split('/')[1] else 0 for x in r])

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0.0, c-h), min(1.0, c+h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', nargs='+', required=True,
                    help='label=glob pairs, e.g. 5=exp2_heterogeneous/v5_floor_noES')
    ap.add_argument('--xname', default='setting')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for s in a.settings:
        lab, sub = s.split('=', 1)
        g = glob.glob(os.path.join(HERE, sub, 'data', '*', 'succ.json'))
        if not g:
            print(f'  !! no succ.json for {lab} ({sub})'); continue
        d = json.load(open(g[0]))
        for key, name in NAMES + EXTRA:
            o = outcomes(d, key)
            if not len(o): continue
            k, n = int(o.sum()), len(o)
            lo, hi = wilson(k, n)
            rows.append({a.xname: lab, 'method': name, 'k': k, 'n': n,
                         'perfect_recovery': round(k/n, 4),
                         'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                         'outcomes': ''.join(map(str, o))})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.drop(columns=['outcomes']).to_string(index=False))

if __name__ == '__main__':
    main()
