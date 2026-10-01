# Causal Chamber — real data, with and without latent confounding

RCA-DCM against five RCAEval baselines on the Causal Chamber light-tunnel dataset, in two
conditions: the true root cause **observed**, and the true root cause **hidden** (and so
acting as a latent confounder among its former children).

**n = 52 cases per condition.** Figure: [`out/`](out/) · Data: [`out/summary.csv`](out/summary.csv)

---

## 1 Setup

| | **no confounder** | **with confounder** |
|---|---|---|
| source | `baselines/causal_chamber/` | `baselines/causal_chamber_hidden_rc/` |
| root cause | an **observed** node | **hidden** — column dropped from the data |
| graph | true DAG, 38 nodes | true graph over the 37 remaining nodes **plus** the hidden node as an ADMG bidirected confounder group among its former children |
| \|R\*\| | **1** (single true cause) | **1–7** — 36 cases with 1, 6 with 2, 4 with 3, 6 with 7 |
| DCM | `engine='normal_once'`, fixed 50 epochs (no early stopping), `metric=wasserstein` | same |

Hiding a column is what creates the confounding: its children remain dependent through it,
but it is no longer available to condition on. Correctness is credited to **any** direct child
of the hidden cause (`hidden_config.acc_at_k`), since the cause itself is unobservable.

**Metric — PRR.** `exact_match_prr`: 1 if the top-|R\*| ranked nodes equal R\* exactly, else 0.
This is the same all-or-nothing criterion the LIT experiments call *perfect recovery*, so all
three paper figures share a y-axis. Note PRR ≡ acc@1 in the no-confounder condition, because
\|R\*\| = 1 there.

```bash
# baselines (both conditions)
python3 baselines/causal_chamber/run_baselines_cc.py
python3 baselines/causal_chamber_hidden_rc/run_baselines_cc.py
# DCM, hidden-RC, no early stopping
python3 baselines/causal_chamber_hidden_rc/run_dcm.py --gpu_id 1 --thr_num 12 \
    --no_early_stop --num_epc 50 --out baselines/causal_chamber_hidden_rc/out/dcm_results_noES.csv
python3 baselines/paper_experiments/causal_chamber/build_summary.py --dcm_hidden dcm_results_noES.csv
python3 baselines/paper_experiments/causal_chamber/plot_cc.py
```

## 2 Results (PRR, n = 52)

| method | no confounder | with confounder | Δ |
|---|---:|---:|---:|
| **RCA-DCM** | 0.885 | **0.865** | **−0.02** |
| RCG | **1.000** | 0.500 | −0.50 |
| RCD | 0.904 | 0.692 | −0.21 |
| CIRCA | 0.135 | 0.731 | +0.60 |
| BARO | 0.058 | 0.135 | +0.08 |
| NSigma | 0.058 | 0.135 | +0.08 |

**Without confounding RCA-DCM does not win** — RCG is perfect (1.000) and RCD reaches 0.904.
**With confounding it leads** (0.865 vs 0.731 for the best baseline) and is the only method
that barely moves between conditions (−0.02, against RCG's −0.50 and RCD's −0.21).

That is the expected shape: the method is built for latent confounding and pays no dividend
without it. The paper text should own this rather than let a reader discover it in the figure.

### Where the advantage comes from: multi-cause recovery

Splitting the confounded condition by \|R\*\|:

| method | PRR, \|R\*\| = 1 (n=36) | PRR, \|R\*\| > 1 (n=16) |
|---|---:|---:|
| **RCA-DCM** | 0.972 | **0.625** |
| CIRCA | 0.944 | 0.250 |
| RCG | 0.667 | 0.125 |
| **RCD** | **1.000** | **0.000** |
| BARO / NSigma | 0.194 | 0.000 |

**RCD is perfect when there is one root cause and scores zero — 0 of 16 — when there is more
than one.** Its `acc@1 = acc@3 = acc@5 = 0.904` exactly: the ranking below position 1 carries
no information, because elimination yields one surviving candidate and leaves the rest
arbitrary. Fine for top-1, fatal for exact-set recovery.

RCA-DCM recovers the full set in 10 of 16 multi-cause instances. It scores **each node
independently** by whether that node's own conditional mechanism changed, so when a hidden
cause has several children they all score high and cluster at the top together — nothing in
the scoring forces a single winner, unlike elimination (RCD) or propagate-and-attribute
(RCG, CIRCA).

## 3 Graph choice: true ADMG vs sink surrogates

The confounded condition was also run with two deliberately wrong graphs (`--sink`,
`--sink_confounded`), where every other node is treated as a parent of the candidate:

| graph handed to DCM | acc@1 |
|---|---:|
| true graph + ADMG confounders | **0.865** |
| sink | 0.731 |
| sink confounded | 0.712 |

Knowing the true graph is worth ~0.15, but the sink graph still retains ~85% of the accuracy.
This is consistent with the supergraph argument in
[`../exp_misspec/`](../exp_misspec/README.md): a denser graph remains Markov to the data, so
over-specification costs parameters, not correctness.

## 4 Honest notes

- **Early stopping barely matters here** (PRR 0.846 → 0.865, 1 paired win, 0 losses). Contrast
  LIT v10, where it cost more than half the accuracy (0.54 → 0.24): exact match over 5 root
  causes compounds snapshot noise, whereas here \|R\*\| averages 1.96.
- **PRR is uninformative in the no-confounder condition** — \|R\*\| = 1 there, so PRR ≡ acc@1
  identically for every method.
- **The \|R\*\| > 1 split is n = 16**, so 0.625 vs 0.250 rests on 10 vs 4 instances. The
  direction is clear and RCD's 0/16 is unambiguous, but report the subset size.
- **1800 `KeyError`s** appear in the DCM log for constant-valued nodes (`osr_c`,
  `osr_angle_*`, `diode_vis_*`). Pre-existing ordering bug in `train_normal_once.py:495`:
  a debug-print block reads `nrm_df[node]` before the `if node in diff: return None` guard at
  line 503, and constants have already been dropped from the frame. Those nodes are skipped
  either way, so results appear unaffected — but it is unfixed.

## 5 Figure

![causal chamber](../out/panel_causal_chamber_forest.png)

Panel (c) of the three-panel row; see [`../out/FIGURES.md`](../out/FIGURES.md).
