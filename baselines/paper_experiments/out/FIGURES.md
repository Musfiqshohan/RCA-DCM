# Paper figures — files and captions

All figures report the **same metric**: exact recovery of the root-cause set
(top-|R\*| predictions equal R\* exactly). LIT calls this *perfect recovery*; the causal
chamber calls it *PRR*. Identical criterion, so one shared y-axis is meaningful.

Regenerate:

```bash
python3 baselines/paper_experiments/plot_row.py    --palette forest --height 1.42 [--hatch_dcm]
python3 baselines/paper_experiments/plot_panels.py --palette forest              [--hatch_dcm]
```

---

## Combined row (recommended) — 5.5in × 1.42in, one `\includegraphics`

| file | RCA-DCM bars |
|---|---|
| `row_all3_forest.pdf` | plain |
| `row_all3_forest_hatch.pdf` | hatched |

Also in `deep` and `vivid` palettes.

### Caption

> **Figure N.** Exact root-cause-set recovery across three benchmarks (n=100 per LIT
> setting, n=52 per chamber condition; 95% confidence intervals). **(a)** Synthetic ADMGs with
> 6 observed and 4 latent variables as confounding strength λ increases. **(b)**
> Heterogeneous anomalies, where shift magnitude scales with topological position so a
> non-root-cause descendant can out-shift a true cause. **(c)** Causal chamber (real data),
> with the true root cause observed versus hidden and modelled as a latent confounder.
> RCA-DCM is the only method that stays strong across all three: the marginal-shift
> detectors (NSigma, BARO) collapse whenever a shift is inherited rather than injected, and
> the graph-based methods (RCD, RCG, CIRCA) degrade once latent confounding breaks the
> observed DAG they rely on.

**Shorter:**

> **Figure N.** Exact root-cause-set recovery (95% confidence intervals). **(a)** increasing
> latent confounding, **(b)** heterogeneous anomalies where a non-cause can out-shift a true
> cause, **(c)** real causal-chamber data with the root cause observed vs. hidden. RCA-DCM
> is the only method strong in all three settings.

---

## Individual panels — 1.85in each, three per row

| file | contents | legend |
|---|---|---|
| `panel_exp1_confounding_forest.pdf` | (a) λ ∈ {3, 10} | **yes** — serves all three |
| `panel_exp2_heterogeneous_forest.pdf` | (b) n ∈ {5, 10} | no |
| `panel_causal_chamber_forest.pdf` | (c) no conf. / conf. | no |

`_hatch` variants available. The exp1 panel is ~0.26in taller because it carries the
shared legend; align with `\raisebox` or use the combined row instead.

### Sub-captions

- **(a)** Synthetic ADMGs, 6 observed + 4 latent variables, ⌊p/2⌋ = 3 root causes;
  λ scales all latent→observed coefficients.
- **(b)** Heterogeneous anomalies, no latent confounders, ⌊n/2⌋ root causes; injected shift
  scales with topological position, so shifts accumulate along directed paths.
- **(c)** Causal chamber. *no conf.*: root cause observed, true DAG, |R\*| = 1.
  *conf.*: root cause hidden and represented as an ADMG bidirected confounder among its
  former children, |R\*| = 1–7.

---

## Misspecification figure (separate, not part of the row)

`../exp_misspec/out/misspec_wasserstein.pdf` — 2.3in × 1.35in, four bars on the same
Exp 1 λ=10 datasets with only the graph perturbed.

> **Figure N.** Graph misspecification. RCA-DCM's exact root-cause-set recovery on the same
> 100 datasets when the supplied graph is perturbed (n=100, 95% confidence intervals). Adding
> 50% spurious edges (p=0.39) or deleting 50% of the bidirected edges (p=1.00) leaves accuracy
> statistically unchanged; only deleting directed edges degrades it (0.88 → 0.76, p=0.019,
> paired McNemar). See [`../exp_misspec/README.md`](../exp_misspec/README.md).

## Design notes

- **Value labels rotated 90°.** At ~1.8in per panel with 12 bars there is ~0.1in per bar;
  horizontal labels collided ("0.060.06"). Rotated they are both larger and unambiguous.
- **Hatch on RCA-DCM only** (`--hatch_dcm`). Keeps the proposed method identifiable in
  greyscale print and for colourblind readers, where hue alone does not distinguish it.
- **RCG is crimson, not olive green** — at this size the earlier green was confusable with
  RCA-DCM's, the one bar that must stand out.
- **Baselines first, RCA-DCM last**, so the proposed method reads as the endpoint.
- **Error bars are 95% confidence intervals** computed by the **Wilson score** method.
  Say "95% confidence interval" in the caption; the method name belongs in the appendix.
  The textbook interval `p ± 1.96·sqrt(p(1-p)/n)` is unusable here because several results
  sit at the extremes: at RCG's 1.00 (52/52) it returns a zero-width bar claiming certainty,
  and at BARO's 0.06 it returns a negative lower bound. Wilson gives [0.93, 1.00] and
  [0.02, 0.16]. Mid-range (e.g. 0.54) the two agree to two decimals.

## Status

**All data final.** Every result uses fixed-epoch training (no early stopping) and n=100 per
LIT setting / n=52 per chamber condition.

| panel | source |
|---|---|
| (a) | `exp1_confounding/ls{3,10}_noES` |
| (b) | `exp2_heterogeneous/{v5_floor_noES,v10_noES}` |
| (c) | `causal_chamber` + `causal_chamber_hidden_rc/out/dcm_results_noES.csv` |

Per-experiment write-ups: [`exp1_confounding/README.md`](../exp1_confounding/README.md),
[`exp2_heterogeneous/README.md`](../exp2_heterogeneous/README.md),
[`causal_chamber/README.md`](../causal_chamber/README.md),
[`exp_misspec/README.md`](../exp_misspec/README.md).

**RCA-DCM is significantly better than all five baselines in all four LIT settings**
(paired McNemar, p < 0.05 throughout; weakest is BARO at λ=3, p=0.027). On the chamber it
leads under confounding (0.87 vs 0.73 for the best baseline) but not without it, where RCG
is perfect (1.00) and RCD reaches 0.90.

Early stopping scores from a single eval snapshot rather than a trailing average over the
last 10 eval points. On LIT v10 that cost more than half the accuracy (0.54 → 0.24), which
is why every reported number uses fixed epochs. On the chamber the same change was worth
only ~0.02 (PRR 0.846 → 0.865) — exact-match over 5 root causes compounds snapshot noise,
whereas the chamber's |R*| averages 1.96.
