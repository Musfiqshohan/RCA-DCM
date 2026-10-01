# Experiment 1: causal chamber, real graph, true cause observed

DCM and 5 RCAEval baselines (nsigma, baro, rcd, circa, rcg) against
`lt_interventions_standard_v1` (light tunnel, standard configuration), using the real,
published causal graph (`ground_truth.py` — from the `causalchamber` package, verified
against the installed package and against the dataset's own columns, see that file's
docstring) instead of a sink graph. 52 cases, one per single-variable intervention; the true
cause is the intervened column itself, directly observed.

## How to reproduce

```bash
# From the repository root.

# 1. Baselines (nsigma, baro, rcd, circa, rcg) -- fast, CPU-only.
python3 baselines/causal_chamber/run_baselines_cc.py
# -> baselines/causal_chamber/out/baselines_results.csv

# 2. DCM (engine=normal_once, early_stop=True), real graph. --gpu_id required.
python3 baselines/causal_chamber/run_dcm.py --gpu_id 0
# -> baselines/causal_chamber/out/dcm_results.csv
```

Both scripts print a summary (Acc@1/3/5, error counts) at the end, write incrementally, and
keep all output under this folder's own `out/` (nothing under `dcm/out/`).

## Baseline results (n=52, 0 errors)

Acc@k: is the single true cause anywhere in the top-k? AC@k/Avg@k/PRR: the ASE'24-style
metrics below.

| method | Acc@1 | Acc@3 | Acc@5 | AC@1 | AC@3 | AC@5 | Avg@5 | PRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rcg** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| rcd | 0.904 | 0.923 | 0.923 | 0.904 | 0.923 | 0.923 | 0.919 | 0.904 |
| circa | 0.135 | 0.462 | 0.635 | 0.135 | 0.462 | 0.635 | 0.435 | 0.135 |
| nsigma | 0.058 | 0.212 | 0.308 | 0.058 | 0.212 | 0.308 | 0.192 | 0.058 |
| baro | 0.058 | 0.135 | 0.192 | 0.058 | 0.135 | 0.192 | 0.127 | 0.058 |

## DCM vs. all baselines

**Final** — both the DCM run (`engine=normal_once`, `early_stop=True`, `max_epc=50`, real
graph, GPU0) and all 5 baselines are complete (n=52, 0 errors each).

### Aggregate

`AC@k`/`Avg@k`/`PRR` per the ASE'24-style formulas in "Metrics" below (with a single true
cause here, `|V_rc|=1` always, so these are numerically identical to the simpler `acc@k`).

| method | AC@1 | AC@3 | AC@5 | Avg@5 | PRR |
|---|---:|---:|---:|---:|---:|
| **rcg** | **1.000** | 1.000 | 1.000 | **1.000** | **1.000** |
| DCM | 0.885 | 1.000 | 1.000 | 0.977 | 0.885 |
| rcd | 0.904 | 0.923 | 0.923 | 0.919 | 0.904 |
| circa | 0.135 | 0.462 | 0.635 | 0.435 | 0.135 |
| nsigma | 0.058 | 0.212 | 0.308 | 0.192 | 0.058 |
| baro | 0.058 | 0.135 | 0.192 | 0.127 | 0.058 |

`rcg` is the only method that's perfect on every case. DCM is close (0.885 AC@1, recovers to
1.000 by AC@3) — narrowly ahead of `rcd` on `AC@3`/`AC@5`/`Avg@5` but slightly behind it on
raw `AC@1`/`PRR`. Not the leader here, second- or third-best depending on the column.

### Per variable family (AC@1)

Cases grouped by which physical subsystem the intervened variable belongs to (not a
dataset-provided label — grouped here by variable role, matching the dataset's own
experiment-protocol groupings in the causal-chamber README). n=52/52 both DCM and baselines.

| family | n | DCM | rcg | rcd | circa | nsigma | baro |
|---|---:|---:|---:|---:|---:|---:|---:|
| color (R/G/B) | 6 | 1.000 | 1.000 | 1.000 | 0.500 | 0.333 | 0.000 |
| polarizer angle | 4 | 1.000 | 1.000 | 0.250 | 0.250 | 0.000 | 0.250 |
| position lights (L) | 6 | 1.000 | 1.000 | 1.000 | 0.333 | 0.000 | 0.000 |
| diode calibration (D) | 9 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| sensor gain (T) | 18 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| angle/current calib (O/R) | 9 | **0.333** | 1.000 | 0.778 | 0.111 | 0.111 | 0.222 |

DCM is perfect on 5 of 6 families — **`angle/current calib (O/R)` is its one weak spot**
(0.333), the sole reason its aggregate `AC@1` (0.885) falls short of `rcg`'s (1.000): all of
DCM's misses are concentrated in this one family (`osr_c`, `v_c`, `osr_angle_1/2`,
`v_angle_1/2`) — every other family is a clean sweep. `rcg` is perfect everywhere. `rcd` is
strong except on `polarizer angle` (0.250). `circa`/`nsigma`/`baro` are inconsistent across
families — no family is uniformly easy for them the way it is for DCM/`rcg`.

## Metrics

Formulas exactly as specified for this project's multi-root-cause evaluation (`A` = the set
of cases, `R^a[i]` = the i-th ranked node for case `a`, `V_rc^a` = its accepted root-cause
set):

```
acc@k(ranked, true) = 1  if true in ranked[:k]  else 0     (this project's original metric)

AC@k  = (1/|A|) * sum_a [ sum_{i<=k} 1[R^a[i] in V_rc^a] / min(k, |V_rc^a|) ]
Avg@k = (1/k) * sum_{j=1..k} AC@j
PRR   = (1/|A|) * sum_a 1[ {R^a[i]}_{i<=|V_rc^a|} == V_rc^a ]     (exact recovery, k=|V_rc^a|)
```

`AC@k` gives partial credit for every correct hit in the top-k (not just whether there was
at least one, like `acc@k`), normalized so a method that finds the whole accepted set within
its first `min(k,|V_rc|)` guesses scores exactly 1.0. `PRR` is fixed at `k = |V_rc^a|` per
case (not swept over 1/3/5) and requires *exact* set equality, not partial credit.

**Here every case has exactly one accepted answer (`|V_rc^a| = 1`), so `AC@k` reduces
mathematically to `acc@k`, and `PRR` reduces to `acc@1`** — confirmed identical in the table
above. Nothing new for Experiment 1 specifically; these are included for consistency with
Experiment 2 (`../causal_chamber_hidden_rc/`), where `|V_rc^a|` can be up to 7 and all three
metrics diverge meaningfully from `acc@k`.

## Intuition: why each baseline lands where it does

This is a striking reversal from this project's microservice results (Sock Shop / Online
Boutique), where `nsigma`/`baro` led and `rcg`/`rcd` trailed. The reasons are structural, not
incidental:

- **`rcg` — perfect, and that's expected, not lucky.** `rcg` scores each candidate by
  conditional mutual information with the fault indicator, *given that candidate's parents in
  the supplied graph*. With the real graph, that conditioning is exact, and — crucially —
  mutual information makes **no assumption about the functional form** of how a variable
  affects its children. It doesn't need to know the physics, only the topology.

- **`rcd` — strong, close to `rcg`.** Causal-discovery-based, not graph-fed the same way, but
  the underlying signal is unusually clean here: an intervention in this dataset is a hard,
  large step change in an actuator's operating range (`uniform_reference` samples the full
  range; `_mid`/`_strong` narrow it to a specific sub-band), not a subtle drift. A real
  discovery procedure has a lot to work with.

- **`circa` — only moderate, despite also using the real graph.** `circa`'s regression-based
  hypothesis test (`ANMRegressor`, plain linear regression per candidate) assumes each node
  is an *additive, linear* function of its parents. The light tunnel's actual physics aren't:
  polarizer transmission follows Malus's law (∝ cos²), sensor responses saturate, and light
  intensities combine multiplicatively across sources — genuinely nonlinear relationships a
  linear regressor systematically mis-fits. Same graph as `rcg`, much weaker method for *this*
  physical system specifically — a nonparametric graph-aware method (`rcg`) beats a
  parametric one (`circa`) precisely because the parametric assumption is wrong here.

- **`nsigma`/`baro` — near floor, and structurally so, not just weak.** Both score each
  column by its own marginal shift (a z-score/robust-anomaly test per variable, no
  conditioning on structure at all). The problem: in this dataset, a source variable's
  *downstream effects* (`ir_*`, `vis_*`, `current`, ...) routinely show a **larger** marginal
  shift than the source variable's own (comparatively modest) range-narrowing — a strong
  color intervention swings the sensor readings far more, in raw z-score terms, than it
  swings the LED-intensity setting itself. A graph-free method has no way to tell "this
  variable moved a lot because it's the cause" from "this variable moved a lot because it's a
  strongly-affected symptom," and with 38 candidates and direct, high-gain physical
  propagation (not the noisier, indirect propagation seen in the microservice datasets), that
  confusion is close to systematic here rather than occasional.
