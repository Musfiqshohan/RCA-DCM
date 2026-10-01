# Graph Misspecification — same data, deliberately wrong graph

Does RCA-DCM still recover the root causes when the graph it is given is **wrong**?
The datasets are held fixed and only the graph handed to DCM is perturbed, so any change
in accuracy is attributable to the graph error alone.

Figure: [`out/misspec_wasserstein.pdf`](out/) · Data: [`out/summary.csv`](out/summary.csv)

---

## 1 Why the data is held fixed

An earlier design varied `--edge_density`, but that regenerates the true graph **and** the
data from it, leaving DCM with a *correct* graph throughout — it measures sparse-vs-dense
capability, not misspecification. Here the 100 datasets are reused verbatim and only the
graph changes.

## 2 Setup

> **Dataset location.** The datasets are *not* duplicated into this folder. All four arms
> read, read-only, from
> `baselines/paper_experiments/exp1_confounding/ls10_noES/data/lit_random_v10_lat4_none_latnstrg10.0_Ns5000As5000/`
> (100 × `dataset<i>_seed<r>/`, 285 MB total, each holding `disc_nrm.csv`, `disc_anm.csv`,
> `cont_*.csv`, `true_rc.json` and its own `graph.yaml`). `preserve.sh` records an MD5 for
> every file actually read, so the dependency on Experiment 1 is explicit and any later drift
> is detectable.

Datasets are the 100 already generated for **Experiment 1, λ = 10** (6 observed + 4 latent
variables, 3 root causes, 5000 normal + 5000 anomalous samples). Every dataset carries its
own `graph.yaml`, so the true ADMG is known per instance. Across the 100 instances the true
graphs are **all distinct** (mean 7.0 directed edges, 4.0 bidirected/confounder pairs;
6 nodes → 15 possible pairs, so 8 free directed and 11 free bidirected slots).

| arm | graph handed to DCM | per-instance change |
|---|---|---|
| **control** | true ADMG | — |
| **dense (+50%)** | 7 → **11** directed, 4 → **6** bidirected | +4 directed, +2 bidirected |
| **bi_sparse (−50%)** | 7 directed, 4 → **2** bidirected | −2 bidirected |

Everything else is identical to the Exp 1 run: `engine='original'`, fixed 50 epochs (no
early stopping), `lat_dim=10`, `metric=wasserstein`. DCM only — it is the one method whose
input graph is being corrupted.

```bash
D=baselines/paper_experiments/exp1_confounding/ls10_noES/data/*/
R=baselines/paper_experiments/exp_misspec
python3 $R/run_misspec.py --data_root "$D" --mode dense     --frac 0.5 \
    --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_dense50
python3 $R/run_misspec.py --data_root "$D" --mode bi_sparse --frac 0.5 \
    --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 1 --out $R/ls10_bisparse50
python3 $R/plot_misspec.py
```

### Perturbation rules

- **Spurious directed edges** are drawn only from node pairs where the parent precedes the
  child in the true topological order, so the perturbed graph stays acyclic and DCM's
  top-sort remains valid. The error is a *wrong parent set*, not a malformed graph.
- **Spurious confounders** are added only between pairs that do not already share one.
- **Removed confounders** are sampled uniformly from the existing groups.

## 3 What each arm tests (they are not symmetric)

**Adding edges is over-specification, not error.** If `P` is Markov with respect to `G` and
`G ⊆ G'`, then `P` is Markov with respect to `G'`. A denser graph — extra directed edges or
extra confounders — entails *fewer* conditional independences and therefore makes **no false
claims**. It costs parameters and estimation variance, not correctness.

**Removing bidirected edges is under-specification, and can genuinely bias.** In an ADMG,
adjacent nodes are never m-separated. Dropping `X↔Y` makes the pair separable, so the graph
entails a CI that (under faithfulness) the data violates — DCM is told a confounded pair is
unconfounded.

One nuance verified by enumerating d-separations on the canonical DAG: dropping `X↔Y` adds
**no** new CI when a directed edge between `X` and `Y` remains *and* no collider path through
the endpoints depended on the `↔`. Such removals are invisible to any CI-based method while
still changing the causal semantics. This arm mixes both kinds; the split is not separated
out here.

## 4 Reproducibility

| property | status |
|---|---|
| source datasets modified | **no** — opened read-only; 0 files touched |
| perturbed graph saved per instance | **yes** — `<out>/graphs/<mode>_<i>.yaml` |
| perturbation deterministic | **yes** — `rng = Random(1000 + i)`, verified same-seed→same-graph |
| arms paired | **yes** — instance `i` gets the matched seed in every arm, so McNemar applies |
| results overwritten | **no** — each arm writes its own `<out>/misspec_<mode>.csv` |
| results written incrementally | **yes** — one row per instance, usable before n=100 |
| per-instance DCM output | `<out>/runs/<mode>_<i>/` |

**Limit:** DCM training itself is unseeded (`BUG A1-20`), so re-running an arm reproduces the
*graph* exactly but not the fitted model. Run-to-run variation at n=100 is roughly ±0.05 —
the reason the comparisons below are paired rather than across independent runs.

## 5 Results

**n = 100 per arm, `wasserstein`, paired McNemar against the control** (all arms ran on the
same 100 datasets, so the pairing is exact).

| arm | graph given to DCM | perfect recovery | 95% CI | vs control (w/l) | p |
|---|---|---:|---|---:|---|
| **control** | true graph | **0.880** | [0.80, 0.93] | — | — |
| **dense** | +50% directed & bidirected | 0.840 | [0.76, 0.90] | 4 / 8 | 0.39 ns |
| **bi_sparse** | −50% bidirected | 0.870 | [0.79, 0.92] | 4 / 5 | 1.00 ns |
| **both_sparse** | −50% directed & bidirected | **0.760** | [0.67, 0.83] | 5 / 17 | **0.019 \*** |

Other metrics on the same runs:

| arm | wasserstein | mmd | flow | edges changed / instance |
|---|---:|---:|---:|---:|
| control | 0.880 | 0.900 | 0.460 | 0 |
| dense | 0.840 | 0.850 | 0.360 | 6.0 |
| bi_sparse | 0.870 | 0.900 | 0.430 | 2.0 |
| both_sparse | 0.760 | 0.770 | 0.480 | 6.0 |

### Reading

**Adding edges costs nothing** (0.88 → 0.84, p = 0.39). Four spurious directed edges and two
spurious confounders per instance leave accuracy statistically unchanged — as §3 predicts,
a supergraph remains Markov to the data, so over-specification makes no false claims.

**Dropping half the confounders costs nothing** (0.88 → 0.87, p = 1.00). Removing bidirected
edges decorrelates genuinely shared noise, but does not remove information the conditional
mechanism needs.

**Deleting directed edges is the one perturbation that hurts** (0.88 → 0.76, p = 0.019).
With half the directed edges gone, `P(V | pa(V))` is fit against an incomplete parent set and
a real parent's influence is absorbed into the noise term. Even so, RCA-DCM retains **0.76**.

Because `bi_sparse` and `both_sparse` delete the **same** confounders, the gap between them
isolates *directed-edge loss* as the sole failure mode — the confounder half contributes
nothing to the degradation.

### How the error bars were computed, and how to compare arms

The count itself is exact: 88 of 100 is 88%, with no uncertainty. The interval answers a
different question — *what is the accuracy on this class of problems?* The 100 instances are
an independent sample from the generator (100 distinct DAGs, 85 distinct root-cause sets), so
the rate is a binomial proportion and the interval is the **Wilson score** interval for it.
Wilson rather than `p ± 1.96·sqrt(p(1−p)/n)` because that textbook form breaks near 0 and 1
(it returns zero width at a rate of 1.00 and negative lower bounds near 0).

**Do not compare arms by their intervals.** `control` [0.80, 0.93] and `both_sparse`
[0.67, 0.83] overlap, yet the difference is significant (p = 0.019). The intervals treat the
arms as two independent samples, which they are not: every arm ran on the **same** datasets,
so 83 of the 100 instances agree and carry no information. The paired test uses only the 22
**discordant** pairs (5 vs 17) and is far more powerful. The intervals are for reporting a
single arm's accuracy; McNemar is for comparing arms.

One caveat on the two `ns` arms: DCM training is unseeded, and retraining on byte-identical
data and graphs flips about **11%** of instances (measured: 88.9% per-instance agreement
between two control runs, McNemar p = 0.29). So `dense` and `bi_sparse` are *not
distinguishable* from the control at n = 100 — a stronger claim than that would need seeding
or more runs. The `both_sparse` effect is larger than that noise floor.

## 6 Related evidence already in hand

- **Same-MEC misspecification** (`RCA/.../pag_to_admg`): 48 ADMGs from one PAG — identical CI
  structure, different directed/bidirected patterns. DCM 0.826 vs RCD 0.741, BARO 0.547
  (n=201, archived). ⚠️ A second copy of that `succ.json` at n=303 disagrees
  (DCM 0.680 < RCD 0.716); resolve before citing.
- **Maximal over-specification** (causal chamber, sink graph): every node treated as a parent.
  DCM 0.87 → 0.73 — consistent with the supergraph argument in §3.

## 7 Figure

![misspecification](out/misspec_wasserstein.png)

`out/misspec_wasserstein.pdf` — 2.3in × 1.35in, generated by `plot_misspec.py`; 95% Wilson
intervals, same style as the other paper panels. Green = true graph, light green =
over-specified, amber = confounders dropped, red = the arm that actually degrades.

Regenerate: `python3 plot_misspec.py [--metric mmd|flow]`.
Per-arm numbers land in `out/summary.csv`.
