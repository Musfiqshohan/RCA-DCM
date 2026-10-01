# Experiment 2: hidden true root cause, children as accepted answers

Variant of the causal-chamber experiment (`../causal_chamber/`) that assumes the true
intervened variable is **unobserved**. Every case still comes from
`lt_interventions_standard_v1`, but the true-cause column is removed from the data and
from the graph before anything sees it. Success is redefined accordingly: the true cause
can no longer be ranked at all, so a method is scored on whether it points at one of the
true cause's **direct children** in the real causal graph — the observable symptoms one
hop downstream of the hidden cause.

## What's different from Experiment 1

| | Exp 1 (`../causal_chamber/`) | Exp 2 (here) |
|---|---|---|
| True cause column | present in data, is the answer | **removed** from data; kept only as an unobserved confounder |
| Candidate nodes | all 38 | 37 (case-specific: whichever column is hidden) |
| Graph | full 38-node/57-edge real DAG | 37 directed nodes + the hidden column represented as an ADMG confounder over its former children (DCM only — see below) |
| "Correct" | top-k contains the true cause | top-k contains **any** of the true cause's direct children |
| Engine | `normal_once` + early stop | `normal_once` + early stop (same, standing default) |

## The hidden node is a confounder, not just a deletion

The true cause isn't merely dropped — its causal role doesn't disappear just because it's
unobserved. Its former children still share it as a common cause, so the graph handed to DCM
represents that explicitly as an ADMG confounder group, keyed by the hidden node's own name:

```yaml
confounders:
  blue:
  - current
  - ir_1
  - ir_2
  - ir_3
  - vis_1
  - vis_2
  - vis_3
```

i.e. exactly the `dag` structure of Experiment 1 minus the `blue` node itself, plus this one
`confounders` entry saying "these children have an unobserved common cause." This is DCM-RCA's
own point (per the project README: ADMGs, not DAGs — bidirected edges standing in for
confounders it doesn't assume away) — Experiment 2 is a direct test of that capability, not a
generic ablation.

**Baselines (nsigma/baro/rcd/circa/rcg) don't have an ADMG/confounder concept** — RCAEval's
`graph=` parameter is a plain directed graph. They get the same 37-node directed structure
(hidden column's edges simply absent) either way; only DCM's graph actually differs between
"dropped" and "confounder."

**Required a small fix to make N-way confounder groups work at all**: `dcm/functions/models.py`'s
`DCM_FlOW.__init__` previously assumed every confounder group has exactly 2 members
(`v1, v2 = pair`) — every existing example in this codebase (`dcm/dat_gen/graphs.yaml`) is
pairwise, so this was never exercised beyond 2. Generalized to a loop over the whole group
(2026-08-19); the rest of the class already summed an arbitrary number of confounder-noise
draws per node, so this was the only blocking line. Verified directly: a 3-node synthetic
confounder group and the real 7-child `blue` group both wire every listed child to the same
shared latent, unconfounded nodes get `[]`.

## Accepted root-cause set — how it's computed

For a case whose true (hidden) cause is `X`, the accepted set is `children(X)` read directly
from the real graph (`baselines/causal_chamber/ground_truth.py`'s edge list) — every node `Y`
such that `X -> Y` is a real edge. Every one of the 29 manipulable variables has at least one
child (verified directly, no degenerate empty sets), and the accepted set is always a subset
of the 9 sensor-output nodes (`ir_*`, `vis_*`, `angle_*`, `current`) since those are the only
nodes any source variable ever points to in this graph.

## Accuracy metric — stated explicitly, per the task's own requirement

```
acc@k(ranked, accepted_set) = 1  if ranked[:k] contains ANY member of accepted_set
                             = 0  otherwise
```

This is **top-k "at least one hit"**, the direct multi-label generalization of the
single-label `acc@k = int(true_cause in ranked[:k])` already used everywhere else in this
project (`run_sink_graph.py`, `baselines/causal_chamber/run_baselines_cc.py`). It is
deliberately **not** ASE'24's partial-credit `AC@k` (divides by `min(k, |accepted_set|)`,
counts every hit within top-k, not just whether there was one) — kept alongside it, not
replaced by it: `AC@k`/`Avg@k`/`PRR` were added afterward (see "Baseline results" below) as
the formally-specified metrics, while `acc@k` stays as the simpler, already-established one.
Implemented once each, in `hidden_config.py`, shared by both the baseline and DCM runners so
no definition can drift between them.

## Data: what's saved, and why

`hidden_config.build_all_hidden_datasets()` materializes, once per **unique** hidden column
(27 of them — several cases share a true cause at different intervention strengths, e.g.
`uniform_blue_mid.csv`/`uniform_blue_strong.csv` both hide `blue`):

```
data/<hidden_column>/
  reference_hidden.csv   # uniform_reference.csv with that column dropped (shared normal file)
  graph.yaml             # the 37-node ADMG, that column removed as node + as any child's parent
  uniform_<...>.csv      # each anomalous file that hides this column, column dropped
```

~63MB total, written once and reused by both `run_baselines_cc.py` and `run_dcm.py` (not
regenerated per run — `build_all_hidden_datasets()` is idempotent, just overwrites).

## How to reproduce

```bash
# From the repository root.

# 1. Baselines first (nsigma, baro, rcd, circa, rcg) -- fast, CPU-only.
#    Builds data/ automatically if not already present.
python3 baselines/causal_chamber_hidden_rc/run_baselines_cc.py
# -> baselines/causal_chamber_hidden_rc/out/baselines_results.csv

# 2. DCM (engine=normal_once, early_stop=True), real per-case graph.
#    --gpu_id is required (no default) -- this experiment used GPU1,
#    Experiment 1 (../causal_chamber/) used GPU0 concurrently.
python3 baselines/causal_chamber_hidden_rc/run_dcm.py --gpu_id 1
# -> baselines/causal_chamber_hidden_rc/out/dcm_results.csv
```

Both scripts print a summary table (Acc@1/3/5, error/timeout counts) at the end and write
incrementally, so a partial run is never lost. All output stays under this folder's own
`out/` — nothing is written to `dcm/out/`.

## Baseline results (n=52, 0 errors)

Acc@k: at least one accepted answer in top-k. AC@k/Avg@k/PRR: the ASE'24-style metrics —
see `../causal_chamber/README.md`'s "Metrics" section for the exact formulas (`A`, `R^a[i]`,
`V_rc^a` notation); this is the experiment where they actually diverge from `acc@k`, since
`|V_rc^a|` ranges from 1 to 7 here (not always 1).

| method | Acc@1 | Acc@3 | Acc@5 | AC@1 | AC@3 | AC@5 | Avg@5 | PRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **rcd** | **0.904** | 0.904 | 0.904 | **0.904** | 0.849 | 0.846 | 0.863 | **0.692** |
| circa | 0.846 | 0.942 | 0.962 | 0.846 | 0.897 | 0.935 | 0.895 | 0.731 |
| rcg | 0.654 | 0.731 | 0.827 | 0.654 | 0.673 | 0.801 | 0.696 | 0.500 |
| nsigma | 0.327 | 0.404 | 0.423 | 0.327 | 0.314 | 0.362 | 0.333 | 0.135 |
| baro | 0.288 | 0.404 | 0.423 | 0.288 | 0.324 | 0.369 | 0.328 | 0.135 |

Two things worth reading carefully here, both real properties of the formula, not artifacts:

- **`PRR` ranks methods differently than every other column** (`circa` 0.731 > `rcd` 0.692 >
  `rcg` 0.500), even though `rcd` leads on `Acc@k`/`AC@1`. `PRR` requires *exact* recovery of
  the whole accepted set at `k = |V_rc^a|` (varies 1-7 per case) — a method that reliably
  finds *one* correct symptom early (`rcd`'s pattern — see below) but not the *rest* of a
  multi-child accepted set scores worse here than one that's merely very good across the
  whole top-k (`circa`).
- **`AC@k` is not monotonic in `k`** — `nsigma`'s `AC@3` (0.314) is *lower* than its `AC@1`
  (0.327). This is correct: `AC@k`'s denominator is `min(k, |V_rc^a|)`, which grows with `k`
  too, so a method that doesn't add proportionally more correct hits between rank 1 and rank
  3 can see its score dip even while `acc@k` (which only asks "at least one hit," and can
  only go up) keeps climbing. Don't read a small `AC@k` dip as a bug.

`rcd`'s `PRR` (0.692) — the highest *among baselines* despite the reordering above (DCM
later surpasses it, see "DCM vs. all baselines" below) — inherits the same short-ranking
behavior as Experiment 1 (29/52 cases return only 1 candidate): when the accepted set also
happens to have exactly 1 member (a hidden node with a single child) and `rcd`'s one guess
is right, `PRR` is trivially satisfied. Check `accepted_rc`'s length alongside `rcd_top5`'s
length if this needs auditing further.

## DCM vs. all baselines

**Final** — both the DCM run (`engine=normal_once`, `early_stop=True`, `max_epc=50`,
case-specific real graph with the hidden node as an ADMG confounder, GPU1) and all 5
baselines are complete (n=52, 0 errors each).

### Aggregate

| method | AC@1 | AC@3 | AC@5 | Avg@5 | PRR |
|---|---:|---:|---:|---:|---:|
| **DCM** | 0.865 | 0.904 | 0.933 | 0.894 | **0.846** |
| rcd | **0.904** | 0.849 | 0.846 | 0.863 | 0.692 |
| circa | 0.846 | 0.897 | **0.935** | **0.895** | 0.731 |
| rcg | 0.654 | 0.673 | 0.801 | 0.696 | 0.500 |
| nsigma | 0.327 | 0.314 | 0.362 | 0.333 | 0.135 |
| baro | 0.288 | 0.324 | 0.369 | 0.328 | 0.135 |

DCM **leads every method on `PRR`** (0.846 vs. `circa`'s 0.731) by a wide margin — the metric
that requires recovering the *entire* accepted set exactly, which is where DCM's
confounder-aware modeling matters most. It also beats `circa` on `AC@1` (0.865 vs 0.846),
though `rcd` still edges both on raw `AC@1` (0.904); `circa` narrowly leads `AC@5`/`Avg@5`.
Reading across all 5 columns: DCM is strongest or co-strongest on 3 of 5, not a clean sweep,
but the clear leader on the metric that matters most for this task (exact recovery).

### Per variable family (AC@1 / PRR)

Same family grouping as `../causal_chamber/README.md` (by the *hidden* variable's role, not
its children's). n=52/52 both DCM and baselines.

| family | n | DCM AC@1 | rcd | circa | rcg | nsigma | baro | DCM PRR | rcd PRR | circa PRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| color (R/G/B) | 6 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| polarizer angle | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 1.000 |
| position lights (L) | 6 | 0.167 | 0.167 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| diode calibration (D) | 9 | 1.000 | 1.000 | 1.000 | 0.667 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| sensor gain (T) | 18 | 1.000 | 1.000 | 1.000 | 0.667 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| angle/current calib (O/R) | 9 | 0.778 | **1.000** | 0.778 | 0.667 | 0.778 | 0.778 | 0.778 | **1.000** | 0.778 |

Two clear, honest patterns, not uniform DCM dominance:
- **`position lights (L)` is hard for every method** (DCM/`rcd`/`circa` all 0.167, `rcg`/
  `nsigma`/`baro` all 0.000, PRR 0.000 across the board) — the 6 `l_11`...`l_32` variables'
  children apparently aren't distinctive enough from other position lights' children for
  anyone to reliably tell them apart once the true cause is hidden. This is the family
  dragging down every method's aggregate score, DCM included.
- **`angle/current calib (O/R)` is the one family where `rcd` clearly beats DCM** (1.000 vs.
  0.778 on both `AC@1` and `PRR`) — the only family in either experiment where a baseline
  outright wins. Everywhere else DCM matches or leads.

## DCM with a sink graph instead of the real graph (ablation)

Same data (hidden column dropped), same evaluation, same engine (`normal_once`,
`early_stop=True`, `max_epc=50`) — **only the graph changes**: no real edges or confounder at
all, just the sink-graph convention (every other node becomes the candidate's parent, per
node under test — see `run_dcm.py --sink`). Run split across both GPUs (26 cases each),
n=52 total, 0 errors. Output kept separate: `out/dcm_results_sink.csv`.

| | AC@1 | AC@3 | AC@5 | Avg@5 | PRR |
|---|---:|---:|---:|---:|---:|
| **DCM, real graph** | 0.865 | 0.904 | 0.933 | 0.894 | 0.846 |
| **DCM, sink graph** | 0.731 | 0.792 | 0.834 | 0.789 | 0.654 |
| Δ (sink − real) | −0.134 | −0.112 | −0.099 | −0.105 | −0.192 |

The real graph (with the confounder representation) helps substantially and consistently —
worst on `PRR` (−0.192), the metric requiring exact recovery of the whole accepted set, which
is exactly where knowing the true structure (rather than conditioning on all 36 other nodes
indiscriminately) should matter most.

**The degradation is not spread evenly — it's almost entirely concentrated in one family:**

| family | AC@1 sink | AC@1 real | Δ | PRR sink | PRR real | Δ |
|---|---:|---:|---:|---:|---:|---:|
| color (R/G/B) | 0.000 | 1.000 | **−1.000** | 0.000 | 1.000 | **−1.000** |
| polarizer angle | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | **−1.000** |
| position lights (L) | 0.000 | 0.167 | −0.167 | 0.000 | 0.000 | 0.000 |
| diode calibration (D) | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| sensor gain (T) | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| angle/current calib (O/R) | 0.778 | 0.778 | 0.000 | 0.778 | 0.778 | 0.000 |

`color (R/G/B)` collapses completely without the real graph (1.000 → 0.000 on both metrics) —
each of `red`/`green`/`blue` causally affects **7** children (`current` + all 6 `ir_*`/`vis_*`
sensors, the broadest reach of any family). With a sink graph, DCM conditions each candidate
on all 36 *other* nodes regardless of whether they're real parents, which dilutes exactly the
broad, diffuse signal this family produces. `polarizer angle`'s `AC@1` is unaffected (still
1.000) but its `PRR` collapses (1.000 → 0.000) — top-1 stays right, but the sink graph stops
DCM from getting the *entire* 3-member accepted set (`angle_2`,`ir_3`,`vis_3` for `pol_2`,
similarly for `pol_1`) exactly right. Every other family — the ones with narrower causal
reach (1-2 children) — is **completely unaffected** by dropping the real graph. This is a
clean, localized effect, not a diffuse one.

## Intuition: why the ranking flips relative to Experiment 1

Compare against `../causal_chamber/README.md`'s table for the same 5 methods on the
*non*-hidden task: there, `rcg` was perfect (1.000) and `circa` was weak (0.135); here
`circa` (0.846) clearly beats `rcg` (0.654), and `nsigma`/`baro` jump roughly 5-6x (0.058 →
0.29-0.33). None of this is noise — each shift traces to a specific mechanism of what
"hiding the cause" actually changes:

- **`rcg` drops the most (1.000 → 0.654) because its whole method depends on conditioning on
  the *correct* parent set, and that set is now deliberately wrong.** `rcg` scores a
  candidate by mutual information with the fault indicator *given that candidate's parents in
  the supplied graph*. In Experiment 1, sink nodes' parent sets were complete and exact, so
  the conditioning was exact. Here, a sink node like `ir_1`'s parent set has had the true
  cause (e.g. `blue`) surgically removed — `rcg` is now conditioning on an intentionally
  incomplete parent set, which is precisely the condition its exactness advantage relied on.

- **`circa` improves the most in relative terms (0.135 → 0.846) because its regression-residual
  test doesn't need the missing parent to be *in* the model to notice its absence.** `circa`
  asks "how anomalous is this node's residual, given its (possibly incomplete) parents." When
  `blue` is the omitted variable and every *other* parent of `ir_1` (red, green, `l_11`,
  `l_12`, `t_ir_1`, `diode_ir_1`) is untouched by this intervention, the shift caused by
  `blue` shows up entirely in the unexplained residual — `circa`'s test still flags it, even
  without `blue` in the model. Its Experiment-1 weakness (a linear-regression functional-form
  mismatch against nonlinear optics) matters less here because the *target* set is broader (up
  to 7 acceptable answers instead of 1), giving a biased-but-still-anomalous-looking node more
  chances to land in top-k.

- **`nsigma`/`baro` jump ~5-6x for a much blunter reason: the target set changed to match their
  own bias, not because they got smarter.** Experiment 1's README noted these marginal,
  graph-free methods systematically over-rank downstream *symptoms* over the true upstream
  *cause*, because symptoms show larger raw marginal shifts — that was their failure mode
  there. In Experiment 2, the accepted answers **are** those same downstream symptoms. The
  exact same ranking behavior that looked like a systematic error before now scores as
  correct — this is a property of the redefined task, not evidence the methods reason about
  hidden confounding at all.

- **`rcd` is essentially unchanged (0.904 in both experiments) and flat across k=1/3/5.**
  Consistent with a discovery-based method that either correctly localizes the anomalous local
  neighborhood on the first try or misses it outside the top-5 entirely — a graded/blurred
  ranking isn't really part of its failure mode either way, so hiding the cause (which mostly
  changes *which* answer is "correct" among neighbors `rcd` was already finding) barely moves
  its score.
