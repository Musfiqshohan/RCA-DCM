# Sample data and outputs

These files are a few real cases from the experiments, small enough to open in
any editor. They show the column layout. They are not the full datasets.

## Data

### Synthetic graph (`synthetic/lambda10_one_graph/`)

One replicate from the latent-confounding experiment at λ = 10
(6 observed variables, 4 latent confounders, 3 root causes).

| file | what it is |
|---|---|
| `cont_nrm.csv` | 5000 normal samples. Columns `X1` … `X6`. |
| `cont_anm.csv` | 5000 anomalous samples, same columns. |
| `graph.yaml` | The ADMG: directed edges under `dag`, latent confounders under `confounders`. |
| `true_rc.json` | Ground-truth root causes. Here `{2, 4, 6}`, meaning `X2`, `X4`, `X6`. |

### Sock Shop (`microservice/sock-shop-2/carts_cpu/1/`)

One CPU-stress replicate. This is the layout `run_sink_graph.py` expects:
`<service>_<fault>/<replicate>/`.

| file | what it is |
|---|---|
| `simple_data.csv` | Time series. `time`, then `<service>_<metric>` (`carts_cpu`, `carts_mem`, …). |
| `inject_time.txt` | Unix time when the fault starts. Rows with `time` below this are normal. |

### Online Boutique (`microservice/online-boutique/currencyservice_delay/2/`)

Same idea. The CSV is named `data.csv`. The fault is a delay on `currencyservice`.

### Causal chamber (`causal_chamber/`)

Two CSVs from the public light-tunnel dataset `lt_interventions_standard_v1`.

| file | what it is |
|---|---|
| `uniform_reference.csv` | Reference (normal) run. |
| `uniform_blue_mid.csv` | One intervention run. `intervention` names the setting; sensor columns (`red`, `green`, `blue`, `ir_*`, `vis_*`, …) are the measurements. |

A full causal-chamber run needs the reference file plus every intervention CSV.
These two files are the format only.

## Outputs

### One Sock Shop ranking (`results/sockshop_carts_cpu_r1/`)

What one case writes after training.

| file | what it is |
|---|---|
| `root_cause_results.csv` | The ranking. Column `node`, score `tvd_diff`. Highest first. On this case `carts` is first, which is the injected service. |
| `all_ranking.csv` | The same case under every metric (`mmd`, `flow`, `wasserstein`, `baro`) and every aggregation. `r1_node` is rank 1. |
| `all_metrics.csv` | Acc@k and the rank score for this one case. |
| `stopping.csv` | Epoch where each candidate stopped. |
| `graph.yaml` | The sink-confounded graph built for this case. |

### One synthetic ranking (`results/synthetic_lambda10/`)

The matching λ = 10 replicate. `root_cause_results.csv` ranks `X4`, `X2`, `X6`
first. That set is exactly `true_rc.json`, so this replicate is a perfect recovery.

### Tables behind the README numbers (`results/`)

| file | covers |
|---|---|
| `sockshop_sink_results.csv`, `onlineboutique_sink_results.csv` | Plain sink graph, 125 cases each. |
| `sockshop_sink_confounded_results.csv`, `onlineboutique_sink_confounded_results.csv` | Sink-confounded star. |
| `*_baselines_results.csv` | NSigma, BARO, CIRCA, RCD, RCG. |
| `*_sink_confounded_vs_baselines.csv` | RCA-DCM and the baselines on the same cases. |
| `causal_chamber/dcm_results.csv` | Observed root cause. |
| `causal_chamber_hidden/dcm_results_noES.csv` | Hidden root cause, fixed 50 epochs. |
| `causal_chamber*/baselines_results.csv` | Baselines on the chamber. |

Each microservice results row is one case: `fault`, `case`, `replicate`, `true`,
then `wasserstein_add_acc@1` and `wasserstein_add_top1`. The published metric is
Wasserstein with the `add` aggregation. `wasserstein_add_top1` is the service
ranked first.
