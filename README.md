# RCA-DCM

Root-cause analysis with deep causal models on acyclic directed mixed graphs.
For each candidate node the method ties that node's mechanism across a normal
regime and an anomalous regime, then scores the node by how far apart the two
generated distributions remain. A higher score means a more likely root cause.

Python 3.10. A GPU is optional. The published synthetic runs used CPU. The
causal-chamber runs used a GPU.

## Start here

```bash
pip install -r requirements.txt
bash examples/run_sample.sh
```

That scores the one Sock Shop case included in this repository
(`carts` under CPU stress, replicate 1). It uses the published microservice
settings: sink-confounded star, original engine, early stopping, at most 50
epochs. On the machine that produced the tables this case took about 15 seconds.

When it finishes, open:

```text
dcm/out/sockshop_sample/cpu/carts_cpu_r1/root_cause_results.csv
```

The first row is the predicted root cause. The injected service is `carts`.
A ranking from the published run of the same case is already in
`examples/results/sockshop_carts_cpu_r1/root_cause_results.csv`, where `carts`
is ranked first.

Training is unseeded, so your scores will not match that file digit for digit.
The ordering is what to look at.

## What the data and the output look like

`examples/` holds a few real files. Open those before downloading anything else.
`examples/README.md` says what each column means.

| you want to see | open |
|---|---|
| A synthetic normal / anomalous pair and the true causes | `examples/synthetic/lambda10_one_graph/` |
| The ranking for that graph (`X4`, `X2`, `X6`, which is exact) | `examples/results/synthetic_lambda10/root_cause_results.csv` |
| One Sock Shop time series | `examples/microservice/sock-shop-2/carts_cpu/1/simple_data.csv` |
| One Online Boutique time series | `examples/microservice/online-boutique/currencyservice_delay/2/data.csv` |
| One causal-chamber reference and one intervention | `examples/causal_chamber/` |
| The 125-case tables behind the numbers below | `examples/results/*_results.csv` |

## Published results

PRR here is exact set recovery: the top-|R\*| nodes equal the true root-cause
set. On the microservice cases there is one true cause, so that PRR equals
Acc@1. The plain-sink PRR column further down is a different, rank-based score
stored in those CSVs; it is labeled as such.

### Latent confounding (synthetic)

6 observed variables, 4 latent confounders, 3 root causes, 100 replicates,
fixed 50 epochs, Wasserstein. λ is the latent-edge strength.

| method | PRR, λ = 3 | PRR, λ = 10 |
|---|---:|---:|
| **RCA-DCM** | **0.99** | **0.84** |
| BARO | 0.91 | 0.65 |
| NSigma | 0.86 | 0.62 |
| RCG | 0.85 | 0.39 |
| RCD | 0.74 | 0.59 |
| CIRCA | 0.62 | 0.38 |

Source: `baselines/paper_experiments/exp1_confounding/out/summary.csv`.
Write-up: `baselines/paper_experiments/exp1_confounding/README.md`.

### Heterogeneous anomaly (synthetic)

No latent confounders. n = 5 has 2 root causes. n = 10 has 5.

| method | PRR, n = 5 | PRR, n = 10 |
|---|---:|---:|
| **RCA-DCM** | **0.98** | **0.54** |
| RCG | 0.88 | 0.36 |
| CIRCA | 0.85 | 0.34 |
| BARO | 0.66 | 0.14 |
| NSigma | 0.65 | 0.14 |
| RCD | 0.65 | 0.28 |

Source: `baselines/paper_experiments/exp2_heterogeneous/out/summary.csv`.

### Graph misspecification

Same λ = 10 data as above. Only the graph handed to RCA-DCM changes.

| graph given to RCA-DCM | PRR |
|---|---:|
| True graph | 0.88 |
| 50% extra directed and bidirected edges | 0.84 |
| 50% of the bidirected edges removed | 0.87 |
| 50% of directed and bidirected edges removed | 0.76 |

Source: `baselines/paper_experiments/exp_misspec/out/summary.csv`.

### Causal chamber (light tunnel, real hardware)

52 interventions. Observed: the true cause is a measured variable. Hidden: the
true cause is treated as an unobserved confounder, and a hit is any of its
children. RCA-DCM on the hidden task is the fixed-50-epoch run
(`dcm_results_noES.csv`).

| setting | method | AC@1 | AC@3 | AC@5 | PRR |
|---|---|---:|---:|---:|---:|
| Observed | **RCG** | **1.000** | 1.000 | 1.000 | **1.000** |
| Observed | RCD | 0.904 | 0.923 | 0.923 | 0.904 |
| Observed | RCA-DCM | 0.885 | 1.000 | 1.000 | 0.885 |
| Observed | CIRCA | 0.135 | 0.462 | 0.635 | 0.135 |
| Observed | NSigma | 0.058 | 0.212 | 0.308 | 0.058 |
| Observed | BARO | 0.058 | 0.135 | 0.192 | 0.058 |
| Hidden | RCD | **0.904** | 0.849 | 0.846 | 0.692 |
| Hidden | **RCA-DCM** | 0.865 | 0.904 | 0.933 | **0.846** |
| Hidden | CIRCA | 0.846 | 0.897 | **0.935** | 0.731 |
| Hidden | RCG | 0.654 | 0.673 | 0.801 | 0.500 |
| Hidden | NSigma | 0.327 | 0.314 | 0.362 | 0.135 |
| Hidden | BARO | 0.288 | 0.324 | 0.369 | 0.135 |

Per-case rows: `examples/results/causal_chamber/` and
`examples/results/causal_chamber_hidden/`. On the observed task RCG recovers
every case. On the hidden task RCA-DCM leads on exact set recovery (PRR 0.846).

### Microservices

125 cases each (5 services × 5 faults × 5 replicates). RCA-DCM uses the
sink-confounded star, original engine, early stopping, at most 50 epochs,
Wasserstein. CIRCA and RCG use the call graph in
`baselines/RCAEval/call_graphs.py`. NSigma, BARO, and RCD do not use a graph.

| dataset | method | Acc@1 | Acc@3 | Acc@5 | mean sec/case |
|---|---|---:|---:|---:|---:|
| Sock Shop | **RCA-DCM** | **0.888** | 0.992 | 0.992 | 13.7 |
| Sock Shop | NSigma | 0.752 | 0.968 | 0.992 | |
| Sock Shop | BARO | 0.744 | 0.984 | 0.992 | |
| Sock Shop | CIRCA | 0.648 | 0.960 | 0.992 | |
| Sock Shop | RCG | 0.536 | 0.672 | 0.704 | |
| Sock Shop | RCD | 0.384 | 0.544 | 0.584 | |
| Online Boutique | **RCA-DCM** | **0.776** | 0.896 | 0.976 | 29.6 |
| Online Boutique | RCG | 0.712 | 0.832 | 0.848 | |
| Online Boutique | NSigma | 0.704 | 0.912 | 0.960 | |
| Online Boutique | BARO | 0.616 | 0.904 | 0.944 | |
| Online Boutique | CIRCA | 0.520 | 0.888 | 0.984 | |
| Online Boutique | RCD | 0.488 | 0.592 | 0.616 | |

Per fault, top-1 / top-3 / top-5, sink-confounded RCA-DCM:

| dataset | CPU | MEM | DISK | DELAY | LOSS | average |
|---|---|---|---|---|---|---|
| Sock Shop | 1.00 / 1.00 / 1.00 | 0.96 / 0.96 / 0.96 | 0.76 / 1.00 / 1.00 | 0.92 / 1.00 / 1.00 | 0.80 / 1.00 / 1.00 | 0.89 / 0.99 / 0.99 |
| Online Boutique | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.64 / 0.88 / 1.00 | 0.80 / 1.00 / 1.00 | 0.44 / 0.60 / 0.88 | 0.78 / 0.90 / 0.98 |

The plain sink graph (no confounders) gives the same top-1 service as the
sink-confounded star on all 125 cases of each dataset. Its aggregate Acc@1 is
0.880 on Sock Shop and 0.776 on Online Boutique. The rank-based score in those
plain-sink CSVs averages 0.976 (Sock Shop) and 0.945 (Online Boutique).

Case-level files:

- `examples/results/sockshop_sink_confounded_results.csv`
- `examples/results/onlineboutique_sink_confounded_results.csv`
- `examples/results/sockshop_baselines_results.csv`
- `examples/results/onlineboutique_baselines_results.csv`
- `examples/results/sockshop_sink_results.csv`
- `examples/results/onlineboutique_sink_results.csv`

## Reproduce the tables

Run every command from this directory. A fresh training run lands within about
±0.05 of the table, because training is unseeded. Comparisons in the paper are
paired on the same draws.

Redraw the figures from the summary CSVs already in the repo, without training:

```bash
python3 baselines/paper_experiments/plot_row.py
python3 baselines/paper_experiments/exp_misspec/plot_misspec.py
```

### 1. Latent confounding

```bash
bash baselines/paper_experiments/run_rerun_noES.sh
```

One λ on its own:

```bash
python3 baselines/LIT/rca_in_lit_.py \
  --num_vars 10 --latent_vars_perc 0.4 --intv_wgt_func none \
  --latent_edge_strength 10 \
  --methods dcm_flow nsigma baro circa rcd rcg \
  --num_epc 50 \
  --num_obs_normal 5000 --num_obs_anomalous 5000 \
  --data_root baselines/paper_experiments/exp1_confounding/ls10_noES/data \
  --out_root  baselines/paper_experiments/exp1_confounding/ls10_noES/dcm_out \
  --thr_num 6 --gpu_id 0 --tot 100
```

This generates its own data under `--data_root`. One replicate of that data is
already in `examples/synthetic/lambda10_one_graph/`.

### 2. Heterogeneous anomaly

```bash
bash baselines/paper_experiments/run_v5_floor.sh
bash baselines/paper_experiments/run_v10_noES.sh
```

### 3. Graph misspecification

Run experiment 1 at λ = 10 first. These commands only change the graph.

```bash
D=baselines/paper_experiments/exp1_confounding/ls10_noES/data/lit_random_v10_lat4_none_latnstrg10.0_Ns5000As5000
R=baselines/paper_experiments/exp_misspec
python3 $R/run_misspec.py --data_root "$D" --mode true        --frac 0   --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_control
python3 $R/run_misspec.py --data_root "$D" --mode dense       --frac 0.5 --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_dense50
python3 $R/run_misspec.py --data_root "$D" --mode bi_sparse   --frac 0.5 --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_bisparse50
python3 $R/run_misspec.py --data_root "$D" --mode both_sparse --frac 0.5 --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_bothsparse50
python3 $R/plot_misspec.py
```

### 4. Causal chamber

Download `lt_interventions_standard_v1` and point `CAUSAL_CHAMBER_DATA` at the
folder that contains the CSVs. Two of those CSVs are in `examples/causal_chamber/`
so you can see the columns. The commands below need the full set.

```bash
export CAUSAL_CHAMBER_DATA=/path/to/lt_interventions_standard_v1/data

python3 baselines/causal_chamber/run_baselines_cc.py
python3 baselines/causal_chamber/run_dcm.py --gpu_id 0 --thr_num 12

python3 baselines/causal_chamber_hidden_rc/run_baselines_cc.py
python3 baselines/causal_chamber_hidden_rc/run_dcm.py --gpu_id 0 --thr_num 12 \
  --no_early_stop --num_epc 50 \
  --out baselines/causal_chamber_hidden_rc/out/dcm_results_noES.csv

python3 baselines/paper_experiments/causal_chamber/build_summary.py --dcm_hidden dcm_results_noES.csv
python3 baselines/paper_experiments/causal_chamber/plot_cc.py
```

The case list is `baselines/causal_chamber/causal_chamber_datasets.yaml`.
The hidden-cause runner writes reduced tables under
`baselines/causal_chamber_hidden_rc/data/` from those CSVs.

### 5. Sock Shop and Online Boutique

Download `sock-shop-2` and `online-boutique` (RCAEval, Zenodo record 13305663)
and unpack them under one directory. The sample cases already in
`examples/microservice/` use that same layout.

```bash
export RCAEVAL_DATA_ROOT=/path/to/rcaeval_data

python3 run_sink_graph.py --dataset sockshop --sink_confounded --engine original \
  --early_stop --max_epc 50 --num_epc 50 --thr_num 14 --tag sink_confounded
python3 run_sink_graph.py --dataset onlineboutique --sink_confounded --engine original \
  --early_stop --max_epc 50 --num_epc 50 --thr_num 12 --tag sink_confounded

python3 baselines/RCAEval/run_baselines.py --dataset sockshop
python3 baselines/RCAEval/run_baselines.py --dataset onlineboutique
python3 score_sink.py sockshop onlineboutique
```

`RCAEVAL_DATA_ROOT` defaults to `data/rcaeval` inside this repository.
`score_sink.py` prints Acc@k from `dcm/out/{dataset}_sink_results.csv` (plain sink graph).
The sink-confounded numbers in the table above come from
`dcm/out/{dataset}_sink_confounded_results.csv`, which `run_sink_graph.py` writes
when you pass `--tag sink_confounded`.

## Where things live

```text
examples/                    a few datasets and the published output tables
dcm/                         RCA-DCM (training, flows, graphs)
run_sink_graph.py            microservice runner
score_sink.py                Acc@k tables from a results CSV
baselines/LIT/               synthetic generator and experiment driver
baselines/RCAEval/           NSigma, BARO, CIRCA, RCD, RCG
baselines/causal_chamber/    causal chamber, observed root cause
baselines/causal_chamber_hidden_rc/   causal chamber, hidden root cause
baselines/paper_experiments/ launch scripts, plots, summary CSVs
```
