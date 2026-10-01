# RCA-DCM

Root-cause analysis with deep causal models on acyclic directed mixed graphs.
Each candidate is scored by tying its mechanism across a normal regime and an
anomalous regime and measuring how far apart the two generated distributions
remain.

Python 3.10. A GPU is optional; the published runs used CPU for the synthetic
experiments and a GPU for the causal chamber.

```bash
pip install -r requirements.txt
```

Run every command below from this directory. Synthetic experiments generate
their own data. Real datasets are not included.

Aggregated result tables are already in `baselines/paper_experiments/**/out/summary.csv`
(and the misspecification arm CSVs). Regenerate the figures without retraining:

```bash
python3 baselines/paper_experiments/plot_row.py
python3 baselines/paper_experiments/exp_misspec/plot_misspec.py
```

Training is unseeded, so a fresh run matches the tables only up to run-to-run
noise of roughly ±0.05. Comparisons in the paper are paired on the same draws.

---

## 1. Latent confounding (synthetic)

6 observed variables, 4 latent confounders, 3 root causes, 5000 + 5000 samples,
100 replicates, fixed 50 epochs (no early stopping), Wasserstein score.
Reported settings: λ = 3 and λ = 10.

```bash
bash baselines/paper_experiments/run_rerun_noES.sh
```

Equivalent command for one λ:

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

Details: `baselines/paper_experiments/exp1_confounding/README.md`.

## 2. Heterogeneous anomaly (synthetic)

No latent confounders. Intervention weight `linear_dec`. Fixed 50 epochs.
n = 5 uses 2 root causes; n = 10 uses 5.

```bash
bash baselines/paper_experiments/run_v5_floor.sh
bash baselines/paper_experiments/run_v10_noES.sh
```

Details: `baselines/paper_experiments/exp2_heterogeneous/README.md`.

## 3. Graph misspecification

Reuses the λ = 10 datasets from experiment 1 (read-only) and changes only the
graph given to RCA-DCM. Run experiment 1 for λ = 10 first.

```bash
D=baselines/paper_experiments/exp1_confounding/ls10_noES/data/lit_random_v10_lat4_none_latnstrg10.0_Ns5000As5000
R=baselines/paper_experiments/exp_misspec
python3 $R/run_misspec.py --data_root "$D" --mode true       --frac 0   --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_control
python3 $R/run_misspec.py --data_root "$D" --mode dense      --frac 0.5 --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_dense50
python3 $R/run_misspec.py --data_root "$D" --mode bi_sparse  --frac 0.5 --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_bisparse50
python3 $R/run_misspec.py --data_root "$D" --mode both_sparse --frac 0.5 --graph_name lit_random_v10_lat4 --num_epc 50 --limit 100 --gpu_id 0 --out $R/ls10_bothsparse50
python3 $R/plot_misspec.py
```

Details: `baselines/paper_experiments/exp_misspec/README.md`.

## 4. Causal chamber

Place the public `lt_interventions_standard_v1` CSV files in a directory and point
`CAUSAL_CHAMBER_DATA` at it. The case list is
`baselines/causal_chamber/causal_chamber_datasets.yaml`.

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

The hidden-cause runner writes reduced tables under
`baselines/causal_chamber_hidden_rc/data/` from the CSVs above. Those tables
are generated, not shipped.

## 5. Microservices (Sock Shop, Online Boutique)

Download `sock-shop-2` and `online-boutique` (RCAEval release, Zenodo record
13305663) and unpack them so both directories sit under `RCAEVAL_DATA_ROOT`.

```bash
export RCAEVAL_DATA_ROOT=/path/to/rcaeval_data

python3 run_sink_graph.py --dataset sockshop --sink_confounded --engine original \
  --early_stop --max_epc 50 --num_epc 50 --thr_num 14
python3 run_sink_graph.py --dataset onlineboutique --sink_confounded --engine original \
  --early_stop --max_epc 50 --num_epc 50 --thr_num 12

python3 baselines/RCAEval/run_baselines.py --dataset sockshop
python3 baselines/RCAEval/run_baselines.py --dataset onlineboutique
python3 score_sink.py sockshop onlineboutique
```

RCA-DCM receives a sink confounded star. CIRCA and RCG receive the call graph
in `baselines/RCAEval/call_graphs.py`. NSigma, BARO, and RCD do not use a graph.

---

## Layout

```
dcm/                         RCA-DCM library (dcm_rca.py, training, flows)
run_sink_graph.py            microservice RCA-DCM runner
score_sink.py                Acc@k / PRR tables
baselines/LIT/               synthetic ADMG generator and experiment driver
baselines/RCAEval/           NSigma, BARO, CIRCA, RCD, RCG
baselines/causal_chamber/    causal chamber, observed root cause
baselines/causal_chamber_hidden_rc/   causal chamber, hidden root cause
baselines/paper_experiments/ launch scripts, plots, aggregated summaries
```
