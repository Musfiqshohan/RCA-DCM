#!/bin/bash
# PAPER EXPERIMENT 1 -- Nonlinear model with unobserved confounders.
# Data setup replicated from RCA/baselines/LIT/data/lit_random_v10_lat4_none_latnstrg{3,10}:
#   10 vars = 6 observed + 4 latent, --intv_wgt_func none, obs_interv_perc 0.5 -> 3 root causes.
# Experiment configuration is the repository's: DCM engine='original' with EARLY STOPPING
# (--max_epc 50), and the headline metric read off `wasserstein` rather than the `mmd`
# used by the original runs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
V=python3
R=baselines/paper_experiments/exp1_confounding
for LS in 3 10; do
  echo "=== exp1 latent_edge_strength=$LS ==="
  $V baselines/LIT/rca_in_lit_.py \
    --num_vars 10 --latent_vars_perc 0.4 --intv_wgt_func none \
    --latent_edge_strength $LS \
    --methods dcm_flow nsigma baro circa rcd rcg \
    --early_stop --max_epc 50 \
    --num_obs_normal 5000 --num_obs_anomalous 5000 \
    --data_root $R/ls$LS/data --out_root $R/ls$LS/dcm_out \
    --thr_num 6 --gpu_id 0 --tot 100 \
    > logs/paper_exp1_ls$LS.log 2>&1
  echo "  done ls=$LS"
done
echo GPU0_ALL_DONE
