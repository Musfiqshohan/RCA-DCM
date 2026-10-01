#!/bin/bash
# PAPER EXPERIMENT 2 -- Nonlinear model with heterogeneous anomaly.
# Data setup replicated from RCA/baselines/LIT/data/lit_random_v{5,10,15}_lat0_linear_dec:
#   no latent confounders, --intv_wgt_func linear_dec (this is what the stored runs
#   actually used; the paper prose describing "deeper nodes get larger shifts" is
#   linear_INC and does not match the data on disk), obs_interv_perc 0.5 -> (n+1)/2
#   root causes.
# Experiment configuration is the repository's: DCM engine='original' with EARLY STOPPING
# (--max_epc 50), headline metric `wasserstein` rather than the `mmd` of the old runs.
# Smallest graph first so results start landing early.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
V=python3
R=baselines/paper_experiments/exp2_heterogeneous
for NV in 5 10 15; do
  echo "=== exp2 num_vars=$NV ==="
  $V baselines/LIT/rca_in_lit_.py \
    --num_vars $NV --latent_vars_perc 0 --intv_wgt_func linear_dec \
    --methods dcm_flow nsigma baro circa rcd rcg \
    --early_stop --max_epc 50 \
    --num_obs_normal 5000 --num_obs_anomalous 5000 \
    --data_root $R/v$NV/data --out_root $R/v$NV/dcm_out \
    --thr_num 6 --gpu_id 1 --tot 100 \
    > logs/paper_exp2_v$NV.log 2>&1
  echo "  done v=$NV"
done
echo GPU1_ALL_DONE
