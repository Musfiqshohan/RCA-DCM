#!/bin/bash
# EXP 2, v5 -- WITHOUT early stopping AND with floor(n/2) = 2 root causes.
#
# Root-cause count: num_vars_intervened = int((n - n_latent + 1) * obs_interv_perc).
# For n=5 that is int(6 * p).  p=0.5 -> 3 root causes (the (n+1)/2 convention);
# p=0.4 -> 2, which is floor(5/2) and matches the original RCA run.
# Note floor(n/2) and (n+1)/2 agree at n=10 (both 5), so v10 needs no re-run.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
V=python3
R=baselines/paper_experiments/exp2_heterogeneous/v5_floor_noES
$V baselines/LIT/rca_in_lit_.py \
  --num_vars 5 --latent_vars_perc 0 --intv_wgt_func linear_dec \
  --obs_interv_perc 0.4 \
  --methods dcm_flow nsigma baro circa rcd rcg \
  --num_epc 50 \
  --num_obs_normal 5000 --num_obs_anomalous 5000 \
  --data_root $R/data --out_root $R/dcm_out \
  --thr_num 6 --gpu_id 1 --tot 100 \
  > logs/paper_exp2_v5_floor_noES.log 2>&1
echo V5_FLOOR_DONE
