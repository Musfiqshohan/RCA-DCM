#!/bin/bash
# EXP 2, v10 -- re-run WITHOUT early stopping.
#
# Identical to the earlier v10 run except --num_epc 50 replaces
# --early_stop --max_epc 50.  That single change alters two things:
#   (1) every candidate trains the full 50 epochs (early stopping averaged ~27);
#   (2) the score is a TRAILING AVERAGE over the last 10 eval points
#       (_compute_scores) rather than a single snapshot at the stopping epoch
#       (_compute_scores_at_epoch).
# A 14-iteration diagnostic showed this lifts DCM from 0.240 -> 0.571
# (wasserstein) and 0.260 -> 0.429 (mmd, vs 0.448 published), so the published
# result reproduces only without early stopping.
#
# All six methods are re-run together: runs are unseeded, so baseline numbers
# from the early-stopped run were measured on DIFFERENT data and cannot be
# mixed with DCM numbers from this one.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
V=python3
R=baselines/paper_experiments/exp2_heterogeneous/v10_noES
$V baselines/LIT/rca_in_lit_.py \
  --num_vars 10 --latent_vars_perc 0 --intv_wgt_func linear_dec \
  --methods dcm_flow nsigma baro circa rcd rcg \
  --num_epc 50 \
  --num_obs_normal 5000 --num_obs_anomalous 5000 \
  --data_root $R/data --out_root $R/dcm_out \
  --thr_num 6 --gpu_id 0 --tot 100 \
  > logs/paper_exp2_v10_noES.log 2>&1
echo V10_NOES_DONE
