#!/bin/bash
# Re-run EXP 1 (both lambdas) WITHOUT early stopping, for consistency with the
# v10 no-ES run.  Early stopping scores from a single eval snapshot at the stop
# epoch; without it the score is a trailing average over the last 10 eval points.
# On v10 that difference cost more than half the accuracy (0.540 -> 0.240).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
V=python3
R=baselines/paper_experiments/exp1_confounding
for LS in 3 10; do
  $V baselines/LIT/rca_in_lit_.py \
    --num_vars 10 --latent_vars_perc 0.4 --intv_wgt_func none \
    --latent_edge_strength $LS \
    --methods dcm_flow nsigma baro circa rcd rcg \
    --num_epc 50 \
    --num_obs_normal 5000 --num_obs_anomalous 5000 \
    --data_root $R/ls${LS}_noES/data --out_root $R/ls${LS}_noES/dcm_out \
    --thr_num 6 --gpu_id 0 --tot 100 \
    > logs/paper_exp1_ls${LS}_noES.log 2>&1
  echo "  done exp1 ls=$LS (noES)"
done
echo EXP1_NOES_DONE
