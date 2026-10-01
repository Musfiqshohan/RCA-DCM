#!/bin/bash
# TRUE-graph control through the SAME harness as the perturbed arms.
# Without it, "misspecification costs nothing" rests on comparing DCM_RCA-direct
# runs against an rca_in_lit_ run (0.84) -- a cross-harness comparison that
# cannot separate a harness effect from a perturbation effect.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
# wait for the two perturbed arms to release the GPUs
python3 baselines/paper_experiments/exp_misspec/run_misspec.py \
  --data_root "baselines/paper_experiments/exp1_confounding/ls10_noES/data/lit_random_v10_lat4_none_latnstrg10.0_Ns5000As5000/" --mode true --frac 0 --graph_name lit_random_v10_lat4 \
  --num_epc 50 --limit 100 --thr_num 6 --gpu_id 0 --out baselines/paper_experiments/exp_misspec/ls10_control \
  > logs/misspec_control.log 2>&1
echo CONTROL_DONE
