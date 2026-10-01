#!/bin/bash
# 50% deletion of BOTH edge kinds: 7->3 directed, 4->2 bidirected.
# The harshest arm -- removing directed edges strips genuine parents, so the
# candidate's mechanism is fit against an incomplete parent set. Unlike the
# addition arm (a supergraph stays Markov), deletion can inject false CIs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
python3 baselines/paper_experiments/exp_misspec/run_misspec.py \
  --data_root "baselines/paper_experiments/exp1_confounding/ls10_noES/data/lit_random_v10_lat4_none_latnstrg10.0_Ns5000As5000/" --mode both_sparse --frac 0.5 --graph_name lit_random_v10_lat4 \
  --num_epc 50 --limit 100 --thr_num 6 --gpu_id 1 --out baselines/paper_experiments/exp_misspec/ls10_bothsparse50 \
  > logs/misspec_bothsparse50.log 2>&1
echo BOTHSPARSE_DONE
