#!/usr/bin/env bash
# Score the one Sock Shop case shipped in this folder.
# Uses the same graph and early-stopping settings as the published microservice runs.
# A full paper run is the commands in the top-level README, not this script.
set -euo pipefail
cd "$(dirname "$0")/.."
export RCAEVAL_DATA_ROOT="$PWD/examples/microservice"

python3 run_sink_graph.py \
  --dataset sockshop \
  --faults cpu \
  --replicates 1 \
  --sink_confounded \
  --engine original \
  --early_stop \
  --max_epc 50 \
  --thr_num 1 \
  --tag sample

echo
echo "Ranking (highest score first):"
echo "  dcm/out/sockshop_sample/cpu/carts_cpu_r1/root_cause_results.csv"
echo "The injected service is carts. A matching published ranking is in"
echo "  examples/results/sockshop_carts_cpu_r1/root_cause_results.csv"
