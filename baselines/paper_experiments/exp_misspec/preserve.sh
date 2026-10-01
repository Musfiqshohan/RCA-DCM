#!/bin/bash
# Freeze the misspecification experiment for upload alongside the code.
#
# WHY THIS EXISTS: the source datasets live under ANOTHER experiment's folder
# (exp1_confounding/ls10_noES/). If Exp 1 is ever cleaned or re-run, this
# experiment's inputs disappear and its results become unreproducible. This
# script records provenance + checksums, so the dependency is explicit and any
# later drift is detectable.
#
# Datasets themselves are NOT copied (285MB, already in the repo under Exp 1) --
# they are referenced by path and verified by checksum.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$ROOT/dcm/dcm_rca.py" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
cd "$ROOT"
mkdir -p logs
R=baselines/paper_experiments/exp_misspec
SRC=baselines/paper_experiments/exp1_confounding/ls10_noES/data/lit_random_v10_lat4_none_latnstrg10.0_Ns5000As5000
M=$R/MANIFEST

mkdir -p $M
echo "== source datasets =="
# per-dataset checksum over the files the experiment actually reads
: > $M/source_checksums.txt
for d in $SRC/dataset*/; do
  for f in disc_nrm.csv disc_anm.csv true_rc.json graph.yaml; do
    [ -f "$d/$f" ] && md5sum "$d/$f" >> $M/source_checksums.txt
  done
done
echo "  $(wc -l < $M/source_checksums.txt) files checksummed"

echo "== perturbed graphs + results =="
find $R -name "misspec_*.csv" -exec md5sum {} \; > $M/result_checksums.txt
find $R -path "*/graphs/*.yaml" -exec md5sum {} \; >> $M/result_checksums.txt
echo "  $(wc -l < $M/result_checksums.txt) files checksummed"

echo "== provenance =="
{
  echo "misspecification experiment -- provenance"
  echo "frozen: $(date -Is)"
  echo "git branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit: $(git rev-parse HEAD)"
  echo "source datasets: $SRC"
  echo "  count: $(ls -d $SRC/dataset* | wc -l)"
  echo "arms:"
  for a in $R/ls10_*; do
    [ -d "$a" ] || continue
    c=$(ls $a/graphs/*.yaml 2>/dev/null | wc -l)
    r=$(find $a -name 'misspec_*.csv' | head -1)
    echo "  $(basename $a): $c perturbed graphs, rows=$( [ -n "$r" ] && echo $(( $(wc -l < $r) - 1 )) || echo 0 )"
  done
  echo "NOTE: DCM training is unseeded (BUG A1-20); perturbed GRAPHS reproduce exactly,"
  echo "      fitted models do not. Arms are paired by seed=1000+i for valid McNemar."
} > $M/PROVENANCE.txt
cat $M/PROVENANCE.txt

echo "== dropping regenerable training artefacts =="
before=$(du -sh $R 2>/dev/null | cut -f1)
rm -rf $R/ls10_*/runs
echo "  $R: $before -> $(du -sh $R | cut -f1)"
echo "DONE -- $M/ holds checksums + provenance"
