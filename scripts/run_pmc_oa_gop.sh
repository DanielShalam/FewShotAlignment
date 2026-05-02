#!/bin/bash
# Run 8-GPU extraction of PMC-OA features and fit GOP.
# Usage: bash run_pmc_oa_gop.sh <stage>  (stage: img | txt_bmc | txt_qwen | all)
set -u
STAGE="${1:-all}"
N=500000

source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment
export HF_TOKEN=$(cat /efs/user_folders/dnshalam/hf_cache/token)

SCRIPT=scripts/prep_pmc_oa_gop.py
LOGDIR=/efs/user_folders/dnshalam/work/FewShotAlignment/data/pmc_oa/logs
mkdir -p "$LOGDIR"

run_stage () {
  local s=$1
  echo "=== stage=$s ==="
  for r in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$r python -u $SCRIPT --stage $s --rank $r --world 8 --n_samples $N \
      > "$LOGDIR/${s}_r${r}.log" 2>&1 &
  done
  wait
  echo "[done] stage=$s"
}

if [[ "$STAGE" == "all" ]]; then
  run_stage img
  run_stage txt_bmc
  run_stage txt_qwen
  python -u $SCRIPT --stage merge --world 8 --n_samples $N
elif [[ "$STAGE" == "merge" ]]; then
  python -u $SCRIPT --stage merge --world 8 --n_samples $N
else
  run_stage "$STAGE"
fi
