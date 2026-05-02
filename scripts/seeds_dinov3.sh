#!/bin/bash
# Seeds sweep: DINOv3-B on {BUSI,CHMNIST,KneeXray} × {K=4,16} × seeds {0,1}. 12 runs.
set -u
source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment
export HF_TOKEN=$(cat /efs/user_folders/dnshalam/hf_cache/token)

OUT=output/BiomedCoOp_focused
DV3=facebook/dinov3-vitb16-pretrain-lvd1689m
BMC=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

JOBS=()
for DS in BUSI CHMNIST KneeXray; do
  for K in 4 16; do
    for SEED in 0 1; do
      JOBS+=("$DS|$K|$SEED")
    done
  done
done
N=${#JOBS[@]}
NGPU=8
echo "N=$N NGPU=$NGPU"

run_one () {
  local gpu=$1 spec=$2
  IFS='|' read -r DS K SEED <<< "$spec"
  local tag=${DS}_dinov3_k${K}_s${SEED}
  local log=$OUT/${tag}.log
  echo "[$(date +%H:%M:%S)] gpu=$gpu START $tag"
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py --config configs/busi_biomedclip.yaml \
      --output_dir "$OUT/$tag" \
      dataset "$DS" img_src HF img_model "$DV3" txt_src OC txt_model "$BMC" \
      use_op True shots "$K" seed "$SEED" epochs 400 lr 0.00001 \
      > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] gpu=$gpu DONE  $tag"
}

idx=0
declare -A gpu_of_pid
while (( idx < N && idx < NGPU )); do
  run_one "$idx" "${JOBS[$idx]}" &
  gpu_of_pid[$!]=$idx
  idx=$((idx+1))
done

while (( idx < N )); do
  wait -n
  freed=-1
  for pid in "${!gpu_of_pid[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      freed=${gpu_of_pid[$pid]}
      unset 'gpu_of_pid[$pid]'
      break
    fi
  done
  if (( freed < 0 )); then sleep 5; continue; fi
  run_one "$freed" "${JOBS[$idx]}" &
  gpu_of_pid[$!]=$freed
  idx=$((idx+1))
done
wait
echo "=== all $N seeds done ==="
