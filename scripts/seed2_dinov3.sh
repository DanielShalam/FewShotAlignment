#!/bin/bash
# Seed=2 replacement: DINOv3-B on 3 datasets × K={4,16}. 6 runs.
set -u
source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment
export HF_TOKEN=$(cat /efs/user_folders/dnshalam/hf_cache/token)

OUT=output/BiomedCoOp_focused
DV3=facebook/dinov3-vitb16-pretrain-lvd1689m
BMC=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
SEED=2

JOBS=()
for DS in BUSI CHMNIST KneeXray; do
  for K in 4 16; do
    JOBS+=("$DS|$K")
  done
done
N=${#JOBS[@]}
echo "N=$N SEED=$SEED"

run_one () {
  local gpu=$1 spec=$2
  IFS='|' read -r DS K <<< "$spec"
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

# 6 jobs, 8 GPUs -> one wave
for i in "${!JOBS[@]}"; do
  run_one "$i" "${JOBS[$i]}" &
done
wait
echo "=== seed=$SEED sweep done ==="
