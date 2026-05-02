#!/bin/bash
# BiomedCoOp P3 pilot (seed 42 only): 3 datasets × {1,2,4,8,16} shots × vision combos.
set -u
source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment

OUT=output/BiomedCoOp_pilot
mkdir -p "$OUT"

# (dataset, vision_src, vision_model, img_dim, text_src, text_model, use_op)
COMBOS=(
  "BUSI      OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 512 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 False"
  "BUSI      HF facebook/dinov2-base                                            768 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
  "CHMNIST   OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 512 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 False"
  "CHMNIST   HF facebook/dinov2-base                                            768 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
  "KneeXray  OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 512 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 False"
  "KneeXray  HF facebook/dinov2-base                                            768 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
  "KneeXray  HF microsoft/rad-dino                                              768 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
)
SHOTS=(1 2 4 8 16)
SEED=42

JOBS=()
for combo in "${COMBOS[@]}"; do
  for K in "${SHOTS[@]}"; do
    JOBS+=("$combo|$K")
  done
done
N=${#JOBS[@]}
NGPU=8
echo "scheduling $N jobs on $NGPU GPUs"

launch () {
  local gpu=$1 spec=$2
  local combo="${spec%|*}" K="${spec##*|}"
  read -r ds isrc imdl idim tsrc tmdl useop <<< "$combo"
  local vtag=$(echo "$imdl" | tr '/:' '__')
  local tag="${ds}_${vtag}_k${K}_s${SEED}"
  local odir="$OUT/$tag"
  local log="$OUT/${tag}.log"

  CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
      --config configs/busi_biomedclip.yaml \
      --output_dir "$odir" \
      dataset "$ds" \
      img_src "$isrc" img_model "$imdl" \
      txt_src "$tsrc" txt_model "$tmdl" \
      use_op "$useop" \
      shots "$K" seed "$SEED" \
      > "$log" 2>&1
}

pids=(); gpus_in_use=(); idx=0
while (( idx < N && ${#pids[@]} < NGPU )); do
  gpu=${#pids[@]}
  launch $gpu "${JOBS[$idx]}" &
  pids+=($!); gpus_in_use+=($gpu)
  echo "gpu=$gpu pid=$! job=${JOBS[$idx]}"
  idx=$((idx+1))
done

while (( idx < N )); do
  for i in "${!pids[@]}"; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
      free_gpu=${gpus_in_use[$i]}
      unset 'pids[$i]'; unset 'gpus_in_use[$i]'
      pids=("${pids[@]}"); gpus_in_use=("${gpus_in_use[@]}")
      launch $free_gpu "${JOBS[$idx]}" &
      pids+=($!); gpus_in_use+=($free_gpu)
      echo "gpu=$free_gpu pid=$! job=${JOBS[$idx]}"
      idx=$((idx+1))
      break
    fi
  done
  sleep 10
done
wait
echo "=== pilot complete ==="
