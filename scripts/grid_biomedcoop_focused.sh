#!/bin/bash
# Focused sweep: 4-shot + 16-shot × 3 datasets × vision configs. Epochs = 400.
# Sequential GPU dispatch: each GPU takes the next job as it finishes, using wait -n.
set -u
source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment

OUT=output/BiomedCoOp_focused
mkdir -p "$OUT"
SEED=42
EPOCHS=400

# (dataset, vision_src, vision_model, text_src, text_model, use_op)
COMBOS=(
  "BUSI      OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 False"
  "BUSI      HF facebook/dinov2-base                                            OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
  "CHMNIST   OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 False"
  "CHMNIST   HF facebook/dinov2-base                                            OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
  "KneeXray  OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 False"
  "KneeXray  HF facebook/dinov2-base                                            OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
  "KneeXray  HF microsoft/rad-dino                                              OC hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 True"
)
SHOTS=(4 16)

JOBS=()
for combo in "${COMBOS[@]}"; do
  for K in "${SHOTS[@]}"; do JOBS+=("$combo|$K"); done
done
N=${#JOBS[@]}
NGPU=8
echo "N=$N"

run_one () {
  local gpu=$1 spec=$2
  local combo="${spec%|*}" K="${spec##*|}"
  read -r ds isrc imdl tsrc tmdl useop <<< "$combo"
  local vtag=$(echo "$imdl" | tr '/:' '__')
  local tag="${ds}_${vtag}_k${K}_s${SEED}"
  local log="$OUT/${tag}.log"
  echo "[$(date +%H:%M:%S)] gpu=$gpu START $tag"
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
      --config configs/busi_biomedclip.yaml \
      --output_dir "$OUT/$tag" \
      dataset "$ds" \
      img_src "$isrc" img_model "$imdl" \
      txt_src "$tsrc" txt_model "$tmdl" \
      use_op "$useop" \
      shots "$K" seed "$SEED" \
      epochs "$EPOCHS" \
      > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] gpu=$gpu DONE  $tag"
}

# Prime GPUs with the first NGPU jobs; then use `wait -n` to launch next as any finishes.
idx=0
declare -A gpu_of_pid
while (( idx < N && idx < NGPU )); do
  run_one "$idx" "${JOBS[$idx]}" &
  gpu_of_pid[$!]=$idx
  idx=$((idx+1))
done

while (( idx < N )); do
  # Block until one child exits
  wait -n
  # Find which GPU freed up: the pid whose entry is missing
  freed=-1
  for pid in "${!gpu_of_pid[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      freed=${gpu_of_pid[$pid]}
      unset 'gpu_of_pid[$pid]'
      break
    fi
  done
  if (( freed < 0 )); then
    echo "no freed GPU found; sleeping 5s"; sleep 5; continue
  fi
  run_one "$freed" "${JOBS[$idx]}" &
  gpu_of_pid[$!]=$freed
  idx=$((idx+1))
done
wait
echo "=== all $N runs complete ==="
