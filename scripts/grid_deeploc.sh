#!/bin/bash
# Launch the DeepLoc2 grid: shots x text_model x seed. One job per GPU at a time; fill 8 GPUs.
set -u
source /efs/user_folders/dnshalam/work/FewShotAlignment/env.sh
cd /efs/user_folders/dnshalam/work/FewShotAlignment

OUT=output/DeepLoc2/grid
mkdir -p "$OUT"

SHOTS=(1 4 16 64)
SEEDS=(42 43 44)
TXT_KEYS=(pubmed qwen06)

declare -A TXT_PATH=(
  [pubmed]="data/deeploc/class_text_biomednlp_pubmedbert_base_uncased_abstract_fulltext.pt"
  [qwen06]="data/deeploc/class_text_qwen3_embedding_0_6b.pt"
)

# job = (k, t, s)
JOBS=()
for k in "${SHOTS[@]}"; do for t in "${TXT_KEYS[@]}"; do for s in "${SEEDS[@]}"; do
  JOBS+=("$k $t $s")
done; done; done

NGPU=8
pids=()
gpus_in_use=()

launch () {
  local gpu=$1 k=$2 t=$3 s=$4
  local tag="esm150_${t}_k${k}_seed${s}"
  local odir="$OUT/$tag"
  local log="$OUT/${tag}.log"
  # pick batch_size so train_loader has >=2 batches even at tiny K (10*K samples total).
  local bs=$((10*k)); (( bs > 64 )) && bs=64; (( bs < 8 )) && bs=8
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
      --config configs/deeploc2_esm150_qwen.yaml \
      --output_dir "$odir" \
      shots "$k" seed "$s" batch_size "$bs" \
      text_features_path "${TXT_PATH[$t]}" \
      tune_t_steps 6 \
      > "$log" 2>&1
  touch "$OUT/${tag}.done"
}

idx=0
N=${#JOBS[@]}
echo "scheduling $N jobs on $NGPU GPUs"

# prime the pool
while (( idx < N && ${#pids[@]} < NGPU )); do
  gpu=${#pids[@]}
  read -r k t s <<< "${JOBS[$idx]}"
  launch $gpu "$k" "$t" "$s" &
  pids+=($!); gpus_in_use+=($gpu)
  echo "launched gpu=$gpu job=${JOBS[$idx]} pid=$!"
  idx=$((idx+1))
done

# feed the pool as jobs finish
while (( idx < N )); do
  # wait for any to finish
  for i in "${!pids[@]}"; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
      free_gpu=${gpus_in_use[$i]}
      unset 'pids[$i]'; unset 'gpus_in_use[$i]'
      pids=("${pids[@]}"); gpus_in_use=("${gpus_in_use[@]}")
      read -r k t s <<< "${JOBS[$idx]}"
      launch $free_gpu "$k" "$t" "$s" &
      pids+=($!); gpus_in_use+=($free_gpu)
      echo "launched gpu=$free_gpu job=${JOBS[$idx]} pid=$!"
      idx=$((idx+1))
      break
    fi
  done
  sleep 5
done
wait
echo "=== grid finished ==="
