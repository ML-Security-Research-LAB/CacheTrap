#!/usr/bin/env bash
# Train the victim classifiers, round-robin across GPUs.
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=("llama2" "llama3_1_8b" "mistral7B" "qwen3B" "deepseek_qwen")
DATASETS=("trec" "arc_easy" "openbookqa" "arc_challenge")

NUM_GPUS=4
OUTPUT_DIR="./TrainedModels"
LOG_DIR="logs/training"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

job_count=0
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    GPU=$((job_count % NUM_GPUS))
    LOG_FILE="${LOG_DIR}/${MODEL}_${DATASET}.log"

    echo "Launching: GPU ${GPU} | model=${MODEL} | dataset=${DATASET} -> ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES=${GPU} python train_model.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1 &

    job_count=$((job_count + 1))
    if ((job_count % NUM_GPUS == 0)); then
      wait
    fi
  done
done

wait
echo ">>> All training jobs completed."
