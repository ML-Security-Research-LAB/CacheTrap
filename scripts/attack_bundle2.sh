#!/usr/bin/env bash
# Bundle 2 - victims trained on ARC-Easy, calibrated on OpenBookQA.
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=("llama2" "llama3_1_8b" "mistral7B" "qwen3B" "deepseek_qwen")
EVAL_DATASETS=("arc_easy")
CALIB_DATASET="openbookqa"

NUM_GPUS=4
LOG_DIR="logs/attack_bundle2"

mkdir -p "${LOG_DIR}"

job_count=0
for EVAL_DATASET in "${EVAL_DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    GPU=$((job_count % NUM_GPUS))
    LOG_FILE="${LOG_DIR}/${MODEL}_${EVAL_DATASET}.log"

    echo "Launching: GPU ${GPU} | model=${MODEL} | eval=${EVAL_DATASET} | calib=${CALIB_DATASET}"
    CUDA_VISIBLE_DEVICES=${GPU} python attack.py \
        --model "${MODEL}" \
        --dataset "${EVAL_DATASET}" \
        --calib_dataset "${CALIB_DATASET}" \
        --threat_model graybox \
        --calib_samples 100 \
        --calib_eval_samples 100 \
        --top_m_per_class 3 \
        --top_k 20 \
        --corrupt_k 1 \
        > "${LOG_FILE}" 2>&1 &

    job_count=$((job_count + 1))
    if ((job_count % NUM_GPUS == 0)); then
      wait
    fi
  done
done

wait
echo ">>> All attack jobs completed."

python summarize_logs.py --log_dir "${LOG_DIR}"
