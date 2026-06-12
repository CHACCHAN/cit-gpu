#!/bin/bash

if [ -f .env ]; then
    source .env
else
    echo "ERROR: .env file not found. Please create it."
    exit 1
fi

if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo "ERROR: NGROK_AUTHTOKEN is not set in .env"
    exit 1
fi

rm -rf logs

echo "Deploying vLLM on Partition: ${SLURM_PARTITION} with Max Performance..."

sbatch \
  --partition="${SLURM_PARTITION}" \
  --cpus-per-task="${SLURM_CPUS}" \
  --gpus-per-task="${SLURM_GPUS}" \
  --mem="${SLURM_MEM}" \
  vllm.job