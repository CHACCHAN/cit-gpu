#!/bin/bash

if [ -f .env ]; then
    source .env
else
    echo "ERROR: .env file not found. Please create it."
    exit 1
fi

export DOCKER_HOST="unix:////run/user/$(id -u)/docker.sock"

rm -rf logs

sbatch \
  --partition="${SLURM_PARTITION}" \
  --cpus-per-task="${SLURM_CPUS}" \
  --gpus-per-task="${SLURM_GPUS}" \
  --mem="${SLURM_MEM}" \
  comfyui.job