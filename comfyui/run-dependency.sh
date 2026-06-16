#!/bin/bash

DEPEND_JOB_ID="${1:-}"

if [ -z "$DEPEND_JOB_ID" ]; then
    echo "ERROR: Job ID required. Usage: bash run-dependency.sh <JobID>"
    exit 1
fi

if [ -f .env ]; then
    source .env
else
    echo "ERROR: .env file not found."
    exit 1
fi

export DOCKER_HOST="unix:////run/user/$(id -u)/docker.sock"

sbatch \
  --dependency=afterany:"${DEPEND_JOB_ID}" \
  --partition="${SLURM_PARTITION}" \
  --cpus-per-task="${SLURM_CPUS}" \
  --gpus-per-task="${SLURM_GPUS}" \
  --mem="${SLURM_MEM}" \
  comfyui.job

echo "Submitted at $(date) with dependency on job ${DEPEND_JOB_ID}"