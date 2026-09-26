#!/bin/bash
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

# 新習志野 A4500 x5 / Ternary Bonsai 2 27B PQ2_0, 1 GPU に 262144 x 1 slot を5つ (shinnarashino/README.md)
# Submit from llama/: sbatch shinnarashino/bonsai.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec python3 -u common/launch.py "$@" \
    --server llama.cpp-prism/build/bin/llama-server \
    --model models/Bonsai-2-27B/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --alias bonsai-2-27b-pq2_0 --replicas 5 -- \
    -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
    --threads 2 --threads-batch 2 -c 262144 --parallel 1 \
    --fit off --jinja
