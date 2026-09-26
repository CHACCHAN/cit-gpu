#!/bin/bash
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

# 津田沼 H100 NVL x2 / MiMo V2.6 Flash RL Q3_K + Heretic LoRA (tsudanuma/README.md)
# Submit from llama/: Slurm copies the script, so its own directory is not usable.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec python3 -u common/launch.py "$@" \
    --server llama.cpp/build/bin/llama-server \
    --model models/Q3_K/MiMo-V2.6-Flash-RL-Q3_K-00001-of-00004.gguf \
    --lora adapters/MiMo-V2.6-Flash-RL-Uncensored-Heretic-lora.gguf \
    --alias mimo-v2.6-flash-rl-q3_k -- \
    -ngl all -sm layer -ts 1,1 -fa on -ctk q8_0 -ctv q8_0 \
    --threads 8 --threads-batch 8 --parallel 3 --kv-unified \
    --kv-unified-per-slot 524288 --fit off --jinja
