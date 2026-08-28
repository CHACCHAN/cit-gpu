#!/bin/bash

# ./taskmanager.sh -u "$USER"

set -euo pipefail

USER_NAME="$USER"
JOB_ID=""

usage() {
    echo "Usage: $0 [-u USER] [-j JOB_ID]"
    exit 1
}

while getopts "u:j:h" opt; do
    case "$opt" in
        u) USER_NAME="$OPTARG" ;;
        j) JOB_ID="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# JOB_ID未指定なら、指定ユーザーの実行中ジョブを探す
if [[ -z "$JOB_ID" ]]; then
    mapfile -t JOBS < <(
        squeue -h \
            -u "$USER_NAME" \
            -t RUNNING \
            -o "%A"
    )

    if [[ ${#JOBS[@]} -eq 0 ]]; then
        echo "実行中のジョブがありません: $USER_NAME"
        exit 1
    fi

    if [[ ${#JOBS[@]} -gt 1 ]]; then
        echo "複数の実行中ジョブがあります:"
        squeue -u "$USER_NAME"
        echo
        echo "次のようにジョブIDを指定してください:"
        echo "  $0 -j <JOB_ID>"
        exit 1
    fi

    JOB_ID="${JOBS[0]}"
fi

echo "Monitoring Job ID: $JOB_ID"
echo "User: $USER_NAME"
sleep 1

srun --jobid="$JOB_ID" --overlap bash -c '
while true; do
    clear

    echo "=== JOB ==="
    echo "Job ID: '"$JOB_ID"'"
    echo

    echo "=== RAM ==="
    free -h

    echo
    echo "=== CPU / Load ==="
    uptime

    echo
    echo "=== GPU ==="
    nvidia-smi \
        --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
        --format=csv,noheader

    sleep 2
done
'