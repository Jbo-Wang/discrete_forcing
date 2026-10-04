#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$project_root:${PYTHONPATH:-}"
exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/robotwin_clean_method.yaml \
  "$@"
