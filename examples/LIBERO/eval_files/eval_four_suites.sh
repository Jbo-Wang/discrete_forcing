#!/bin/bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
STARVLA_PYTHON=${STARVLA_PYTHON:-$(command -v python)}
LIBERO_PYTHON=${LIBERO_PYTHON:-$(command -v python)}
LIBERO_HOME=${LIBERO_HOME:-${PROJECT_ROOT}/third_party/LIBERO}

usage() {
  cat <<'EOF'
Usage:
  eval_four_suites.sh --checkpoint PATH --gpu ID --port PORT [options]

Required:
  --checkpoint PATH       Model .pt/.safetensors checkpoint
  --gpu ID                Physical GPU id exposed to the policy server
  --port PORT             Unique TCP port for this checkpoint evaluation

Options:
  --trials N              Trials per task (default: 50)
  --suite NAME            Evaluate one suite instead of all four
  --seed N                LIBERO seed (default: 7)
  --output-dir PATH       Explicit result directory
  --num-inference-steps N Override checkpoint inference timesteps
  --num-discrete-steps N  Override discrete denoising steps
  --num-continuous-steps N Override continuous denoising steps
  --inference-branch-mode MODE hybrid
  --server-timeout SEC    Model-server startup timeout (default: 600)
  --keep-server           Leave the policy server running after evaluation
  -h, --help              Show this help
EOF
}

checkpoint=""
gpu=""
port=""
trials=50
seed=7
output_dir=""
server_timeout=600
num_inference_steps=""
num_discrete_steps=""
num_continuous_steps=""
inference_branch_mode="hybrid"
keep_server=false
suite=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) checkpoint=${2:?Missing value for --checkpoint}; shift 2 ;;
    --gpu) gpu=${2:?Missing value for --gpu}; shift 2 ;;
    --port) port=${2:?Missing value for --port}; shift 2 ;;
    --trials) trials=${2:?Missing value for --trials}; shift 2 ;;
    --suite) suite=${2:?Missing value for --suite}; shift 2 ;;
    --seed) seed=${2:?Missing value for --seed}; shift 2 ;;
    --output-dir) output_dir=${2:?Missing value for --output-dir}; shift 2 ;;
    --server-timeout) server_timeout=${2:?Missing value for --server-timeout}; shift 2 ;;
    --num-inference-steps) num_inference_steps=${2:?Missing value for --num-inference-steps}; shift 2 ;;
    --num-discrete-steps) num_discrete_steps=${2:?Missing value for --num-discrete-steps}; shift 2 ;;
    --num-continuous-steps) num_continuous_steps=${2:?Missing value for --num-continuous-steps}; shift 2 ;;
    --inference-branch-mode) inference_branch_mode=${2:?Missing value for --inference-branch-mode}; shift 2 ;;
    --keep-server) keep_server=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$checkpoint" && -n "$gpu" && -n "$port" ]] || { usage >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "--gpu must be a non-negative integer" >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 65535 )) || {
  echo "--port must be an integer in [1024, 65535]" >&2
  exit 2
}
[[ "$trials" =~ ^[1-9][0-9]*$ ]] || { echo "--trials must be a positive integer" >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
[[ "$server_timeout" =~ ^[1-9][0-9]*$ ]] || {
  echo "--server-timeout must be a positive integer" >&2
  exit 2
}
if [[ -n "$num_inference_steps" && ! "$num_inference_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "--num-inference-steps must be a positive integer" >&2
  exit 2
fi
if [[ -n "$num_discrete_steps" && ! "$num_discrete_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "--num-discrete-steps must be a positive integer" >&2
  exit 2
fi
if [[ -n "$num_continuous_steps" && ! "$num_continuous_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "--num-continuous-steps must be a positive integer" >&2
  exit 2
fi
if [[ "$inference_branch_mode" != hybrid ]]; then
  echo "--inference-branch-mode must be hybrid" >&2
  exit 2
fi
if [[ -n "$suite" && ! "$suite" =~ ^libero_(spatial|object|goal|10)$ ]]; then
  echo "--suite must be libero_spatial, libero_object, libero_goal, or libero_10" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
checkpoint=$(realpath "$checkpoint")
[[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 1; }
[[ -x "$STARVLA_PYTHON" ]] || { echo "Missing StarVLA Python: $STARVLA_PYTHON" >&2; exit 1; }
[[ -x "$LIBERO_PYTHON" ]] || { echo "Missing LIBERO Python: $LIBERO_PYTHON" >&2; exit 1; }

checkpoint_file=$(basename "$checkpoint")
checkpoint_parent=$(basename "$(dirname "$checkpoint")")
checkpoint_run=$(basename "$(dirname "$(dirname "$checkpoint")")")
checkpoint_label=${checkpoint_run}_${checkpoint_parent}_${checkpoint_file%.*}
checkpoint_label=${checkpoint_label//[^a-zA-Z0-9._-]/_}
if [[ -z "$output_dir" ]]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  output_dir="$(dirname "$(dirname "$checkpoint")")/evaluations/four_suites/${checkpoint_label}_${timestamp}_p${port}_pid$$"
fi
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")

exec 9>"/tmp/starvla_libero_eval_port_${port}.lock"
if ! flock -n 9; then
  echo "Port $port is reserved by another four-suite evaluation." >&2
  exit 1
fi
if ss -H -ltn "sport = :$port" | grep -q .; then
  echo "Port $port is already listening; choose another --port." >&2
  exit 1
fi

export LIBERO_HOME
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${PROJECT_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

server_pid=""
cleanup() {
  status=$?
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    if [[ "$keep_server" == true && $status -eq 0 ]]; then
      echo "Keeping policy server PID $server_pid on port $port"
    else
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

exec > >(tee -a "$output_dir/all_suites.log") 2>&1

"$STARVLA_PYTHON" - "$output_dir/run_metadata.json" "$checkpoint" "$gpu" "$port" "$trials" "$seed" "$num_inference_steps" "$num_discrete_steps" "$num_continuous_steps" "$inference_branch_mode" <<'PY'
import datetime
import json
import socket
import sys

path, checkpoint, gpu, port, trials, seed, inference_steps, discrete_steps, continuous_steps, branch_mode = sys.argv[1:]
metadata = {
    "checkpoint": checkpoint,
    "gpu": int(gpu),
    "port": int(port),
    "trials_per_task": int(trials),
    "seed": int(seed),
    "num_inference_steps": int(inference_steps) if inference_steps else None,
    "num_discrete_steps": int(discrete_steps) if discrete_steps else None,
    "num_continuous_steps": int(continuous_steps) if continuous_steps else None,
    "inference_branch_mode": branch_mode,
    "started_at": datetime.datetime.now().astimezone().isoformat(),
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(metadata, stream, indent=2, ensure_ascii=False)
PY

echo "Checkpoint: $checkpoint"
echo "GPU: $gpu"
echo "Port: $port"
echo "Output: $output_dir"

server_args=(
  --ckpt_path "$checkpoint" \
  --port "$port" \
  --use_bf16 \
  --idle_timeout -1
  --inference_branch_mode "$inference_branch_mode"
)
if [[ -n "$num_inference_steps" ]]; then
  server_args+=(--num_inference_steps "$num_inference_steps")
fi
if [[ -n "$num_discrete_steps" ]]; then
  server_args+=(--num_discrete_steps "$num_discrete_steps")
fi
if [[ -n "$num_continuous_steps" ]]; then
  server_args+=(--num_continuous_steps "$num_continuous_steps")
fi

CUDA_VISIBLE_DEVICES="$gpu" "$STARVLA_PYTHON" deployment/model_server/server_policy.py \
  "${server_args[@]}" \
  >"$output_dir/server.log" 2>&1 &
server_pid=$!
echo "$server_pid" >"$output_dir/server.pid"

deadline=$((SECONDS + server_timeout))
while ! ss -H -ltn "sport = :$port" | grep -q .; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Policy server exited before becoming ready. See $output_dir/server.log" >&2
    wait "$server_pid"
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out after ${server_timeout}s waiting for policy server on port $port." >&2
    exit 1
  fi
  sleep 2
done
if ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Policy server died while port $port became active; refusing to evaluate another listener." >&2
  wait "$server_pid"
fi
echo "Policy server ready (PID $server_pid)."

suites=(libero_spatial libero_object libero_goal libero_10)
[[ -z "$suite" ]] || suites=("$suite")
for suite in "${suites[@]}"; do
  if (( ${#suites[@]} == 1 )); then
    suite_dir="$output_dir"
  else
    suite_dir="$output_dir/$suite"
  fi
  mkdir -p "$suite_dir/videos"
  echo "===== START $suite $(date --iso-8601=seconds) ====="
  CUDA_VISIBLE_DEVICES="$gpu" "$LIBERO_PYTHON" examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "$checkpoint" \
    --args.host 127.0.0.1 \
    --args.port "$port" \
    --args.task-suite-name "$suite" \
    --args.num-trials-per-task "$trials" \
    --args.seed "$seed" \
    --args.video-out-path "$suite_dir/videos" \
    2>&1 | tee "$suite_dir/eval.log"
  cp "$suite_dir/videos/evaluation_summary.json" "$suite_dir/evaluation_summary.json"
  echo "===== END $suite $(date --iso-8601=seconds) ====="
done

if (( ${#suites[@]} == 4 )); then
  "$STARVLA_PYTHON" examples/LIBERO/eval_files/aggregate_four_suites.py "$output_dir"
  echo "Four-suite evaluation completed: $output_dir"
else
  echo "Suite evaluation completed: $output_dir"
fi
