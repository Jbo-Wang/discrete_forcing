#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval_dir="$project_root/examples/Robotwin/eval_files"
starvla_python="${STARVLA_PYTHON:-$(command -v python)}"
robotwin_python="${ROBOTWIN_PYTHON:-}"
robotwin_home="${ROBOTWIN_HOME:-}"
checkpoint=""
gpus="0,1,2,3,4,5,6,7"
port_base=5694
seed=0
task=""
output_dir=""

usage() {
  echo "Usage: ROBOTWIN_HOME=... ROBOTWIN_PYTHON=... bash eval_robotwin.sh --checkpoint PATH [--gpus 0,1] [--task NAME] [--port-base 5694] [--output-dir PATH]" >&2
}

while (( $# )); do
  case "$1" in
    --checkpoint) checkpoint="${2:?}"; shift 2 ;;
    --gpus) gpus="${2:?}"; shift 2 ;;
    --port-base) port_base="${2:?}"; shift 2 ;;
    --seed) seed="${2:?}"; shift 2 ;;
    --task) task="${2:?}"; shift 2 ;;
    --output-dir) output_dir="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ -f "$checkpoint" && -f "$robotwin_home/script/eval_policy.py" && -x "$robotwin_python" ]] || {
  echo "Provide an existing checkpoint, ROBOTWIN_HOME, and ROBOTWIN_PYTHON." >&2
  exit 1
}
[[ "$port_base" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]] || exit 2
IFS=',' read -r -a gpu_ids <<< "$gpus"
(( ${#gpu_ids[@]} > 0 )) || exit 2
for gpu in "${gpu_ids[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
done
(( port_base >= 1024 && port_base + ${#gpu_ids[@]} - 1 <= 65535 )) || {
  echo "Port range is outside [1024, 65535]." >&2
  exit 2
}

checkpoint="$(realpath "$checkpoint")"
if [[ -z "$output_dir" ]]; then
  output_dir="$(dirname "$(dirname "$checkpoint")")/robotwin_eval_logs/clean50_$(date +%Y%m%d_%H%M%S)_pid$$"
fi
[[ ! -e "$output_dir" ]] || { echo "Output directory already exists: $output_dir" >&2; exit 1; }
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"
mapfile -t tasks < <(cd "$project_root" && "$starvla_python" -c \
  'from starVLA.dataloader.gr00t_lerobot.mixtures import ROBOTWIN_CLEAN_TASKS; print("\n".join(ROBOTWIN_CLEAN_TASKS))')
(( ${#tasks[@]} == 50 )) || { echo "Could not load the RoboTwin clean 50 task list." >&2; exit 1; }
if [[ -n "$task" ]]; then
  [[ " ${tasks[*]} " == *" $task "* ]] || { echo "Unknown task: $task" >&2; exit 2; }
  tasks=("$task")
fi
printf '%s\n' "${tasks[@]}" > "$output_dir/tasks.txt"

run_worker() (
  local worker="$1" gpu="$2" port=$((port_base + worker)) index name server_pid="" rc
  trap '[[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true' EXIT
  for (( index=worker; index<${#tasks[@]}; index+=${#gpu_ids[@]} )); do
    name="${tasks[index]}"
    server_pid=""
    if ! "$starvla_python" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$port"; then
      echo "Port $port is occupied" > "$output_dir/$name.status"
      continue
    fi
    (cd "$project_root" && env CUDA_VISIBLE_DEVICES="$gpu" \
      TRITON_CACHE_DIR="${TRITON_CACHE_ROOT:-${TMPDIR:-/tmp}/starvla_${USER:-user}_gpu${gpu}}/triton" \
      PYTHONPATH="$project_root:${PYTHONPATH:-}" \
      "$starvla_python" deployment/model_server/server_policy.py \
        --ckpt_path "$checkpoint" --port "$port" --use_bf16 --idle_timeout -1) \
      > "$output_dir/$name.server.log" 2>&1 &
    server_pid=$!
    ready=false
    for (( attempt=0; attempt<450; attempt++ )); do
      if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
      if "$starvla_python" -c 'from websockets.sync.client import connect; import sys; connect("ws://127.0.0.1:"+sys.argv[1], open_timeout=1).close()' "$port" 2>/dev/null; then
        ready=true
        break
      fi
      sleep 2
    done
    if [[ "$ready" == true ]]; then
      set +e
      (cd "$robotwin_home" && env CUDA_VISIBLE_DEVICES="$gpu" \
        ROBOTWIN_TEST_NUM=100 \
        PYTHONPATH="$robotwin_home:$project_root:$eval_dir:${PYTHONPATH:-}" \
        "$robotwin_python" script/eval_policy.py \
          --config "$eval_dir/deploy_policy.yml" --overrides \
          --task_name "$name" --task_config demo_clean \
          --ckpt_setting "$(basename "$(dirname "$(dirname "$checkpoint")")")" \
          --seed "$seed" --policy_name pi_starvla \
          --train_config_name "$(basename "$(dirname "$(dirname "$checkpoint")")")" \
          --log_save_path "$output_dir" --policy_ckpt_path "$checkpoint" \
          --host 127.0.0.1 --port "$port" --unnorm_key new_embodiment) \
        > "$output_dir/$name.eval.log" 2>&1
      rc=$?
      set -e
    else
      rc=1
    fi
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
    if (( rc == 0 )); then echo SUCCESS > "$output_dir/$name.status";
    else echo "FAILED:$rc" > "$output_dir/$name.status"; fi
    echo "[$gpu] $name: $(cat "$output_dir/$name.status")"
  done
)

pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for index in "${!gpu_ids[@]}"; do
  run_worker "$index" "${gpu_ids[index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

printf 'task\tstatus\tsuccess_rate\n' > "$output_dir/summary.tsv"
overall_rc=0
for name in "${tasks[@]}"; do
  status="$(cat "$output_dir/$name.status" 2>/dev/null || echo MISSING)"
  rate="$(grep -Eo '[0-9]+([.][0-9]+)?%' "$output_dir/$name.eval.log" 2>/dev/null | tail -1 | tr -d '%' || true)"
  printf '%s\t%s\t%s\n' "$name" "$status" "$rate" >> "$output_dir/summary.tsv"
  [[ "$status" == SUCCESS ]] || overall_rc=1
done
echo "Summary: $output_dir/summary.tsv"
exit "$overall_rc"
