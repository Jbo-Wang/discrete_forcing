#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
destination="${1:-datasets}"
mkdir -p "$destination"
for suite in spatial object goal 10; do
  dataset="libero_${suite}_no_noops_1.0.0_lerobot"
  hf download "IPEC-COMMUNITY/$dataset" --repo-type dataset \
    --local-dir "$destination/$dataset"
  cp examples/LIBERO/train_files/modality.json "$destination/$dataset/meta/modality.json"
done
