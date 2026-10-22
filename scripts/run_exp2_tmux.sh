#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec uv run python scripts/gpu_telemetry_runner.py \
  --run-exp2 \
  --models barlowtwins resnet152 densenet161 efficientnetb3 vgg16 vgg19 clip_vit_b16 \
  --xai-methods both \
  2>&1 | tee -a Outputs/Experiment_2_variants/exp2_cnns_tmux.log
