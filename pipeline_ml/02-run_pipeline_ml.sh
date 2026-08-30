#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CKPT="results/train_reference/checkpoints/best.pt"
RESOLVED_CKPT=""

resolve_checkpoint() {
  if [[ -f "$DEFAULT_CKPT" ]]; then
    RESOLVED_CKPT="$DEFAULT_CKPT"
    return
  fi

  local tagged
  tagged=$(ls -1t "results/train_reference/checkpoints"/*_best.pt 2>/dev/null | head -n 1 || true)
  if [[ -n "$tagged" && -f "$tagged" ]]; then
    RESOLVED_CKPT="$tagged"
    return
  fi

  RESOLVED_CKPT=""
}

require_default_checkpoint() {
  resolve_checkpoint
  if [[ -z "$RESOLVED_CKPT" ]]; then
    echo "[ERROR] Missing checkpoint: $DEFAULT_CKPT"
    echo "[ERROR] Missing tagged checkpoints: results/train_reference/checkpoints/*_best.pt"
    echo "Run training first: bash 02-run_pipeline_ml.sh train --device cpu"
    exit 1
  fi
}

ACTION="${1:-train}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$ACTION" in
  train)
    python scripts/train.py \
      --run-name train_reference \
      --epochs 10 \
      --batch-size 4 \
      "$@"
    ;;
  validate)
    require_default_checkpoint
    python scripts/validate.py \
      --checkpoint "$RESOLVED_CKPT" \
      --split val \
      --run-name val_reference \
      "$@"
    ;;
  test)
    require_default_checkpoint
    python scripts/test.py \
      --checkpoint "$RESOLVED_CKPT" \
      --run-name test_reference \
      --save-max-preds 1000000 \
      "$@"
    ;;
  *)
    echo "Usage: bash 02-run_pipeline_ml.sh [train|validate|test] [extra args]"
    exit 1
    ;;
esac
