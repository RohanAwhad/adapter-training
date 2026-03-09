#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-hermes-train}
CUDA_DEVICE=${CUDA_DEVICE:-0}
REPO_DIR=${REPO_DIR:-$(pwd)}

tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
  echo "tmux session already exists: $SESSION_NAME"
  exit 1
}

CMD="cd \"$REPO_DIR\" && export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE && uv run python scripts/train_lora.py"
tmux new -d -s "$SESSION_NAME" "$CMD"

echo "started tmux session: $SESSION_NAME"
echo "attach with: tmux attach -t $SESSION_NAME"
