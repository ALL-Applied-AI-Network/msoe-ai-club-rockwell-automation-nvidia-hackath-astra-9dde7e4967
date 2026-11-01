#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/yt-astraf0a5/workspace
PROJECT="$WORKSPACE/physical_ai_submission_policy8"
DOCKER=/usr/local/bin/docker
NAME=physical-ai-cosmos-demo
IMAGE=nvcr.io/nvidia/pytorch:26.06-py3

test -f "$PROJECT/cosmos_service.py"
test -f "$WORKSPACE/projects/cosmos3/models/Cosmos3-Nano/config.json"

if "$DOCKER" ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
    "$DOCKER" rm -f "$NAME" >/dev/null
fi

"$DOCKER" run -d \
    --name "$NAME" \
    --gpus all \
    --network host \
    --ipc host \
    -v "$WORKSPACE:/workspace" \
    -e COSMOS3_NANO_PATH=/workspace/projects/cosmos3/models/Cosmos3-Nano \
    -e COSMOS3_ROOT=/workspace/projects/cosmos3/cosmos-framework \
    -e HF_HOME=/workspace/projects/cosmos3/hf-cache \
    -e HF_HUB_OFFLINE=1 \
    -e COSMOS_DEMO_LOG_DIR=/workspace/physical_ai_submission_policy8/runtime_logs/cosmos \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    /workspace/projects/cosmos3/venv/bin/python -u \
    /workspace/physical_ai_submission_policy8/cosmos_service.py >/dev/null

echo "Waiting for Cosmos3-Nano to load (container: $NAME) ..."
for _ in $(seq 1 150); do
    if "$PROJECT/.venv/bin/python" -c \
        'import requests; requests.get("http://127.0.0.1:18765/health", timeout=2).raise_for_status()' \
        >/dev/null 2>&1; then
        echo "Cosmos service ready at http://127.0.0.1:18765"
        exit 0
    fi
    if ! "$DOCKER" ps --format '{{.Names}}' | grep -Fxq "$NAME"; then
        "$DOCKER" logs "$NAME" 2>&1 || true
        echo "Cosmos service exited before becoming ready" >&2
        exit 1
    fi
    sleep 2
done

"$DOCKER" logs --tail 100 "$NAME" 2>&1 || true
echo "Cosmos service did not become ready within five minutes" >&2
exit 1
