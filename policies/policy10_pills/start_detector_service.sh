#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/yt-astraf0a5/workspace
PROJECT="$WORKSPACE/physical_ai_submission_policy10"
DOCKER=/usr/local/bin/docker
NAME=physical-ai-trained-detector
IMAGE=nvcr.io/nvidia/pytorch:26.06-py3

test -f "$PROJECT/detector_service.py"
test -f "$WORKSPACE/real_object_classifier_20261031/training/run/weights/best.pt"

if "$DOCKER" ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
    "$DOCKER" rm -f "$NAME" >/dev/null
fi

"$DOCKER" run -d \
    --name "$NAME" \
    --gpus all \
    --network host \
    --ipc host \
    -v "$WORKSPACE:/workspace" \
    -w /workspace/physical_ai_submission_policy10 \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    /workspace/training_six_class_20261030/.venv/bin/python -u \
    detector_service.py >/dev/null

echo "Waiting for the trained Policy 10 full-frame classifier ..."
for _ in $(seq 1 90); do
    if "$PROJECT/.venv/bin/python" -c \
        'import requests; requests.get("http://127.0.0.1:18766/health", timeout=2).raise_for_status()' \
        >/dev/null 2>&1; then
        "$PROJECT/.venv/bin/python" -c \
            'import requests; print(requests.get("http://127.0.0.1:18766/health", timeout=2).json())'
        exit 0
    fi
    if ! "$DOCKER" ps --format '{{.Names}}' | grep -Fxq "$NAME"; then
        "$DOCKER" logs "$NAME" 2>&1 || true
        exit 1
    fi
    sleep 2
done
"$DOCKER" logs --tail 100 "$NAME" 2>&1 || true
exit 1
