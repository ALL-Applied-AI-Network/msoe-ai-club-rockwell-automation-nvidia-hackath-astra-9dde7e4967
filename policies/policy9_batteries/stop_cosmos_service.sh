#!/usr/bin/env bash
set -euo pipefail

DOCKER=/usr/local/bin/docker
NAME=physical-ai-cosmos-demo

if "$DOCKER" ps -a --format '{{.Names}}' | grep -Fxq "$NAME"; then
    "$DOCKER" stop "$NAME" >/dev/null
    echo "Stopped $NAME"
else
    echo "$NAME is not running"
fi
