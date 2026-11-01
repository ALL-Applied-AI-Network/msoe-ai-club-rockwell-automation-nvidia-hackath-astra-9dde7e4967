#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/yt-astraf0a5/workspace/physical_ai_submission_policy9
cd "$PROJECT"
uv sync
uv run python -m unittest discover -s tests -v
uv run run.py --mock --trials 3 --trial-s 8 \
  --instruction "Sort all AAA batteries into the red bin and all AA batteries into the green bin."
echo "Policy 9 preflight passed. No videos, submission, or physical motion were produced."
