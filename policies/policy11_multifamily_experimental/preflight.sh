#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/yt-astraf0a5/workspace/physical_ai_submission_policy11
cd "$PROJECT"
uv sync
uv run python -m unittest discover -s tests -v
uv run run.py --mock --trials 1 --trial-s 8 \
  --instruction "Sort all screws into the red bin, all batteries into the green bin, and all pills into the yellow bin."
uv run run.py --mock --trials 1 --trial-s 8 \
  --instruction "Sort all M4 screws into the blue bin, all M8 screws into the green bin, all round pills into the yellow bin, and all long pills into the red bin."
echo "Policy 11 preflight passed. No videos, submission, or physical motion were produced."
