#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/yt-astraf0a5/workspace/physical_ai_submission_policy8
UV=/root/.local/bin/uv

cd "$PROJECT"
"$UV" sync
"$UV" run python -m unittest discover -s tests -v

"$UV" run run.py --mock --trials 3 --trial-s 8 \
    --instruction "Sort all M4 screws into the red bin and all M8 screws into the green bin."

echo "Policy 8 preflight passed. No videos, physical motion, or submission were produced."
