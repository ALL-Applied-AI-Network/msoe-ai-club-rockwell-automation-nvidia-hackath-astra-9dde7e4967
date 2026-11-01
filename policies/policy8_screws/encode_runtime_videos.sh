#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/yt-astraf0a5/workspace/physical_ai_submission_policy4
ENCODER=/home/yt-astraf0a5/workspace/fanuc_er4ia_sim/encode_video.py
PYTHON=/home/yt-astraf0a5/workspace/fanuc_er4ia_sim/.venv/bin/python

session="${1:-}"
if [[ -z "$session" ]]; then
    session=$(find "$PROJECT/runtime_logs" -maxdepth 1 -type d -name 'policy4_*' -printf '%T@ %p\n' \
        | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
test -n "$session"
test -d "$session"

made=0
while IFS= read -r frames; do
    out="${frames%_frames}.mp4"
    # The official mock exposes decision observations, not continuous rendered
    # motion. One frame per second makes this keyframe audit easy to inspect.
    "$PYTHON" "$ENCODER" --frames "$frames" --out "$out" --fps 1
    made=$((made + 1))
done < <(find "$session" -mindepth 1 -maxdepth 1 -type d -name '*_frames' | sort)

if [[ "$made" -eq 0 ]]; then
    echo "No runtime frame folders found in $session" >&2
    exit 1
fi
echo "Encoded $made rollout video(s) in $session"
