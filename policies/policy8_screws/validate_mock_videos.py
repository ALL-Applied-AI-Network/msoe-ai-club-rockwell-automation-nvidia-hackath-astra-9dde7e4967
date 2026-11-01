"""Validate that every encoded mock video opens and contains all source frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    videos = []
    for path in sorted(args.session.glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        readable, _ = capture.read()
        capture.release()
        source_dir = path.with_name(f"{path.stem}_frames")
        source_frames = len(list(source_dir.glob("*.png")))
        videos.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "source_frames": source_frames,
                "encoded_frames": encoded_frames,
                "fps": fps,
                "readable": bool(readable),
                "complete": bool(readable)
                and encoded_frames == source_frames
                and source_frames > 0,
            }
        )
    report = {
        "session": str(args.session),
        "videos": videos,
        "all_complete": len(videos) == 3
        and all(video["complete"] for video in videos),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["all_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
