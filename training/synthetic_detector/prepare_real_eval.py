#!/usr/bin/env python3
"""Create a reproducible, stratified, real-frame evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    indices = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in indices]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_name(relative: Path) -> str:
    return "__".join(relative.parts).replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-run", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[Path, str]] = []
    run_dirs = sorted(
        path for path in source.rglob("*_frames")
        if path.is_dir() and not path.name.startswith("mock-")
    )
    for run_dir in run_dirs:
        frames = sorted(
            path for path in run_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        candidates.extend((path, "physical_run") for path in evenly_spaced(frames, args.frames_per_run))

    cosmos_images = sorted(
        path for path in source.rglob("cosmos/*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    candidates.extend((path, "cosmos_request") for path in cosmos_images)

    seen_hashes: set[str] = set()
    records = []
    duplicate_count = 0
    for path, group in candidates:
        sha256 = digest(path)
        if sha256 in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(sha256)
        relative = path.relative_to(source)
        destination = images_dir / safe_name(relative)
        shutil.copy2(path, destination)
        records.append(
            {
                "image": destination.name,
                "source": str(path),
                "source_relative": str(relative),
                "group": group,
                "sha256": sha256,
                "bytes": destination.stat().st_size,
            }
        )

    report = {
        "source": str(source),
        "frames_per_run": args.frames_per_run,
        "physical_run_directories": len(run_dirs),
        "cosmos_candidates": len(cosmos_images),
        "selected_candidates": len(candidates),
        "duplicates_removed": duplicate_count,
        "unique_images": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
