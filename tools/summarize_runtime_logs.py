#!/usr/bin/env python3
"""Summarize Astra policy JSONL telemetry without copying camera data.

The policy controllers write one JSON object per state-machine event. This
utility extracts trial-level evidence suitable for audit tables while keeping
raw CSI images, robot observations, and credentials out of the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


POLICY_PATTERN = re.compile(r"physical_ai_submission_policy(?P<number>\d+)")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def infer_policy(path: Path) -> int | None:
    for part in path.parts:
        match = POLICY_PATTERN.fullmatch(part)
        if match:
            return int(match.group("number"))
    return None


def summarize(path: Path, root: Path) -> dict | None:
    rows = load_rows(path)
    if not rows:
        return None
    instruction = next(
        (row.get("instruction") for row in rows if row.get("event") == "instruction"),
        None,
    )
    plan = next((row for row in rows if row.get("event") == "batch_plan"), {})
    complete = next(
        (row for row in reversed(rows) if row.get("event") == "complete"), None
    )
    pickup_counts = Counter(
        row.get("sort_class")
        for row in rows
        if row.get("event") == "pickup_verified" and row.get("sort_class")
    )
    failure_events = [
        str(row.get("event"))
        for row in rows
        if any(
            word in str(row.get("event", "")).lower()
            for word in ("failure", "abort", "error", "blocked", "rejected")
        )
    ]
    start = rows[0].get("time")
    end = rows[-1].get("time")
    duration = end - start if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
    return {
        "policy": infer_policy(path),
        "session": path.parent.name,
        "trial": path.stem,
        "kind": "mock" if path.stem.startswith("mock-") else "physical",
        "rows": len(rows),
        "duration_s": round(duration, 3) if duration is not None else None,
        "instruction": instruction,
        "planned_inventory": plan.get("inventory"),
        "verified_pickups": dict(sorted(pickup_counts.items())),
        "complete": complete is not None,
        "completed_inventory": complete.get("completed") if complete else None,
        "failure_events": failure_events,
        "final_event": rows[-1].get("event"),
        "final_phase": rows[-1].get("phase"),
        "source": str(path.relative_to(root)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root containing policy runtime_logs")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(args.root.rglob("*.jsonl")):
        record = summarize(path, args.root)
        if record is not None:
            records.append(record)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(records, indent=2) + "\n")
    columns = [
        "policy",
        "session",
        "trial",
        "kind",
        "rows",
        "duration_s",
        "instruction",
        "planned_inventory",
        "verified_pickups",
        "complete",
        "completed_inventory",
        "failure_events",
        "final_event",
        "final_phase",
        "source",
    ]
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in (
                "planned_inventory",
                "verified_pickups",
                "completed_inventory",
                "failure_events",
            ):
                row[key] = json.dumps(row[key], sort_keys=True)
            writer.writerow(row)
    print(f"wrote {len(records)} trial summaries")


if __name__ == "__main__":
    main()
