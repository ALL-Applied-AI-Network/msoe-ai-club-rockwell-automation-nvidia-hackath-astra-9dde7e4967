#!/usr/bin/env python3
"""Create verified object-only crops from real CSI scan frames and policy logs."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
IMAGE_ROOT = RESULTS / "object_images_20261030"
IMAGE_MANIFEST = IMAGE_ROOT / "manifest.csv"
CAMERA_FILE = (
    RESULTS
    / "policy10_build/physical_ai_submission_policy10/calibration/cameras.json"
)
LOG_ROOT = ROOT / "source_logs"
OUTPUT = ROOT / "real_crops"
CLASSES = (
    "m4_screw",
    "m8_screw",
    "round_pill",
    "long_pill",
    "aa_battery",
    "aaa_battery",
)
CROP_SIDE = {
    "m4_screw": 96,
    "m8_screw": 112,
    "round_pill": 64,
    "long_pill": 80,
    "aa_battery": 96,
    "aaa_battery": 96,
}


def project_table(point_m: tuple[float, float, float], camera: dict) -> tuple[float, float]:
    transform = camera["T_base_cam"]
    rotation = [row[:3] for row in transform[:3]]
    translation = [row[3] for row in transform[:3]]
    shifted = [point_m[i] - translation[i] for i in range(3)]
    camera_point = [
        sum(rotation[j][i] * shifted[j] for j in range(3)) for i in range(3)
    ]
    x = camera_point[0] / camera_point[2]
    y = camera_point[1] / camera_point[2]
    distortion = 1.0 + camera["k1"] * (x * x + y * y)
    width, height = camera["size"]
    return (
        camera["fx"] * x * distortion + width / 2.0,
        camera["fx"] * y * distortion + height / 2.0,
    )


def first_batch_plan(path: Path) -> dict | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "batch_plan" and event.get("targets"):
            return event
    return None


def battery_targets(image: Image.Image) -> list[dict]:
    """Find the two dark battery bodies in the calibrated loose-object area."""

    thumbnail = image.resize((640, 360), Image.Resampling.BILINEAR).convert("RGB")
    pixels = thumbnail.load()
    width, height = thumbnail.size
    mask = bytearray(width * height)
    for y in range(190, 340):
        for x in range(180, 480):
            red, green, blue = pixels[x, y]
            if (red + green + blue) / 3.0 < 115 and max(red, green, blue) - min(red, green, blue) < 38:
                mask[y * width + x] = 1

    seen = bytearray(width * height)
    components = []
    for y in range(190, 340):
        for x in range(180, 480):
            start = y * width + x
            if not mask[start] or seen[start]:
                continue
            stack = [start]
            seen[start] = 1
            xs, ys = [], []
            while stack:
                index = stack.pop()
                yy, xx = divmod(index, width)
                xs.append(xx)
                ys.append(yy)
                for adjacent in (index - 1, index + 1, index - width, index + width):
                    if 0 <= adjacent < width * height and mask[adjacent] and not seen[adjacent]:
                        seen[adjacent] = 1
                        stack.append(adjacent)
            area = len(xs)
            box_width = max(xs) - min(xs) + 1
            box_height = max(ys) - min(ys) + 1
            long_side, short_side = max(box_width, box_height), min(box_width, box_height)
            if 45 <= area <= 180 and 12 <= long_side <= 28 and 5 <= short_side <= 16:
                components.append(
                    {
                        "area": area,
                        "center": ((min(xs) + max(xs) + 1) / 2.0, (min(ys) + max(ys) + 1) / 2.0),
                        "long_side": long_side,
                    }
                )
    components.sort(key=lambda item: item["area"], reverse=True)
    if len(components) < 2:
        raise RuntimeError(f"Expected two verified battery components, found {components}")
    chosen = components[:2]
    if abs(chosen[0]["long_side"] - chosen[1]["long_side"]) >= 2:
        chosen.sort(key=lambda item: item["long_side"], reverse=True)
    else:
        chosen.sort(key=lambda item: item["area"], reverse=True)
    scale_x, scale_y = image.width / 640.0, image.height / 360.0
    return [
        {
            "class": "aa_battery",
            "pixel_center": [chosen[0]["center"][0] * scale_x, chosen[0]["center"][1] * scale_y],
            "label_source": "verified dark-body geometry",
        },
        {
            "class": "aaa_battery",
            "pixel_center": [chosen[1]["center"][0] * scale_x, chosen[1]["center"][1] * scale_y],
            "label_source": "verified dark-body geometry",
        },
    ]


def crop_square(image: Image.Image, center: tuple[float, float], side: int) -> Image.Image:
    half = side / 2.0
    box = (
        int(round(center[0] - half)),
        int(round(center[1] - half)),
        int(round(center[0] + half)),
        int(round(center[1] + half)),
    )
    return ImageOps.pad(image.crop(box), (side, side), color=(240, 240, 240))


def session_split(session_names: list[str]) -> dict[str, str]:
    ordered = sorted(session_names)
    if len(ordered) < 3:
        raise RuntimeError(f"Need at least three independent sessions, got {ordered}")
    return {
        session: (
            "test" if index == len(ordered) - 1 else "val" if index == len(ordered) - 2 else "train"
        )
        for index, session in enumerate(ordered)
    }


def make_contact_sheet(class_name: str, records: list[dict]) -> None:
    tiles = []
    for record in records:
        image = Image.open(ROOT / record["crop_file"]).convert("RGB")
        tile = Image.new("RGB", (180, 205), "white")
        tile.paste(ImageOps.fit(image, (180, 180)), (0, 0))
        ImageDraw.Draw(tile).text((4, 184), f"{record['split']} {record['session'][-16:]}", fill="black")
        tiles.append(tile)
    columns = min(6, max(1, len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 180, rows * 205), (220, 220, 220))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 180, (index // columns) * 205))
    (ROOT / "review").mkdir(exist_ok=True)
    sheet.save(ROOT / "review" / f"{class_name}_contact.jpg", quality=92)


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (OUTPUT / split / class_name).mkdir(parents=True, exist_ok=True)

    sessions: dict[str, dict] = {}
    with IMAGE_MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["source_kind"] != "real_csi_policy_frame":
                continue
            source_parts = row["original_source"].split("/runtime_logs/", 1)
            if len(source_parts) != 2:
                continue
            relative = Path(source_parts[1])
            if len(relative.parts) != 3:
                continue
            session, frame_dir, frame_name = relative.parts
            if frame_name not in {"frame_00000.png", "frame_00001.png"}:
                continue
            policy = int(row["policy"])
            trial = frame_dir.removesuffix("_frames")
            entry = sessions.setdefault(
                session,
                {
                    "policy": policy,
                    "category": row["category"],
                    "trial": trial,
                    "frames": {},
                },
            )
            entry["frames"][frame_name] = IMAGE_ROOT / row["dataset_file"]

    valid_sessions: dict[str, dict] = {}
    camera = json.loads(CAMERA_FILE.read_text(encoding="utf-8"))["table"]
    for session, entry in sessions.items():
        if "frame_00000.png" not in entry["frames"]:
            continue
        log_path = LOG_ROOT / f"policy{entry['policy']}" / session / f"{entry['trial']}.jsonl"
        if entry["policy"] == 9:
            entry["targets"] = battery_targets(Image.open(entry["frames"]["frame_00000.png"]).convert("RGB"))
        else:
            if not log_path.is_file():
                continue
            plan = first_batch_plan(log_path)
            if not plan:
                continue
            targets = []
            for target in plan["targets"]:
                x_mm, y_mm = target["object_xy_mm"]
                center = project_table((x_mm / 1000.0, y_mm / 1000.0, 0.0), camera)
                targets.append(
                    {
                        "class": target["class"],
                        "pixel_center": list(center),
                        "label_source": "policy batch_plan metric target",
                    }
                )
            entry["targets"] = targets
        valid_sessions[session] = entry

    family_sessions = defaultdict(list)
    for session, entry in valid_sessions.items():
        family_sessions[entry["policy"]].append(session)
    splits = {}
    for policy, names in family_sessions.items():
        splits.update(session_split(names))

    records = []
    counters = defaultdict(int)
    for session, entry in sorted(valid_sessions.items()):
        split = splits[session]
        for frame_name, frame_path in sorted(entry["frames"].items()):
            image = Image.open(frame_path).convert("RGB")
            for target_index, target in enumerate(entry["targets"]):
                class_name = target["class"]
                side = CROP_SIDE[class_name]
                center = tuple(target["pixel_center"])
                crop = crop_square(image, center, side)
                counters[(split, class_name)] += 1
                name = (
                    f"{class_name}__{session}__{entry['trial']}__"
                    f"{frame_name[:-4]}__target{target_index:02d}.jpg"
                )
                destination = OUTPUT / split / class_name / name
                crop.save(destination, quality=96, subsampling=0)
                records.append(
                    {
                        "split": split,
                        "class": class_name,
                        "session": session,
                        "trial": entry["trial"],
                        "frame": frame_name,
                        "source_image": str(frame_path.relative_to(RESULTS)),
                        "crop_file": str(destination.relative_to(ROOT)),
                        "center_x": round(center[0], 3),
                        "center_y": round(center[1], 3),
                        "crop_side_px": side,
                        "label_source": target["label_source"],
                        "bins_included": "no",
                    }
                )

    with (ROOT / "real_crop_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    for class_name in CLASSES:
        make_contact_sheet(class_name, [record for record in records if record["class"] == class_name])

    report = {
        "status": "complete",
        "classes": list(CLASSES),
        "sessions_by_policy": {str(key): len(value) for key, value in sorted(family_sessions.items())},
        "crops": {
            split: {class_name: counters[(split, class_name)] for class_name in CLASSES}
            for split in ("train", "val", "test")
        },
        "total_crops": len(records),
        "bins_in_training": False,
        "split_rule": "complete runtime sessions; no session crosses splits",
    }
    (ROOT / "real_crop_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
