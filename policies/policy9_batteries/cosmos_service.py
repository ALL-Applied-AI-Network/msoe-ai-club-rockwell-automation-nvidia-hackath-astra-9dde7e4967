"""Persistent localhost Cosmos3-Nano service for the FANUC demo policy.

Run this inside the Cosmos container with start_cosmos_service.sh.  Keeping the
model in one process avoids paying its load time on every policy observation.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

SIM_ROOT = Path(os.environ.get("FANUC_SIM_ROOT", "/workspace/fanuc_er4ia_sim"))
if str(SIM_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SIM_ROOT))

from perceive import ask, canonicalise, load_model, parse_json, salvage_objects  # noqa: E402


HOST = os.environ.get("COSMOS_DEMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("COSMOS_DEMO_PORT", "18765"))
LOG_ROOT = Path(
    os.environ.get(
        "COSMOS_DEMO_LOG_DIR",
        "/workspace/physical_ai_submission_policy8/runtime_logs/cosmos",
    )
)
CATEGORIES = {"screw", "pill", "cube", "battery"}
COLORS = {"orange", "yellow", "blue", "red", "green"}
AXIS_CLOSE_RE = re.compile(
    r'("axis_endpoints"\s*:\s*\[\s*\[[^\[\]]+\]\s*,\s*\[[^\[\]]+\])'
    r'(\s*,\s*"graspable")'
)
SORT_CLASSES = ("m4_screw", "m8_screw", "ignore")
SORT_BIN_COLORS = ("red", "green")


def _prompt(category: str, object_color: str | None, destination: str) -> str:
    colour_rule = (
        f"Only report {object_color} {category}s as task objects."
        if object_color
        else f"Report every loose {category} as a task object, regardless of colour."
    )
    return f"""You are the perception system for a tabletop FANUC robot.

The requested object category is {category}. The destination is the {destination}
bin. {colour_rule}

Find every separate requested object lying loose on the white work surface and
find the one {destination} collection bin. Do NOT report an object that is already
inside any bin. Ignore the robot, gripper, fingers, screws that fasten the gripper,
table hardware, cables, shadows, and printed labels.

Use visual appearance, not image location, to identify objects and bin colour.
For an elongated screw or battery, axis_endpoints must mark the two ends of its
long axis. For a cube or round pill, axis_endpoints may be null. grasp_center is
the best top-down pinch point and must lie on the object's visible silhouette.

Reply with ONLY one JSON object in exactly this schema:
{{
  "objects": [
    {{"label":"M4 screw", "category":"{category}", "color":"silver",
      "center":[0.45,0.61], "grasp_center":[0.45,0.61],
      "bbox":[0.42,0.58,0.48,0.64],
      "axis_endpoints":[[0.42,0.62],[0.48,0.60]], "graspable":true}}
  ],
  "bins": [
    {{"label":"{destination} bin", "category":"bin", "color":"{destination}",
      "center":[0.30,0.35], "bbox":[0.20,0.25,0.40,0.45]}}
  ]
}}

All coordinates are NORMALIZED to this image: x is 0 at the left and 1 at the
right; y is 0 at the top and 1 at the bottom. bbox is
[x_min,y_min,x_max,y_max]. Give each visible object once. If none remain outside
the bins, return an empty objects list, but still report the destination bin.
Never invent a requested object or infer one from the instruction alone.
"""


def _sort4_prompt() -> str:
    return """You are the perception system for a tabletop FANUC robot.

This image is a tight crop of the loose-object work area. Find every separate
loose screw visible in it. Task objects are only M4 screws and M8 screws. Any
other visible item is a distractor: report it once with sort_class "ignore" and
a descriptive label. Report only what is visibly present. The number of screws
is unknown and may vary. Never infer an inventory from the instruction, repeat
an object, or invent an object to force a count. The destination bins are
outside this crop and must not be reported as objects.

Distinguish M4 from M8 by visible physical geometry and relative scale: M8 has
the larger shaft and head; M4 is smaller. Do not use image position as a class
hint. Ignore shadows, table markings and any non-task item.

For every screw, axis_endpoints must mark the two ends of its long axis and
grasp_center must be the best top-down pinch point on its visible body.

Reply with ONLY JSON in exactly this schema:
{
  "objects": [
    {"label":"M4 screw", "sort_class":"m4_screw", "center":[0.4,0.6],
     "grasp_center":[0.4,0.6], "bbox":[0.38,0.58,0.42,0.62],
     "axis_endpoints":[[0.38,0.59],[0.42,0.61]], "graspable":true},
    {"label":"non-screw distractor", "sort_class":"ignore", "center":[0.6,0.6],
     "grasp_center":[0.6,0.6], "bbox":[0.58,0.58,0.62,0.62],
     "axis_endpoints":null, "graspable":false}
  ]
}

sort_class must be exactly m4_screw, m8_screw or ignore.
Coordinates must be normalized 0..1 relative to this source image, x left to
right and y top to bottom. bbox is [x_min,y_min,x_max,y_max]. Give each visible
physical object exactly once.
"""


def _parse_model_reply(raw: str) -> dict:
    parsed = parse_json(raw)
    if parsed is not None:
        return parsed
    # Cosmos occasionally closes the second point but not the surrounding
    # axis_endpoints list. Repair only that unambiguous schema-local omission.
    repaired = AXIS_CLOSE_RE.sub(r"\1]\2", raw)
    parsed = parse_json(repaired)
    if parsed is not None:
        return parsed

    salvaged = salvage_objects(raw)
    bins = [
        item
        for item in salvaged
        if str(item.get("category", "")).lower() == "bin"
        or "bin" in str(item.get("label", "")).lower()
    ]
    objects = [item for item in salvaged if item not in bins]
    explicitly_empty = bool(re.search(r'"objects"\s*:\s*\[\s*\]', raw))
    if not explicitly_empty and not objects:
        raise ValueError("Cosmos object list is malformed; refusing to infer completion")
    return {"objects": objects, "bins": bins}


def _coordinate_convention(parsed: dict) -> str:
    values = []

    def add(item):
        if isinstance(item, (list, tuple)):
            for value in item:
                if isinstance(value, (list, tuple)):
                    add(value)
                else:
                    try:
                        number = float(value)
                        if np.isfinite(number):
                            values.append(abs(number))
                    except (TypeError, ValueError):
                        pass

    for item in list(parsed.get("objects", [])) + list(parsed.get("bins", [])):
        add(item.get("center"))
        add(item.get("grasp_center"))
        add(item.get("bbox"))
        add(item.get("axis_endpoints"))
    high = max(values, default=1.0)
    if high <= 1.5:
        return "unit"
    if high <= 1000.0:
        return "grid1000"
    return "pixel"


def _point(
    value,
    label: str,
    image_size: tuple[int, int],
    convention: str,
) -> list[float]:
    try:
        result = [float(value[0]), float(value[1])]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"{label} must contain two numbers") from exc
    if not all(np.isfinite(result)):
        raise ValueError(f"{label} contains a non-finite coordinate")
    if convention == "grid1000":
        result = [result[0] / 1000.0, result[1] / 1000.0]
    elif convention == "pixel":
        width, height = image_size
        result = [result[0] / width, result[1] / height]
    if not all(0.0 <= v <= 1.0 for v in result):
        raise ValueError(f"{label} is outside the source image")
    return result


def _normalise_scene(
    parsed: dict,
    category: str,
    destination: str,
    image_size: tuple[int, int],
) -> dict:
    parsed = canonicalise(parsed or {})
    convention = _coordinate_convention(parsed)
    objects = []
    for raw in parsed.get("objects", []):
        item = canonicalise(raw)
        raw_category = str(item.get("category", "")).lower().rstrip("s")
        if category not in raw_category:
            continue
        item["category"] = category
        item["center"] = _point(
            item.get("center"), "object center", image_size, convention
        )
        item["grasp_center"] = _point(
            item.get("grasp_center", item["center"]),
            "object grasp_center",
            image_size,
            convention,
        )
        endpoints = item.get("axis_endpoints")
        if endpoints is not None:
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise ValueError("axis_endpoints must contain two points")
            item["axis_endpoints"] = [
                _point(endpoints[0], "axis endpoint", image_size, convention),
                _point(endpoints[1], "axis endpoint", image_size, convention),
            ]
        item["color"] = str(item.get("colour", item.get("color", "unknown"))).lower()
        item["graspable"] = bool(item.get("graspable", True))
        if not any(
            np.linalg.norm(np.asarray(item["grasp_center"]) - other["grasp_center"])
            < 0.012
            for other in objects
        ):
            objects.append(item)

    bins = []
    for raw in parsed.get("bins", []):
        item = canonicalise(raw)
        colour = str(item.get("colour", item.get("color", ""))).lower()
        if not colour:
            label = str(item.get("label", "")).lower()
            colour = next((c for c in COLORS if re.search(rf"\b{c}\b", label)), "")
        if colour != destination:
            continue
        item["color"] = colour
        item["center"] = _point(
            item.get("center"), "bin center", image_size, convention
        )
        bbox = item.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bin bbox must contain four normalized numbers")
            bbox = [float(value) for value in bbox]
            if not all(np.isfinite(bbox)):
                raise ValueError("bin bbox contains a non-finite coordinate")
            if convention == "grid1000":
                bbox = [value / 1000.0 for value in bbox]
            elif convention == "pixel":
                width, height = image_size
                bbox = [
                    bbox[0] / width,
                    bbox[1] / height,
                    bbox[2] / width,
                    bbox[3] / height,
                ]
            if not all(0.0 <= v <= 1.0 for v in bbox):
                raise ValueError("bin bbox is outside the source image")
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("bin bbox has inverted bounds")
            item["bbox"] = bbox
        bins.append(item)
    if len(bins) != 1:
        raise ValueError(
            f"Expected exactly one {destination} bin, Cosmos returned {len(bins)}"
        )
    # The model is asked to exclude items already sorted; enforce that again
    # geometrically whenever it supplied the destination-bin silhouette.
    bin_bbox = bins[0].get("bbox")
    if bin_bbox:
        x0, y0, x1, y1 = bin_bbox
        objects = [
            item
            for item in objects
            if not (
                x0 <= item["grasp_center"][0] <= x1
                and y0 <= item["grasp_center"][1] <= y1
            )
        ]
    return {"objects": objects, "bins": bins}


def _normalise_bbox(
    value,
    label: str,
    image_size: tuple[int, int],
    convention: str,
):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    bbox = [float(item) for item in value]
    if not all(np.isfinite(bbox)):
        raise ValueError(f"{label} contains a non-finite coordinate")
    if convention == "grid1000":
        bbox = [item / 1000.0 for item in bbox]
    elif convention == "pixel":
        width, height = image_size
        bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
    if not all(0.0 <= item <= 1.0 for item in bbox):
        raise ValueError(f"{label} is outside the source image")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"{label} has inverted bounds")
    return bbox


def _derive_sort_class(item: dict) -> str:
    explicit = str(item.get("sort_class", "")).lower()
    if explicit in SORT_CLASSES:
        return explicit
    label = " ".join(
        str(item.get(key, "")).lower() for key in ("label", "category", "colour", "color")
    )
    if "m4" in label and "screw" in label:
        return "m4_screw"
    if "m8" in label and "screw" in label:
        return "m8_screw"
    if "distractor" in label or "screw" not in label:
        return "ignore"
    return ""


def _normalise_sort4_scene(
    parsed: dict,
    image_size: tuple[int, int],
    require_bins: bool = True,
) -> dict:
    parsed = canonicalise(parsed or {})
    convention = _coordinate_convention(parsed)
    objects = []
    for raw in parsed.get("objects", []):
        item = canonicalise(raw)
        sort_class = _derive_sort_class(item)
        if not sort_class:
            continue
        item["sort_class"] = sort_class
        item["center"] = _point(
            item.get("center"), "object center", image_size, convention
        )
        item["grasp_center"] = _point(
            item.get("grasp_center", item["center"]),
            "object grasp_center",
            image_size,
            convention,
        )
        item["bbox"] = _normalise_bbox(
            item.get("bbox"), "object bbox", image_size, convention
        )
        endpoints = item.get("axis_endpoints")
        if "screw" in sort_class:
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                raise ValueError(f"{sort_class} is missing two axis endpoints")
            item["axis_endpoints"] = [
                _point(endpoints[0], "axis endpoint", image_size, convention),
                _point(endpoints[1], "axis endpoint", image_size, convention),
            ]
        else:
            item["axis_endpoints"] = None
        item["graspable"] = bool(item.get("graspable", True))
        if not any(
            item["sort_class"] == other["sort_class"]
            and np.linalg.norm(
                np.asarray(item["grasp_center"]) - np.asarray(other["grasp_center"])
            )
            < 0.012
            for other in objects
        ):
            objects.append(item)

    bins = []
    for raw in parsed.get("bins", []):
        item = canonicalise(raw)
        color = str(item.get("colour", item.get("color", ""))).lower()
        if not color:
            label = str(item.get("label", "")).lower()
            color = next((name for name in SORT_BIN_COLORS if name in label), "")
        if color not in SORT_BIN_COLORS:
            continue
        item["color"] = color
        item["center"] = _point(
            item.get("center"), "bin center", image_size, convention
        )
        item["bbox"] = _normalise_bbox(
            item.get("bbox"), "bin bbox", image_size, convention
        )
        if any(existing["color"] == color for existing in bins):
            raise ValueError(f"Cosmos returned more than one {color} bin")
        bins.append(item)
    missing = sorted(set(SORT_BIN_COLORS) - {item["color"] for item in bins})
    if require_bins and missing:
        raise ValueError("Cosmos missed bins: " + ", ".join(missing))

    for bin_item in bins:
        bbox = bin_item.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        objects = [
            item
            for item in objects
            if not (
                x0 <= item["grasp_center"][0] <= x1
                and y0 <= item["grasp_center"][1] <= y1
            )
        ]
    return {"objects": objects, "bins": bins}


class CosmosRuntime:
    def __init__(self):
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        self.model, self.processor = load_model(int(os.environ.get("COSMOS_GPU", "0")))
        self.lock = threading.Lock()
        self.counter = 0

    def infer(self, request: dict) -> dict:
        category = str(request.get("category", "")).lower()
        destination = str(request.get("destination_color", "")).lower()
        object_color = request.get("object_color")
        object_color = str(object_color).lower() if object_color else None
        if category not in CATEGORIES or destination not in COLORS:
            raise ValueError("Unsupported category or destination colour")
        encoded = request.get("image_jpeg_base64")
        if not isinstance(encoded, str) or len(encoded) > 20_000_000:
            raise ValueError("Missing or oversized JPEG payload")
        try:
            binary = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image_jpeg_base64 is invalid") from exc
        image = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG could not be decoded")

        with self.lock:
            self.counter += 1
            stem = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{self.counter:04d}"
            image_path = LOG_ROOT / f"{stem}.jpg"
            cv2.imwrite(str(image_path), image)
            raw = ask(
                self.model,
                self.processor,
                image_path,
                _prompt(category, object_color, destination),
                1800,
            )
            (LOG_ROOT / f"{stem}_raw.txt").write_text(raw, encoding="utf-8")

        parsed = _parse_model_reply(raw)
        height, width = image.shape[:2]
        scene = _normalise_scene(parsed, category, destination, (width, height))
        report = {
            "status": "PASS",
            **scene,
            "model": "Cosmos3-Nano",
            "evidence": stem,
        }
        (LOG_ROOT / f"{stem}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    def infer_sort4(self, request: dict) -> dict:
        encoded = request.get("image_jpeg_base64")
        if not isinstance(encoded, str) or len(encoded) > 20_000_000:
            raise ValueError("Missing or oversized JPEG payload")
        try:
            binary = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image_jpeg_base64 is invalid") from exc
        image = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG could not be decoded")

        with self.lock:
            self.counter += 1
            stem = f"sort4_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{self.counter:04d}"
            image_path = LOG_ROOT / f"{stem}.jpg"
            cv2.imwrite(str(image_path), image)
            raw = ask(
                self.model,
                self.processor,
                image_path,
                _sort4_prompt(),
                2000,
            )
            (LOG_ROOT / f"{stem}_raw.txt").write_text(raw, encoding="utf-8")

        parsed = _parse_model_reply(raw)
        height, width = image.shape[:2]
        scene = _normalise_sort4_scene(parsed, (width, height), require_bins=False)
        report = {
            "status": "PASS",
            **scene,
            "model": "Cosmos3-Nano",
            "evidence": stem,
        }
        (LOG_ROOT / f"{stem}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report


RUNTIME: CosmosRuntime | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[cosmos-service] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"status": "ok", "model": "Cosmos3-Nano"})
        return self._send(404, {"status": "ERROR", "error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path not in ("/infer", "/infer-sort4"):
            return self._send(404, {"status": "ERROR", "error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 21_000_000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            assert RUNTIME is not None
            result = (
                RUNTIME.infer_sort4(request)
                if self.path == "/infer-sort4"
                else RUNTIME.infer(request)
            )
            return self._send(200, result)
        except Exception as exc:  # keep policy failures structured and auditable
            print(f"[cosmos-service] inference failed: {exc}", flush=True)
            return self._send(422, {"status": "ERROR", "error": str(exc)})


def main() -> None:
    global RUNTIME
    RUNTIME = CosmosRuntime()
    print(f"[cosmos-service] ready on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
