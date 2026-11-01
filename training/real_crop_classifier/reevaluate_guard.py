#!/usr/bin/env python3
"""Re-evaluate the selected proposal config with the camera-clear safety guard."""

import json

import numpy as np
from ultralytics import YOLO

from full_frame_detector import (
    CAMERA_FILE,
    MODEL,
    OUTPUT,
    ProposalConfig,
    evaluate,
    load_frames,
    project_table,
)


def main() -> None:
    original = json.loads((OUTPUT / "full_frame_report.json").read_text())
    config = ProposalConfig(**original["best_config"])
    camera = json.loads(CAMERA_FILE.read_text())["table"]
    workspace_polygon = np.round(
        project_table(
            [
                (x, y, -0.0051)
                for x, y in ((0.27, -0.28), (0.50, -0.28), (0.50, 0.28), (0.27, 0.28))
            ],
            camera,
        )
    ).astype(np.int32)
    frames = list(load_frames().values())
    model = YOLO(str(MODEL))
    report = {
        "status": "complete",
        "proposal_parameters_reused_from_train_plus_validation_selection": True,
        "workspace_polygon_px": workspace_polygon.tolist(),
        "best_config": config.__dict__,
        "validation": evaluate(
            [frame for frame in frames if frame["split"] == "val"],
            config,
            workspace_polygon,
            model,
            "guarded_validation",
        ),
        "test": evaluate(
            [frame for frame in frames if frame["split"] == "test"],
            config,
            workspace_polygon,
            model,
            "guarded_test",
        ),
    }
    path = OUTPUT / "guarded_full_frame_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    compact = dict(report)
    compact["validation"] = {
        key: value for key, value in report["validation"].items() if key != "frame_records"
    }
    compact["test"] = {
        key: value for key, value in report["test"].items() if key != "frame_records"
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
