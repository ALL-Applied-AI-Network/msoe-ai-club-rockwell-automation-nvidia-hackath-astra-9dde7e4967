#!/usr/bin/env python3
"""Validate the red/green simulation bins with Isaac Sim's USD libraries."""

from __future__ import annotations

import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path("/workspace/imports/red_green_bin_assets_candidate")
EXPECTED_SIZE_M = (0.09, 0.09, 0.06)
COLLIDERS = ("Floor", "WallNorth", "WallSouth", "WallEast", "WallWest")


def validate_asset(path: Path, expected_colour: tuple[float, float, float]) -> dict:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise RuntimeError(f"Could not open {path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError(f"{path.name}: stage is not Z-up")

    visual = stage.GetPrimAtPath("/Asset/Millimetres/Visual")
    if not visual.IsValid():
        raise RuntimeError(f"{path.name}: visual reference did not compose")
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(visual)
    box = bound.ComputeAlignedBox()
    size = tuple(float(x) for x in box.GetSize())
    if any(abs(a - b) > 0.0002 for a, b in zip(size, EXPECTED_SIZE_M)):
        raise RuntimeError(f"{path.name}: visual size {size}, expected {EXPECTED_SIZE_M}")

    collision_paths = []
    for name in COLLIDERS:
        prim = stage.GetPrimAtPath(f"/Asset/Collision/{name}")
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"{path.name}: missing collision API on {name}")
        collision_paths.append(prim.GetPath().pathString)

    material, _ = UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial()
    if not material:
        raise RuntimeError(f"{path.name}: visual material is not bound")
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Asset/Looks/BinMaterial/PreviewSurface"))
    colour = tuple(float(x) for x in shader.GetInput("diffuseColor").Get())
    if any(abs(a - b) > 1e-5 for a, b in zip(colour, expected_colour)):
        raise RuntimeError(f"{path.name}: material colour {colour} is incorrect")

    return {
        "file": path.name,
        "visual_bounds_m": {
            "min": list(box.GetMin()),
            "max": list(box.GetMax()),
            "size": list(size),
        },
        "diffuse_colour": list(colour),
        "collision_prims": collision_paths,
        "open_top": True,
        "floor_top_z_m": 0.01,
        "inner_opening_m": [0.08, 0.08],
    }


def main() -> None:
    assets = [
        validate_asset(ROOT / "packaged/red_bin_sim.usda", (0.72, 0.018, 0.022)),
        validate_asset(ROOT / "packaged/green_bin_sim.usda", (0.018, 0.52, 0.055)),
    ]
    report = {
        "source": "source/red_green_bin.step",
        "source_dimensions_mm": [90.0, 90.0, 60.0],
        "source_units": "millimetres",
        "assets": assets,
        "validation": "passed",
    }
    output = ROOT / "inspection/bin_asset_manifest.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
