"""Structural and dimensional validation for the new pill/battery assets."""

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()
root = Path(args.root).resolve()
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


manifest = json.loads(
    (root / "inspection/object_asset_manifest.json").read_text(encoding="utf-8")
)
rows = []
passed = True
for shape, item in manifest["assets"].items():
    for variant_name, variant in item["variants"].items():
        path = root / variant["sim_asset"]
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
        default = stage.GetDefaultPrim() if stage else Usd.Prim()
        prims = (
            list(Usd.PrimRange(default, Usd.TraverseInstanceProxies()))
            if default and default.IsValid()
            else []
        )
        meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
        cylinders = [prim for prim in prims if prim.IsA(UsdGeom.Cylinder)]
        colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
        rigid = bool(default and default.HasAPI(UsdPhysics.RigidBodyAPI))
        mass = (
            float(UsdPhysics.MassAPI(default).GetMassAttr().Get() or 0.0)
            if default and default.HasAPI(UsdPhysics.MassAPI)
            else 0.0
        )
        bound_material, _ = UsdShade.MaterialBindingAPI(
            stage.GetPrimAtPath("/Asset/Millimetres/VisualOffset")
        ).ComputeBoundMaterial()
        dependencies = [layer.realPath for layer in stage.GetUsedLayers() if layer.realPath]
        missing_dependencies = [item for item in dependencies if not Path(item).exists()]
        is_battery = shape.endswith("_battery")
        terminal_contract = len(cylinders) == 2 if is_battery else len(cylinders) == 0
        row_pass = bool(
            stage
            and default.IsValid()
            and len(meshes) >= 2
            and len(colliders) == 1
            and rigid
            and mass > 0.0
            and bound_material
            and terminal_contract
            and not missing_dependencies
        )
        passed = passed and row_pass
        rows.append(
            {
                "asset": path.name,
                "shape": shape,
                "variant": variant_name,
                "status": "PASS" if row_pass else "FAIL",
                "expected_dimensions_mm": item["dimensions_mm"],
                "resolved_meshes": len(meshes),
                "collision_meshes": len(colliders),
                "terminal_cylinders": len(cylinders),
                "rigid_body": rigid,
                "mass_kg": mass,
                "material_bound": bool(bound_material),
                "missing_dependencies": missing_dependencies,
            }
        )


def sorted_dimensions(shape):
    return sorted(float(value) for value in manifest["assets"][shape]["dimensions_mm"])


round_dims = sorted_dimensions("round_pill")
long_dims = sorted_dimensions("long_pill")
aa_dims = sorted_dimensions("aa_battery")
aaa_dims = sorted_dimensions("aaa_battery")
pill_contract = bool(
    long_dims[2] / max(long_dims[1], 1e-9) > 1.5
    and long_dims[2] > round_dims[2]
)
battery_contract = bool(
    aa_dims[2] > aaa_dims[2]
    and (aa_dims[0] + aa_dims[1]) / 2.0 > (aaa_dims[0] + aaa_dims[1]) / 2.0
)
v2_opening_mm = 16.4
grasp_contract = {
    shape: ((dims[0] + dims[1]) / 2.0) < v2_opening_mm
    for shape, dims in (
        ("round_pill", round_dims),
        ("long_pill", long_dims),
        ("aa_battery", aa_dims),
        ("aaa_battery", aaa_dims),
    )
}
contracts_pass = pill_contract and battery_contract and all(grasp_contract.values())
passed = passed and contracts_pass

report = {
    "status": "PASS" if passed else "FAIL",
    "old_pill_assets_reused": manifest.get("old_pill_assets_reused"),
    "contracts": {
        "pill_shape_separation": "PASS" if pill_contract else "FAIL",
        "aa_larger_than_aaa": "PASS" if battery_contract else "FAIL",
        "v2_opening_mm": v2_opening_mm,
        "cross_axis_grasp_fits_v2": grasp_contract,
    },
    "assets": rows,
}
output = root / "inspection/isaac_asset_validation.json"
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
app.close()
raise SystemExit(0 if passed else 1)
