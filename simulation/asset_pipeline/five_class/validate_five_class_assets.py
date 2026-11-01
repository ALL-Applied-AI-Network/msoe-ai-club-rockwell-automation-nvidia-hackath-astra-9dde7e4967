"""Fast structural validation of the packaged Isaac Sim object assets."""

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
    (root / "inspection" / "object_asset_manifest.json").read_text(
        encoding="utf-8"
    )
)
expected = {}
for shape, item in manifest["assets"].items():
    dimensions_m = [float(value) * 0.001 for value in item["dimensions_mm"]]
    for color, variant in item["variants"].items():
        expected[Path(variant["sim_asset"]).name] = {
            "shape": shape,
            "color": color,
            "dimensions_m": dimensions_m,
        }
rows = []
passed = True
for name, spec in expected.items():
    path = root / "packaged" / name
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    default = stage.GetDefaultPrim() if stage else Usd.Prim()
    prims = (
        list(Usd.PrimRange(default, Usd.TraverseInstanceProxies()))
        if default and default.IsValid()
        else []
    )
    meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
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
    row_pass = bool(
        stage
        and default.IsValid()
        and len(meshes) >= 2
        and len(colliders) == 1
        and rigid
        and mass > 0.0
        and bound_material
        and not missing_dependencies
    )
    passed = passed and row_pass
    rows.append(
        {
            "asset": name,
            "shape": spec["shape"],
            "color": spec["color"],
            "status": "PASS" if row_pass else "FAIL",
            "expected_dimensions_m": spec["dimensions_m"],
            "resolved_meshes": len(meshes),
            "collision_meshes": len(colliders),
            "rigid_body": rigid,
            "mass_kg": mass,
            "material_bound": bool(bound_material),
            "resolved_layers": dependencies,
            "missing_dependencies": missing_dependencies,
        }
    )

long_size = manifest["assets"]["long_pill"]["dimensions_mm"]
round_size = manifest["assets"]["round_pill"]["dimensions_mm"]
long_aspect = max(long_size) / max(min(long_size), 1e-9)
round_aspect = max(round_size) / max(min(round_size), 1e-9)
shape_contract = bool(long_aspect > round_aspect and long_aspect > 1.5)
passed = passed and shape_contract
report = {
    "status": "PASS" if passed else "FAIL",
    "shape_contract": {
        "status": "PASS" if shape_contract else "FAIL",
        "long_aspect_ratio": long_aspect,
        "round_aspect_ratio": round_aspect,
    },
    "assets": rows,
}
output = root / "inspection" / "isaac_asset_validation.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
app.close()
raise SystemExit(0 if passed else 1)
