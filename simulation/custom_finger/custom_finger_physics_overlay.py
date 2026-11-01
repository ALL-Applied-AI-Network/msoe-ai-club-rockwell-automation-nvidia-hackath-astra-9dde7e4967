"""Non-destructive custom finger CAD overlay with rigid core and compliant silicone pads."""

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


COMPRESSION_LIMIT_M = 0.001
OPEN_INNER_GAP_M = 0.0164
CLOSED_INNER_GAP_M = 0.0044
MOUNT_OFFSET_M = 0.0052
MOUNT_TO_TIP_M = 0.0687643899917603
SILICONE_STIFFNESS_N_M = 5000.0
SILICONE_DAMPING_N_S_M = 25.0
SILICONE_MAX_FORCE_N = 20.0

# Candidate values until measured masses arrive. The core uses the owner's PLA
# statement even though the supplied CAD role names contain TPU.
CORE_MATERIAL_ASSUMPTION = "PLA (owner statement; CAD names say TPU)"
RIGID_JAW_MASS_KG = 0.048
SILICONE_MASS_KG = 0.012

CORE_PHYSICS_PATH = "/World/Physics/FingerV2Core"
SILICONE_PHYSICS_PATH = "/World/Physics/FingerV2Silicone"
CORE_VISUAL_PATH = "/World/Looks/FingerV2Core"
SILICONE_VISUAL_PATH = "/World/Looks/FingerV2Silicone"


def _display_material(stage, path, color, roughness):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def _make_materials(stage):
    core = UsdShade.Material.Define(stage, CORE_PHYSICS_PATH)
    core_api = UsdPhysics.MaterialAPI.Apply(core.GetPrim())
    core_api.CreateStaticFrictionAttr().Set(0.55)
    core_api.CreateDynamicFrictionAttr().Set(0.45)
    core_api.CreateRestitutionAttr().Set(0.02)

    silicone = UsdShade.Material.Define(stage, SILICONE_PHYSICS_PATH)
    silicone_api = UsdPhysics.MaterialAPI.Apply(silicone.GetPrim())
    silicone_api.CreateStaticFrictionAttr().Set(1.5)
    silicone_api.CreateDynamicFrictionAttr().Set(1.3)
    silicone_api.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(
        silicone.GetPrim()
    ).CreateFrictionCombineModeAttr().Set("max")

    core_visual = _display_material(
        stage, CORE_VISUAL_PATH, (0.18, 0.20, 0.24), 0.58
    )
    silicone_visual = _display_material(
        stage, SILICONE_VISUAL_PATH, (0.12, 0.60, 0.78), 0.74
    )
    return core, silicone, core_visual, silicone_visual


def _bind_physics_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _bind_visual_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )


def _expand_instances(root):
    """Turn referenced CAD instances into editable composed prims."""
    while True:
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
        if not instances:
            return
        for prim in instances:
            prim.SetInstanceable(False)


def _mesh_prims(root):
    return [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]


def _prepare_asset(root, physics_material, visual_material):
    _expand_instances(root)
    visual_root = root.GetChild("Visual")
    collision_root = root.GetChild("Collision")
    if not visual_root.IsValid() or not collision_root.IsValid():
        raise RuntimeError(
            f"Expected Visual and Collision branches under {root.GetPath()}"
        )

    visual_meshes = _mesh_prims(visual_root)
    collision_meshes = _mesh_prims(collision_root)
    if not visual_meshes or not collision_meshes:
        raise RuntimeError(
            f"Missing visual/collision meshes under {root.GetPath()}: "
            f"{len(visual_meshes)}/{len(collision_meshes)}"
        )

    for prim in visual_meshes:
        _bind_visual_material(prim, visual_material)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)

    if collision_root.IsA(UsdGeom.Imageable):
        UsdGeom.Imageable(collision_root).MakeInvisible()
    for prim in collision_meshes:
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        collision.CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "convexDecomposition"
        )
        contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        contact.CreateContactOffsetAttr().Set(0.0002)
        contact.CreateRestOffsetAttr().Set(0.0)
        _bind_physics_material(prim, physics_material)

    return {
        "visual_meshes": [prim.GetPath().pathString for prim in visual_meshes],
        "collision_meshes": [
            prim.GetPath().pathString for prim in collision_meshes
        ],
    }


def _disable_old_finger(stage, jaw_path):
    jaw = stage.GetPrimAtPath(jaw_path)
    keep = {"Carriage", "SMCJaw"}
    hidden = []
    disabled_colliders = []
    for child in jaw.GetChildren():
        if child.GetName() in keep:
            continue
        for prim in Usd.PrimRange(child):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(prim.GetPath().pathString)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                disabled_colliders.append(prim.GetPath().pathString)

    still_visible = []
    for child in jaw.GetChildren():
        if child.GetName() in keep:
            continue
        for prim in Usd.PrimRange(child):
            if (
                prim.IsA(UsdGeom.Imageable)
                and UsdGeom.Imageable(prim).ComputeVisibility()
                != UsdGeom.Tokens.invisible
            ):
                still_visible.append(prim.GetPath().pathString)
    if still_visible:
        raise RuntimeError(
            f"Legacy finger visuals remain visible under {jaw_path}: {still_visible}"
        )
    return hidden, disabled_colliders


def _jaw_local_translation(jaw_prim):
    matrix = UsdGeom.Xformable(jaw_prim).GetLocalTransformation()
    if isinstance(matrix, tuple):
        matrix = matrix[0]
    return Gf.Vec3d(matrix.ExtractTranslation())


def apply(
    stage,
    left_rigid_asset,
    right_rigid_asset,
    left_silicone_asset,
    right_silicone_asset,
    robot_prim="/World/ER4iA",
    flange_path="/World/ER4iA/J6_link",
):
    """Replace legacy finger visuals/colliders for this stage only."""
    core, silicone, core_visual, silicone_visual = _make_materials(stage)
    result = {
        "finger_id": "finger_v02_candidate",
        "core_material_assumption": CORE_MATERIAL_ASSUMPTION,
        "open_inner_gap_m": OPEN_INNER_GAP_M,
        "closed_inner_gap_m": CLOSED_INNER_GAP_M,
        "mount_offset_per_side_m": MOUNT_OFFSET_M,
        "mount_to_tip_m": MOUNT_TO_TIP_M,
        "compression_limit_per_side_m": COMPRESSION_LIMIT_M,
        "stiffness_n_m": SILICONE_STIFFNESS_N_M,
        "damping_n_s_m": SILICONE_DAMPING_N_S_M,
        "candidate_masses_kg": {
            "jaw_plus_core": RIGID_JAW_MASS_KG,
            "silicone_each": SILICONE_MASS_KG,
        },
        "sides": {},
    }

    for side, sign, rigid_asset, silicone_asset in (
        ("L", +1.0, left_rigid_asset, left_silicone_asset),
        ("R", -1.0, right_rigid_asset, right_silicone_asset),
    ):
        jaw_path = f"{robot_prim}/Jaw{side}"
        jaw = stage.GetPrimAtPath(jaw_path)
        if not jaw.IsValid():
            raise RuntimeError(f"Missing jaw prim {jaw_path}")
        hidden, old_colliders = _disable_old_finger(stage, jaw_path)

        rigid = UsdGeom.Xform.Define(stage, jaw_path + "/FingerV2Rigid")
        rigid.AddTranslateOp().Set(Gf.Vec3d(0.0, sign * MOUNT_OFFSET_M, 0.0))
        rigid.GetPrim().GetReferences().AddReference(rigid_asset)
        rigid_geometry = _prepare_asset(rigid.GetPrim(), core, core_visual)
        UsdPhysics.MassAPI.Apply(jaw).CreateMassAttr().Set(RIGID_JAW_MASS_KG)

        silicone_path = f"{robot_prim}/FingerV2Silicone{side}"
        silicone_root = UsdGeom.Xform.Define(stage, silicone_path)
        jaw_translation = _jaw_local_translation(jaw)
        silicone_root.AddTranslateOp().Set(
            jaw_translation + Gf.Vec3d(0.0, sign * MOUNT_OFFSET_M, 0.0)
        )
        silicone_root.GetPrim().GetReferences().AddReference(silicone_asset)
        silicone_geometry = _prepare_asset(
            silicone_root.GetPrim(), silicone, silicone_visual
        )
        body = UsdPhysics.RigidBodyAPI.Apply(silicone_root.GetPrim())
        body.CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(silicone_root.GetPrim()).CreateMassAttr().Set(
            SILICONE_MASS_KG
        )
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(silicone_root.GetPrim())
        physx_body.CreateEnableCCDAttr().Set(True)
        physx_body.CreateLinearDampingAttr().Set(0.05)
        physx_body.CreateAngularDampingAttr().Set(0.10)

        joint_name = f"silicone_{side.lower()}_compression"
        joint = UsdPhysics.PrismaticJoint.Define(
            stage, f"{flange_path}/Gripper/{joint_name}"
        )
        joint.CreateBody0Rel().SetTargets([jaw_path])
        joint.CreateBody1Rel().SetTargets([silicone_path])
        joint.CreateAxisAttr().Set("Y")
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(0.0, sign * MOUNT_OFFSET_M, 0.0)
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateCollisionEnabledAttr().Set(False)
        if sign > 0:
            joint.CreateLowerLimitAttr().Set(0.0)
            joint.CreateUpperLimitAttr().Set(COMPRESSION_LIMIT_M)
        else:
            joint.CreateLowerLimitAttr().Set(-COMPRESSION_LIMIT_M)
            joint.CreateUpperLimitAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(SILICONE_STIFFNESS_N_M)
        drive.CreateDampingAttr().Set(SILICONE_DAMPING_N_S_M)
        drive.CreateMaxForceAttr().Set(SILICONE_MAX_FORCE_N)
        drive.CreateTargetPositionAttr().Set(0.0)

        result["sides"][side] = {
            "jaw_path": jaw_path,
            "rigid_path": rigid.GetPath().pathString,
            "silicone_path": silicone_path,
            "joint": joint_name,
            "rigid_geometry": rigid_geometry,
            "silicone_geometry": silicone_geometry,
            "hidden_legacy_visuals": hidden,
            "disabled_legacy_colliders": old_colliders,
        }

    print(
        "[custom-finger-physics] opening=16.4 mm; closed=4.4 mm;",
        "silicone travel=1.0 mm/side;",
        f"spring={SILICONE_STIFFNESS_N_M:.0f} N/m;",
        f"damping={SILICONE_DAMPING_N_S_M:.1f} N*s/m",
        flush=True,
    )
    return result
