# Code architecture

## 1. Runtime boundary

Each policy is a self-contained Python project built around the CSI cell client.
The public entry point is `policy.py::Policy.act(obs)`. Policies request the
1920 x 1080 table-camera JPEG stream and return bounded Cartesian TCP actions
at 0.1-second intervals. Depth was available in downloaded CSI datasets but was
not requested or used by these final policy classes (`DEPTH = []`).

`run.py` and `cell_client.py` provide the CSI submission, approval, observation,
and action transport. `mock_cell.py` is an interface and bounds checker; its
synthetic test card does not validate perception or grasp success.

The runtime observation supplies:

- `obs["images"]["table"]`: source image for objects and bins;
- `obs["trial"]["instruction"]`: authoritative class-to-bin mapping;
- `obs["tcp_pose"]` and `obs["joints"]`: current robot/gripper state.

## 2. Shared control architecture

Policies 8–11 use the same guarded state-machine pattern:

```text
instruction
   ↓
scan → two-frame plan confirmation
   ↓
approach → descend → close → lift
   ↓
above bin → lower → open → retreat
   ↓
verify source object disappeared
   ↓
camera-clear SAFE_HOME → relocalize remaining targets
   └───────────────────────────────┐
                                   ↓
                      two clean final scans → complete
```

The controllers do not blindly replay fixed coordinates. They build an initial
inventory, verify target stability, refine the selected object immediately
before grasping, and reassociate remaining objects after each verified
placement. A missing, merged, unstable, unsupported, or ambiguous observation
causes a hold, retry, or fail-closed abort rather than unverified motion.

Important constants in the final code:

| Control item | Value |
| --- | ---: |
| Pick approach Z | -245 mm |
| Grasp Z | -334 mm |
| Transport/lift Z | -225 mm |
| Camera-clear home | `[300, 0, -90, -180, 0, 0, open]` |
| Two-frame XY stability | 2 mm |
| Two-frame yaw stability | 4 degrees |
| Bin-center stability | 2 mm |
| Trajectory linear increment | at most 5 mm |
| Trajectory angular increment | at most 3 degrees |
| Cell TCP Z envelope | -334 to 300 mm |

The three mandatory real-motion interlocks are checked on every physical
trial: `DEMO_ALLOW_REAL_ROBOT`, `DEMO_UTOOL1_CONFIRMED`, and
`DEMO_TOUCH_OFF_CONFIRMED` must all equal `YES`.

## 3. Calibration and coordinate conversion

`calibration/cameras.json` stores CSI camera intrinsics and camera-to-base
extrinsics. `calibration/geometry.py` provides:

- projection of base-frame points into camera pixels;
- conversion of table pixels back to base coordinates at a defined Z plane;
- depth-to-base helpers retained from the CSI scaffold;
- FANUC flange/camera transforms.

The final policies project observations onto the calibrated table surface at
`SURFACE_Z = -0.0051 m`. Pixel detections are therefore converted to robot XY
through calibration rather than an empirical image-to-motion scale.

## 4. Dynamic instructions and bins

The CSI instruction delivered at runtime is parsed before perception. Policies
8–10 require exactly two supported classes and two different destination
colors. They support red/green or the older red/yellow combination and reject
missing destinations, same-bin assignments, unsupported families, and
mid-trial prompt changes.

Bins are detected by color, then measured in calibrated metric space. The code
distinguishes:

| Profile | Physical body | Release Z |
| --- | --- | ---: |
| old | approximately 50 x 50 x 55 mm | -270 mm |
| new | approximately 90 x 90 x 60 mm | -265 mm |

Bin classification must be unambiguous. The selected profile is retained while
the robot temporarily occludes part of a bin.

## 5. Policy-specific perception

### Policy 8 — M4/M8 screws

Primary path: deterministic full-frame calibrated screw scan.

1. Restrict processing to the calibrated loose-object workspace.
2. Exclude detected destination-bin regions.
3. Build dark-object components using absolute and adaptive masks.
4. Require stable silhouette fits under two segmentation thresholds.
5. Measure head, shaft, total length, and aspect in millimeters.
6. Classify M4 versus M8 by disjoint calibrated geometry ranges.
7. Use the shaft midpoint—not the head—as the grasp point and align finger yaw
   to the long axis.

Cosmos3-Nano is only a fallback proposal source if the local scan cannot form a
safe inventory. Cosmos labels and axes are not trusted as final robot geometry.
Counts are variable; 24 objects is a safety ceiling, not an expected count.

### Policy 9 — AAA/AA batteries

Primary path: full-frame classical proposals plus a six-class YOLO11s crop
classifier. The detector service:

1. rejects a scan if the wrist/gripper blocks the source workspace;
2. masks the calibrated workspace and colored bins;
3. extracts category-agnostic loose-object boxes;
4. classifies each 96 x 96 crop;
5. restricts probabilities to the active AAA/AA class pair;
6. accepts a proposal at pairwise confidence 0.75 or requests fallback.

The policy then re-segments every candidate and measures physical length and
width. Geometry—not the learned label—is authoritative: AAA length must fall
below the 48.9 mm boundary and AA at or above 49.0 mm, with additional width
and aspect gates. Cosmos is retained for failed, empty, or low-margin scans.

### Policy 10 — round/long pills

Policy 10 combines two proposal sources:

- a local saturation/color component scan covering observed red/yellow pill
  colors; and
- the same task-gated six-class crop classifier.

All proposals pass calibrated two-threshold shape verification. Round pills
must fit the 8–15 mm near-circular range; long pills must fit the longer,
narrower 15.5–24 mm range. Color proposes a candidate but never selects its
class or destination. Policy 10 does not use Cosmos at runtime.

### Policy 11 — mixed-family experiment

Policy 11 parses either family-level or subtype-level instructions and routes
screw, battery, and pill perception through one controller. It supports red,
green, yellow, and blue destinations, uses a 48-object ceiling, and rejects
objects from different families closer than 18 mm.

The physical logs show useful partial behavior, but no trial reached the
required completion event. This folder is therefore labeled experimental.

## 6. Learned models

Two different model concepts were evaluated:

1. A synthetic full-frame six-class YOLO detector. It scored very highly on
   synthetic data but failed the real CSI domain gate through extensive
   background hallucinations. It was not used as the final motion authority.
2. A bin-free six-class crop classifier trained from balanced synthetic and
   real-domain crops. Policies 9 and 10 use this checkpoint as a proposal
   classifier, with task-pair restriction and deterministic metric checks.

This distinction matters: the final robot policies are hybrid systems, not
end-to-end learned control policies. Camera data and learned classification
help propose object identity; calibrated geometry and deterministic safety
logic determine motion.

## 7. Simulation and asset pipeline

`simulation/custom_finger/` contains Isaac Sim asset construction, clearance, and
physics-validation code for the custom fingers. The validated configuration
used a 16.4 mm nominal open gap, 4.4 mm closed inner gap, approximately
68.76 mm mount-to-tip length, and a 1 mm compliance allowance per side.

`simulation/asset_pipeline/` contains conversion, packaging, and validation
scripts for screws, pills, batteries, and old/new bin assets. CAD and generated
USD binaries are not included in the public source repository.

## 8. Logging and auditability

Every controller writes JSONL events such as `instruction`, `batch_plan`,
`motion`, `pickup_verified`, `relocalization`, `final_source_audit`, `abort`,
and `complete`. A trial is counted as complete only when the controller records
`event="complete"` after two clean source scans. See `LOG_EVIDENCE.md` and the
machine-readable summaries for the exact counting rules.
