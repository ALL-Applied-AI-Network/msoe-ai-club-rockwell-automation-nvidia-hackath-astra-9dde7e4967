# Policy 9 runbook

## Contract

- AAA batteries (smaller) -> red bin
- AA batteries (larger) -> green bin
- Runtime count is variable; 24 is only a configurable safety ceiling
- Exact instruction: `Sort all AAA batteries into the red bin and all AA batteries into the green bin.`

The CSI instruction may swap destinations and may use the old red/yellow bin
set. The old 50 x 50 x 55 mm profile is detected at runtime and uses a -270 mm
release; the new 90 x 90 x 60 mm profile uses -265 mm. Ambiguous mappings or
bin dimensions stop the policy.

The local full-frame proposal scanner and six-class crop classifier run first.
Low-margin proposals are still required to pass projected length, width,
aspect ratio, two-threshold segmentation stability, workspace position, and
two-frame agreement. Calibrated geometry—not the classifier label, color, or
image position—decides AA versus AAA. Cosmos3-Nano is requested only when the
local classifier is ambiguous, unavailable, or produces no verified object.
Confirmation and per-pick relocalization use local calibrated windows.

Measured CAD: AAA 44.0 x 10.2 mm; AA 49.0 x 14.0 mm. Both are grasped at the
body midpoint and the command depth remains -334 mm.

## Check and submit

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy9
./preflight.sh
uv run run.py --submit
```

Submission is a manual user action. Do not copy another policy's `.request.json`.

## Approved CSI run

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy9
# Start in this order: Cosmos removes any older detector container, then the
# small classifier is loaded beside it on port 18766.
./start_cosmos_service.sh
./start_detector_service.sh
export DEMO_ALLOW_REAL_ROBOT=YES
export DEMO_UTOOL1_CONFIRMED=YES
export DEMO_TOUCH_OFF_CONFIRMED=YES
uv run run.py
```

Both health checks must pass before the run. Policy 9 requires two clean
residual scans before declaring completion and fails closed on camera
occlusion, ambiguous geometry, missing bins, or unstable detections.

The order above is mandatory because `start_cosmos_service.sh` removes an older
detector container before loading Cosmos. After testing, stop both services
with `./stop_detector_service.sh` and `./stop_cosmos_service.sh`.
