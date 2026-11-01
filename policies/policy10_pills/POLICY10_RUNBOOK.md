# Policy 10 runbook

## Contract

- Round pills (smaller) -> red bin
- Long pills (larger/longer) -> green bin
- Runtime count is variable; 24 is only a configurable safety ceiling
- Exact instruction: `Sort all round pills into the red bin and all long pills into the green bin.`

The CSI instruction may swap destinations and may use the old red/yellow bin
set. The old 50 x 50 x 55 mm profile is detected at runtime and uses a -270 mm
release; the new 90 x 90 x 60 mm profile uses -265 mm. Ambiguous mappings or
bin dimensions stop the policy.

Policy 10 merges the new full-frame crop classifier with the proven local
pill component scan. It does not use Cosmos. Neither color nor the model
label selects the final class: calibrated projected length, width, aspect
ratio, two-threshold segmentation stability, workspace position, and
two-frame agreement decide round versus long.

Measured CAD: round 11.5 x 11.5 x 5.5 mm; long 18.7 x 8.5 x 7.2 mm. Both are
grasped at their body midpoint and the command depth remains -334 mm.

## Check and submit

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy10
./preflight.sh
uv run run.py --submit
```

Submission is a manual user action. Do not copy another policy's `.request.json`.

## Approved CSI run

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy10
./start_detector_service.sh
export DEMO_ALLOW_REAL_ROBOT=YES
export DEMO_UTOOL1_CONFIRMED=YES
export DEMO_TOUCH_OFF_CONFIRMED=YES
uv run run.py
```

Starting the classifier keeps the trained proposal path available. If it is
unavailable, Policy 10 can continue only when the proven local pill scan finds
proposals that pass all calibrated geometry checks. It requires two clean
residual scans before declaring completion and fails closed on camera
occlusion, ambiguous geometry, missing bins, no valid proposals, or unstable
detections.

Policies 9 and 10 share classifier port 18766 and are intentionally run one at
a time. `start_detector_service.sh` replaces the other policy's classifier.
After testing, release it with `./stop_detector_service.sh`.
