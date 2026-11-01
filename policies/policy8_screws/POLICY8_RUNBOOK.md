# Policy 8 runbook

## Task contract

Use this instruction family; destinations may be swapped:

`Sort all M4 screws into the red bin and all M8 screws into the green bin.`

Policy 8 rejects prompts mentioning cubes, pills, or batteries. It reads M4
and M8 destinations from the CSI instruction and supports the new red/green
bins or old red/yellow bins. Ambiguous or same-bin mappings cannot move the
robot.

Bin size is automatic per detected bin. The old 50 x 50 x 55 mm profile uses
a -270 mm release; the new 90 x 90 x 60 mm profile uses -265 mm. An ambiguous
size measurement stops the policy.

The number of screws is not fixed. The default guarded ceiling is 24 total
screws; this is a safety limit, not an expected count. Change it only for an
approved placement with `DEMO_MAX_SCREWS`.

## Perception and recovery

1. A calibrated dual-mask full-field scan searches the complete loose-screw
   area without a fixed object count.
2. Cosmos receives a tight crop only if the local scan cannot form a safe
   inventory.
3. Every proposal must have a locally supported screw silhouette.
4. Calibrated head width, shaft width, and total length classify M4 versus M8.
5. The grasp point is placed at the shaft midpoint, never on the screw head.
6. Remaining planned screws are relocalized before grasping.
7. After the plan is exhausted, a fresh full-field scan finds any omitted or
   displaced screw.
8. A residual inventory must agree across two frames before motion.
9. Completion requires two consecutive clean scans.

## Safe local checks

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy8
uv sync
uv run python -m unittest discover -s tests -v
uv run run.py --mock --trials 3 --trial-s 8 \
  --instruction "Sort all M4 screws into the red bin and all M8 screws into the green bin."
```

The preflight performs the same checks without creating videos:

```bash
./preflight.sh
```

## Optional fallback service

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy8
./start_cosmos_service.sh
```

The local path normally completes without a Cosmos request. Keep the service
available during a physical run so the fallback is ready if shadows or an
unusual orientation defeat the local scan. Only one shared Cosmos service can
use port 18765.

## Submit manually

Policy 8 contains no inherited request file. When ready:

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy8
uv run run.py --submit
```

Do not copy `.request.json` from Policies 1-7.

## Run after CSI approval

With a CSI operator present and after verifying UTOOL1/touch-off:

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy8
export DEMO_ALLOW_REAL_ROBOT=YES
export DEMO_UTOOL1_CONFIRMED=YES
export DEMO_TOUCH_OFF_CONFIRMED=YES
uv run run.py
```

Policy 8 commands `-334 mm` for both screw sizes and releases at `-265 mm` for
the 60 mm-tall bins. If classification, bin
verification, relocalization, or residual scanning is ambiguous, it holds and
fails closed rather than guessing.
