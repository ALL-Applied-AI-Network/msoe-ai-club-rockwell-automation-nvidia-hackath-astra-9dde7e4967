# Policy 11 runbook

## Feasibility and scope

Policy 11 composes the proven Policy 8 screw geometry, Policy 9 battery
geometry, and Policy 10 pill geometry behind one prompt parser and one guarded
motion controller. It supports variable counts and prompt-selected subsets.

Supported examples:

```text
Sort all screws into the red bin, all batteries into the green bin, and all pills into the yellow bin.
```

```text
Sort all M4 screws into the blue bin, all M8 screws into the green bin, all round pills into the yellow bin, and all long pills into the red bin.
```

A family-level rule activates both of its subtypes. A subtype rule activates
only the named subtype. Mixing a family rule and subtype rule for the same
family, omitting a destination, changing the instruction mid-trial, or using
an unsupported object/color fails before further motion.

The object count is not fixed. `DEMO_MAX_POLICY11_OBJECTS=48` is only a safety
ceiling. Cross-family objects closer than 18 mm fail closed.

## Bin and motion safety

- Old 50 x 50 x 55 mm bin: 27.5 mm projection plane, -270 mm release.
- New 90 x 90 x 60 mm bin: 30 mm projection plane, -265 mm release.
- Supported colors: red, green, yellow, blue.
- Every successful drop returns to `SAFE_HOME` before the next relocalization
  or final audit. This is the post-first-item-stall fix used by Policy 8 and
  now verified for Policies 8, 9, 10, and 11.
- All grasp depths remain -334 mm.

## No-robot preflight

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy11
./preflight.sh
```

No videos are generated.

## Services for a physical run

Start Cosmos first, then the six-class detector:

```bash
cd /home/yt-astraf0a5/workspace/physical_ai_submission_policy11
./start_cosmos_service.sh
./start_detector_service.sh
```

Then, only during an approved CSI slot with the operator present:

```bash
export DEMO_ALLOW_REAL_ROBOT=YES
export DEMO_UTOOL1_CONFIRMED=YES
export DEMO_TOUCH_OFF_CONFIRMED=YES
uv run run.py
```

## Submission note

The included submission YAML requests the first, family-level example. To ask
CSI for the second subtype-level task, update both submission YAML files before
manually running `uv run run.py --submit`. Runtime flexibility does not change
the physical scene that CSI approves and stages.

Policy 11 has more perception work than Policies 8-10 and therefore a longer
initial scan. It has passed parser, state-machine, and no-robot mock checks, but
a mixed six-class physical scene still requires a camera-only validation before
the first robot-enabled rollout.
