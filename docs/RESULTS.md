# Results and limitations

## Scope of this report

This report separates four kinds of evidence that should not be mixed:

1. final physical CSI trials on October 31, 2026;
2. earlier physical development trials on October 30;
3. offline perception evaluation on collected images; and
4. Isaac Sim validation.

The strongest evidence is a physical trial containing an explicit
`event="complete"` after two clean source-table audits. A logged
`pickup_verified` event is useful evidence for an individual transfer, but it
does not by itself prove that the entire trial completed. A manually stopped
log is not converted into a failure or success after the fact.

## Final physical CSI results

### Summary

| Policy | Final run ID | Official trial IDs observed | Attempt log segments | Trial IDs eventually complete | Verified objects in completed attempts |
| --- | --- | ---: | ---: | ---: | ---: |
| 8: M4/M8 screws | `r20261031-0f36a5` | 4 | 7 | 4 | 17 screws |
| 9: AAA/AA batteries | `r20261031-36d31f` | 7 | 10 | 1 | 3 batteries |
| 10: round/long pills | `r20261031-72fefc` | 2 | 2 | 2 | 8 pills |
| 11: mixed-family experiment | `r20261031-ef6221` | 3 | 4 | 0 | 0 completed-trial objects |

“Eventually complete” means that the same official trial ID may have one or
more earlier aborted attempt logs before a later attempt recorded completion.
This is why Policy 8 has four completed trial IDs but seven attempt segments.

### Policy 8 — screws

| Trial | Attempts represented in logs | Completed inventory | Logged elapsed time of completed attempt | Outcome |
| --- | ---: | --- | ---: | --- |
| `t00` | 3 | 2 M4 | 45.9 s | two relocalization aborts, then complete |
| `t01` | 1 | 2 M4 + 1 M8 | 69.6 s | complete |
| `t02` | 2 | 3 M4 + 3 M8 | 146.7 s | one relocalization abort, then complete |
| `t03` | 1 | 3 M4 + 3 M8 | 147.4 s | complete |

The completed attempts contain 10 M4 and 7 M8 transfers, 17 total. This is the
best final physical evidence in the project. It also demonstrates variable
object count: the controller completed inventories of 2, 3, and 6 objects.

Qualification: this was not four first-attempt successes. Three relocalization
aborts occurred across `t00` and `t02` before the successful retries.

### Policy 9 — batteries

| Trial | What the log shows | Outcome |
| --- | --- | --- |
| `t00` | 1 AAA + 1 AA transferred; final audit had only one clean scan | not complete |
| `t01` | 2 AAA + 1 AA transferred; two clean final scans | complete, 190.0 s |
| `t02` | first attempt aborted after repeated AA confirmation failures; second transferred 3 AAA and stopped before final audit | not complete |
| `t03` | 1 AAA + 1 AA transferred; log stopped during verification | not complete |
| `t04` | no runtime JSONL found | no software result available |
| `t05` | 2 AAA + 1 AA transferred; final audit still saw one AA | not complete |
| `t06` | 2 AAA + 1 AA transferred; only one clean final scan | not complete |
| `t07` | three segments: stopped after one AAA, one centering abort, then stopped after two AAA | not complete |

Policy 9 could classify and move individual batteries, but end-to-end
reliability was poor. The recurring software-level problems were unstable AA
centering/confirmation and final-source audit uncertainty. It is inaccurate to
claim that Policy 9 reliably solved battery sorting.

### Policy 10 — pills

| Trial | Completed inventory | Logged elapsed time | Outcome |
| --- | --- | ---: | --- |
| `t00` | 2 round + 2 long | 99.0 s | complete |
| `t01` | 2 round + 2 long | 95.4 s | complete |

Policy 10 completed both final trials with eight verified transfers and no
failure events in those two attempt logs. This is the cleanest final run set.
Earlier October 30 logs also contain three complete four-pill trials after an
initial confirmation abort, but those are development evidence rather than the
final run set.

### Policy 11 — mixed-family experiment

Policy 11 did not complete a physical trial. `t00` failed perception due to
cross-family overlap and a missing battery metric proposal. `t01` transferred
one M4, one AAA, and one round pill but did not obtain two clean final audits.
`t02` first failed on cross-family proximity; a retry transferred the same
three-family set, but the final audit found a residual M8 screw. The code is
retained because its dynamic prompt and multi-family routing are useful
engineering work, but it must be labeled experimental.

## Offline perception results

### Synthetic full-frame detector

A YOLO11s detector trained on the six synthetic classes reported:

| Synthetic test metric | Value |
| --- | ---: |
| Precision | 0.9992 |
| Recall | 0.9990 |
| mAP50 | 0.9950 |
| mAP50–95 | 0.9235 |
| Training time | 2,901.8 s |

Those scores did not transfer to the real CSI background. On the real-image
domain gate, the detector hallucinated large numbers of pill and battery
objects from background/shadow structure. It was therefore rejected as the
final full-frame motion authority. This is an important negative result: high
synthetic test accuracy alone was not sufficient.

### Real/synthetic crop classifier

The replacement was a bin-free six-class crop classifier. The training set
contained 400 synthetic and 400 sampled real crops per class (4,800 total).
Validation and test crops were kept out of the sampled training set and were
split by recorded session.

| Evaluation | Result |
| --- | ---: |
| Held-out real crop top-1 | 19/20 (95%) |
| Only error | one M8 classified as M4 at confidence 0.510 |
| Held-out clear full-frame target proposals | 20/20 |
| Extra proposals | 0 |
| Raw six-class decisions | 19/20 |
| Correct after task-pair restriction | 20/20 |
| Accepted at pairwise confidence 0.75 | 18/20 |
| Accepted false positives | 0 |
| Clear frames incorrectly blocked by camera guard | 0/6 |
| Robot-occluded frames correctly blocked | 7/7 |
| Training time | 110.1 s |

Checkpoint SHA-256:

`e88b2cb43dc00edf9d7872cd5aaac45fc6d2ef864d3d3879d2c36a754a924673`

The sample is small and includes adjacent frame pairs. These numbers justify
using the classifier as a proposal source with deterministic geometry checks;
they do not justify a broad claim of 95% real-world system accuracy.

## Simulation results

The custom finger integration completed a six-cube stack in Isaac Sim:

| Metric | Result |
| --- | ---: |
| Pickups / placements | 6 / 6 |
| Maximum lateral placement error | 0.383 mm |
| Maximum yaw error modulo 90° | 1.525° |
| Maximum tilt | 0.596° |
| Maximum simulated compression | 0.147 mm left, 0.151 mm right |
| Nominal open / closed inner gap | 16.4 / 4.4 mm |
| Mount-to-tip length | 68.764 mm |

This validates the simulated geometry and control path, not physical grasp
performance. Material mass values were assumptions and mount frames were
inferred from symmetric CAD geometry.

An additional four-way simulated sort placed eight of eight objects into the
correct bins: two yellow cubes, two red cubes, two M4 screws, and two M8
screws. That run used Cosmos for scene-level detection and deterministic
motion/geometry for execution.

## What can and cannot be claimed

Safe, evidence-backed statements:

- Policy 8 eventually completed all four observed final screw trial IDs and
  logged 17 verified screw transfers in the completed attempts.
- Policy 10 completed both final pill trials and logged eight verified pill
  transfers.
- Policy 9 completed one final battery trial with three verified transfers;
  other trials show useful partial pickups but weak end-to-end reliability.
- Policy 11 demonstrated partial multi-family handling but no complete trial.
- The final software was a hybrid perception and deterministic control system,
  not an end-to-end learned robot policy.

Statements that the current evidence does not support:

- “All policies worked flawlessly.”
- “Policy 9 solved battery sorting reliably.”
- “Policy 11 completed mixed-object sorting.”
- “The learned detector controlled the robot end to end.”
- “The custom fingers improved success by X% over stock fingers.” No
  controlled stock-finger baseline exists in the inspected code/log evidence.
- “No deformation or damage occurred.” That requires visual or physical
  inspection evidence from the hardware team, not software logs.

## Principal limitations

- Only the table RGB stream was used; available CSI depth was not requested by
  the final policies.
- The crop-classifier real test set was very small.
- Shadows, specular battery wrappers, object overlap, and wrist occlusion
  remained significant perception risks.
- Completion was based on visual source disappearance and clean rescans; there
  was no force sensor or in-hand object sensor.
- The physical evidence is not a controlled comparison against the original
  gripper fingers.
- Logged elapsed time includes perception-service latency and operator/run
  conditions; it is not a pure industrial cycle-time benchmark.

