# Policy 8 validation

Validated on 2026-10-30 without submitting a request or commanding the real
robot.

## Automated checks

- Python compilation: pass
- Policy 8 unit/contract tests: 9/9 pass
- Cell mock: pass, 81 actions, 0 rejected, 12 ms `act()` latency
- Grasp depth contract: M4 and M8 both remain at -334 mm
- Submission request file: absent

## Saved CSI image replay

The final fused scanner was replayed against five saved real CSI scenes:

| Saved scene | M4 | M8 |
|---|---:|---:|
| Policy 2 full scene | 3 | 3 |
| Policy 6 physical full-frame scene | 3 | 3 |
| Policy 6 random layout | 3 | 3 |
| Policy 6 bottom-edge layout | 3 | 3 |
| Policy 6 live frame with robot/shadows | 3 | 3 |

A separate real Policy 3 frame containing eight loose screws produced eight
targets, demonstrating that inventory is not fixed at six. A cleared frame
from the same physical run produced zero source targets even though the robot
and a filled destination bin remained visible.

These are offline perception checks, not proof of physical grasp success.
Policy 8 still requires an approved CSI rollout with the operator confirmations
listed in `POLICY8_RUNBOOK.md`.
