# Policy 10 validation

Validated on 2026-10-30 without submission or physical robot motion.

- Trained checkpoint SHA-256: `1819742b774abc2a16cb47eb858a602dcaf8e0195676add27b2caab1ad18870b`
- Unit and task-contract tests: 8/8 pass
- Cell mock: pass, 80 actions, 0 rejected
- Held-out Isaac test split: 150/150 scenes have exact in-workspace round/long pill counts
- End-to-end trained perception, red/green bin detection, calibrated XY/yaw,
  and TCP safety gate: pass on a held-out pill scene
- Submission request file: absent
- Videos generated: none

Round/long classification requires non-overlapping calibrated length, width,
and aspect ranges in addition to the trained proposal, an approved workspace
location, two-frame plan confirmation, and two clean residual scans.

This is synthetic held-out and mock validation. No real CSI pill images were
available, so physical detection and grasp success still require an approved,
operator-supervised CSI rollout.
