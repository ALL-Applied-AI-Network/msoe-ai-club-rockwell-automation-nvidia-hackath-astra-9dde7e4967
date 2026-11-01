# Policy 9 validation

Validated on 2026-10-30 without submission or physical robot motion.

- Proposal model: Cosmos3-Nano
- Unit and task-contract tests: 12/12 pass
- Cell mock: 3 rollouts, 240 actions, 0 rejected
- Held-out Isaac test split: 150/150 scenes have exact in-workspace AA/AAA counts
- Two consecutive saved real CSI frames: Cosmos found both visible batteries;
  geometry classified AAA as 46.9 x 13.2 mm and AA as 52.4 x 19.2 mm; local
  confirmation shifted each centre only 0.46 mm
- Red/green bin replay: correct bins selected and yellow CAUTION tape rejected
- Verified red/green bin silhouettes are enforced as no-pick zones during
  Cosmos inventory and final residual audits
- Submission request file: absent
- Videos generated: none

The final battery verifier tolerates shadow-expanded silhouettes but still
requires a Cosmos battery proposal, calibrated battery-scale length and aspect,
an approved workspace location, two-frame plan confirmation, and two clean
residual scans. Confirmation and relocalization use calibrated local windows
instead of repeating Cosmos inference.

Detection was replayed on saved real CSI frames, but grasp success still
requires an approved, operator-supervised CSI rollout.
