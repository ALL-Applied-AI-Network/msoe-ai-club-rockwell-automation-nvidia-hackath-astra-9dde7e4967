# Astra Policy 8

Dedicated variable-count M4/M8 screw sorting policy.

- M4 screws -> red bin
- M8 screws -> green bin
- Number and placement of screws are discovered at runtime
- No pills, batteries, or cubes are valid Policy 8 targets
- Cosmos grounds the scene; calibrated local geometry verifies every screw,
  classifies its size, and places the grasp on the shaft
- A full calibrated source-field scan supplements Cosmos so an initially
  omitted screw can still be recovered
- Completion requires two consecutive clean source scans
- Grasp depth remains `-334 mm`
- Real motion remains locked behind the three operator confirmation flags

Policy 8 is isolated at:

`/home/yt-astraf0a5/workspace/physical_ai_submission_policy8`

It intentionally has no copied `.request.json`; submission is a separate user
action.
