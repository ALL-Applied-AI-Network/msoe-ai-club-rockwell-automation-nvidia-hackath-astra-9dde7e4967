# Team ASTRA Physical AI Sorting System

Code and technical evidence for Team ASTRA's FANUC ER-4iA sorting project at
the 2026 Physical AI Hackathon for the Factory Floor.

The system used one fixed pair of custom fingers for three task-specific
sorting policies:

- Policy 8: variable-count M4/M8 screw sorting;
- Policy 9: variable-count AAA/AA battery sorting;
- Policy 10: variable-count round/long pill sorting.

Policy 11 combined all three object families behind a dynamic prompt parser,
but remained experimental and did not complete a physical trial. It is kept in
the repository for engineering traceability, not presented as a successful
final policy.

## What is in this repository

```text
policies/
  policy8_screws/                 final CSI screw policy
  policy9_batteries/              final CSI battery policy
  policy10_pills/                 final CSI pill policy
  policy11_multifamily_experimental/
training/
  synthetic_detector/             synthetic six-class detector pipeline
  real_crop_classifier/           real/synthetic crop-classifier pipeline
simulation/
  custom_finger/                   Isaac Sim finger integration and validation
  asset_pipeline/                 object/bin conversion and dataset scripts
docs/
  ARCHITECTURE.md
  RESULTS.md
  LOG_EVIDENCE.md
  CODE_MANIFEST.md
  runtime_log_summary.{json,csv}
tools/
  summarize_runtime_logs.py
SHA256SUMS                         integrity manifest for this snapshot
```

The Policy 8–10 source trees are the versions used for the final CSI work on
October 31, 2026.

## System in one paragraph

The CSI client supplies a calibrated table-camera image, robot state, and a
natural-language sorting instruction. A fail-closed parser converts the
instruction to class-to-bin assignments. Each task uses a different hybrid
perception path, but learned outputs are proposals rather than motion
authority: calibrated image geometry verifies object class, center, yaw,
separation, workspace bounds, and bin geometry. A deterministic state machine
then confirms the scene across frames, executes bounded Cartesian trajectories,
verifies source-object disappearance after each placement, returns to a
camera-clear home pose, relocalizes the remaining objects, and requires two
clean final scans before logging completion.

## Final physical evidence

The final October 31 logs support these claims:

| Policy | Official trial IDs observed | Trials with a logged `complete` event | Verified objects in those completed trials |
| --- | ---: | ---: | ---: |
| Policy 8, screws | 4 | 4 | 17 screws |
| Policy 9, batteries | 7 | 1 | 3 batteries |
| Policy 10, pills | 2 | 2 | 8 pills |
| Policy 11, mixed experimental | 3 | 0 | 0 completed-trial objects |

These are software-log outcomes, not a controlled comparison against stock
metal fingers. Some trials required retries, and partial trials are retained in
the evidence files. See [docs/RESULTS.md](docs/RESULTS.md) for the complete,
qualified interpretation.

## Models and data

Model weights, raw CSI recordings, CAD owned by project contributors, and
generated datasets are not committed. The final crop-classifier checkpoint
used by Policies 9 and 10 had SHA-256:

`e88b2cb43dc00edf9d7872cd5aaac45fc6d2ef864d3d3879d2c36a754a924673`

Its held-out crop result was 19/20 top-1 correct. The full-frame guarded replay
found 20/20 target proposals, no extra proposals, and 20/20 correct decisions
after task-pair restriction; 18/20 passed the conservative 0.75 confidence
gate. These are small collected-scene evaluations, not production-level
generalization results.

## Safety boundary

This code is a research prototype. Real motion is locked unless all three
runtime confirmations equal `YES`:

- `DEMO_ALLOW_REAL_ROBOT`
- `DEMO_UTOOL1_CONFIRMED`
- `DEMO_TOUCH_OFF_CONFIRMED`

The policies also enforce calibrated workspace and TCP bounds, including the
CSI lower Z limit of `-334 mm`. Do not use this repository to command hardware
without the cell owner, approved tooling, current calibration, safety review,
and an operator present.

## Local interface check

Each policy is an independent `uv` project. The CSI-provided mock validates API
shape and motion limits, not physical perception or grasping. Use the matching
task instruction instead of `run.py`'s generic cube default. Example:

```bash
cd policies/policy10_pills
uv run run.py --mock --instruction \
  "Sort all round pills into the red bin and all long pills into the green bin."
```

No private `.request.json`, run token, result key, or raw robot dataset is
included.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Results and limitations](docs/RESULTS.md)
- [Log evidence and counting rules](docs/LOG_EVIDENCE.md)
- [Code manifest and provenance](docs/CODE_MANIFEST.md)
- [Repository validation](docs/REPOSITORY_VALIDATION.md)
- [Full-file integrity manifest](SHA256SUMS)

## License

The CSI client scaffold included an Apache-2.0 license, retained under
[`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES/CSI_CLIENT_APACHE-2.0.txt).
Team ASTRA has not selected a repository-wide license in this snapshot. The
repository owner must confirm third-party redistribution rights and choose a
license for team-created code before public release.
