# Code manifest and provenance

## Snapshot policy

This repository contains the source used for the final October 31 CSI work,
plus selected simulation and training pipelines needed to explain how it was
built. Policy 8–10 were reconstructed from the complete local submission trees
and the final integration snapshots. Policy 11 matched the final DGX source.

## Policy folders

| Folder | Status | Primary perception | Final physical evidence |
| --- | --- | --- | --- |
| `policies/policy8_screws` | final as-run | calibrated dark-component screw scan; Cosmos fallback | 4/4 official IDs eventually complete |
| `policies/policy9_batteries` | final as-run | crop classifier + metric geometry; Cosmos fallback | 1/7 official IDs complete |
| `policies/policy10_pills` | final as-run | color/component proposals + crop classifier + metric geometry | 2/2 complete |
| `policies/policy11_multifamily_experimental` | experimental as-run | combined family routing | 0/3 complete |

Each policy folder includes its own CSI client scaffold, calibration, run
entry point, policy, service launchers, and task notes so that it can be
reviewed independently.

## Integrity hashes for principal as-run files

| File | SHA-256 |
| --- | --- |
| Policy 8 `policy.py` | `2fb82881cec6f72a74f4499ea350074285e8262676cdeb1ceb13884c1cfaaeec` |
| Policy 8 `cosmos_service.py` | `cb6a91d105518149cf10db4ed0ed87973ed0511e369da16493ea1097f7425d65` |
| Policy 9 `policy.py` | `93c6df1fa194e182ca91228083df68ee6128116a02ed43e838244ecab07aeef0` |
| Policy 9 `detector_service.py` | `fbf00fcc316cf29c75c50435bea569b687e7dd8f225e84b9831abefe3e65f437` |
| Policy 9 `start_detector_service.sh` | `5a9973d3a219c441921d7f25b88de9bc1f87ce5fbc21b5c3973384d5b66f80a4` |
| Policy 10 `policy.py` | `919fff2b1adbb27c608f5171b948fc418d629c13a8a2487fe83fa0d2e6598980` |
| Policy 10 `detector_service.py` | `fbf00fcc316cf29c75c50435bea569b687e7dd8f225e84b9831abefe3e65f437` |
| Policy 11 `policy.py` | `b8c5ca2b9378d55907fdf24e29711c75f90e610cf7e85a059dfe94dec9fd605e` |
| Policy 11 `detector_service.py` | `177c3f2a95631520d3c9a79f5cae9a82d900c387c2b0b2054b7d6bff936f55ac` |

The shared final `run.py` hash is
`136dd8d38a8633e53e4a1171e9472174c8ba71d5c24d647d1db1722e3ced7c44`.

`SHA256SUMS` records the hash of every publishable file in this repository
snapshot except the manifest itself and Git metadata.

The external crop-classifier weight file is not committed. The expected
checkpoint SHA-256 is:

`e88b2cb43dc00edf9d7872cd5aaac45fc6d2ef864d3d3879d2c36a754a924673`

## Training source

`training/synthetic_detector/` contains the synthetic full-frame YOLO pipeline
and the reports documenting why it was rejected for real full-frame use.

`training/real_crop_classifier/` contains dataset preparation, training,
evaluation, full-frame proposal, and camera-guard code. Dataset and model
artifacts are deliberately absent; the JSON reports and expected weight hash
make the experiment auditable without publishing CSI images.

## Simulation source

`simulation/custom_finger/` contains the non-destructive custom finger overlay,
clearance, physics, and sorting scripts with its final validation summary.

`simulation/asset_pipeline/` contains reusable asset packaging and dataset
generation/validation scripts for screws, pills, batteries, and bins. CAD and
generated USD assets remain outside the source repository because ownership
and distribution permission must be confirmed separately.

## Intentionally excluded

- `.request.json`, approval tokens, result keys, and private result URLs;
- raw CSI archives, RGB/depth frames, videos, and runtime JSONL files;
- detector/classifier weights and generated datasets;
- CAD source and generated USD binaries;
- virtual environments, caches, and editor files.

## Third-party boundary

The CSI client scaffold carried Apache-2.0 licensing and its license text is
retained under `THIRD_PARTY_LICENSES/`. This snapshot does not select a
repository-wide license for Team ASTRA's own code. Before making the repository
public, a human owner should choose that license and confirm that the team has
permission to redistribute every copied CSI scaffold file, calibration file,
and any future model weights or CAD assets. Do not add those binary assets
simply because they are available internally.

## GitHub publication checklist

1. Review team/contact fields in each `submission/submission.yaml`.
2. Confirm redistribution permission for CSI scaffold and calibration files,
   then choose a repository-wide license for team-created code.
3. Keep all `.request.json` files and private result links out of Git.
4. Decide whether model weights will be released separately; if yes, publish
   their license and SHA-256.
5. Create the GitHub repository, add it as `origin`, review `git status`, and
   commit only after a teammate completes the privacy/license review.
6. Never enable the real-robot flags outside an approved CSI slot with an
   operator present.

