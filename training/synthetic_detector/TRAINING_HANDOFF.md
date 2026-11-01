# Six-class synthetic detector — training handoff

Completed: 2026-10-30

## Scope

- Classes: `m4_screw`, `m8_screw`, `round_pill`, `long_pill`, `aa_battery`, `aaa_battery`
- Dataset: 1,500 Isaac Sim scenes (1,200 train / 150 validation / 150 test)
- Model: YOLO11s initialized from `yolo11s.pt`
- Training: 30 epochs, 1280 px input, batch 8, FP32 (`amp=False`)
- Best checkpoint selected using the validation split; the test split remained held out until final evaluation.

## Final metrics

| Split | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Validation | 0.99895 | 0.99874 | 0.99500 | 0.92556 |
| Test | 0.99918 | 0.99896 | 0.99500 | 0.92350 |

Test mAP@50–95 by class:

| Class | mAP@50–95 |
|---|---:|
| m4_screw | 0.886 |
| m8_screw | 0.949 |
| round_pill | 0.858 |
| long_pill | 0.895 |
| aa_battery | 0.986 |
| aaa_battery | 0.967 |

## Artifacts

- Local checkpoint: `artifacts/best.pt`
- Local report: `artifacts/training_report.json`
- Local learning history: `artifacts/results.csv`
- Local test diagnostics: `artifacts/final_test/`
- Remote run: `/home/yt-astraf0a5/workspace/training_six_class_20261030/baseline_yolo11s_30e_1280`
- Remote checkpoint: `/home/yt-astraf0a5/workspace/training_six_class_20261030/baseline_yolo11s_30e_1280/train/weights/best.pt`

SHA-256 for `artifacts/best.pt`:

`1819742b774abc2a16cb47eb858a602dcaf8e0195676add27b2caab1ad18870b`

## Interpretation

The synthetic held-out results are strong and the labeled-object confusion matrix is diagonal. This proves the pipeline learned the six simulated classes, but it does not yet prove real-camera robustness. The next gate is inference on CSI camera frames, followed by synthetic-to-real tuning or real-image fine-tuning if recall or localization drops.
