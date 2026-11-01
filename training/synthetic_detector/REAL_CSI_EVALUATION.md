# Real CSI evaluation — six-class detector

Date: 2026-10-30

## Decision

**FAIL — do not integrate this checkpoint into the picking policy yet.**

The synthetic test result did not transfer to the real CSI camera domain. The detector frequently assigns pill and battery classes to cubes, bins, robot hardware, cables, shadows, and table features. Some real screws are localized, but M4/M8 classification is also not reliable enough for sorting.

## Evaluation set

- 194 unique real CSI images
- 162 stratified frames from 22 physical-run sequences
- 32 unique Cosmos request images
- All mock frames excluded
- No evaluation image was used during training
- These screw/cube runs contain no pills or batteries; therefore every pill or battery detection is a confirmed false positive

## Results

Inference used the trained `best.pt` checkpoint at 1280 px on the DGX GPU. Processing 194 images took 6.526 seconds.

| Confidence threshold | Confirmed false pill/battery detections | False detections/image | Images with ≥1 false detection | Screw detections* |
|---:|---:|---:|---:|---:|
| 0.25 | 1,748 | 9.01 | 194/194 (100.0%) | 1,263 |
| 0.50 | 1,250 | 6.44 | 194/194 (100.0%) | 1,006 |
| 0.75 | 776 | 4.00 | 192/194 (99.0%) | 857 |
| 0.80 | 591 | 3.05 | 190/194 (97.9%) | 812 |
| 0.90 | 93 | 0.48 | 68/194 (35.1%) | 523 |

\*The screw count is not precision or recall. The frames are unlabeled and temporally related, and the detector sometimes confuses M4 with M8 or fires on background hardware.

At confidence 0.50, the 1,250 confirmed false detections were:

- `aa_battery`: 458
- `aaa_battery`: 122
- `long_pill`: 454
- `round_pill`: 216

The failure occurs in both sources:

- Physical-run frames: 1,025 confirmed false detections across 162/162 images
- Cosmos request images: 225 confirmed false detections across 32/32 images

## Evidence

- `review/false_positive_cosmos_crop.jpg`: screws and cubes labeled as AA/AAA batteries
- `review/screws_detected_but_wrong_classes.jpg`: real screws localized, but all treated as M4 while unrelated structures become pills/batteries
- `review/empty_table_false_positives.jpg`: robot and background features labeled as pills/batteries after objects were removed
- `inference_report.json`: every predicted box and confidence
- `detections.csv`: flat detection table
- `manifest.json`: immutable source path and SHA-256 for every evaluated image

## Required correction

1. Keep this checkpoint out of the robot policy.
2. Split future real data by complete physical run, never by adjacent frames, to prevent temporal leakage.
3. Annotate real screw positives and real hard negatives (cubes, bins, cables, robot, shadows, table seams).
4. Fine-tune with a balanced mixture of synthetic data and real hard examples.
5. Use task-specific class allowlists: screw runs should expose only M4/M8; pill and battery classes should remain disabled until real examples exist.
6. Capture real pill and battery frames before claiming real-world performance for those four classes.
7. Re-run this gate on entirely held-out runs. Integration requires no high-confidence background hallucinations and reliable M4/M8 separation.
