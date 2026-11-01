# Bin-free six-class classifier results

Completed: 2026-10-31

## Dataset

- Classes: M4 screw, M8 screw, round pill, long pill, AA battery, AAA battery
- Training: 4,800 balanced crops (400 synthetic + 400 real-domain samples per class)
- Validation: 24 real CSI crops from held-out runtime sessions
- Test: 20 real CSI crops from different held-out runtime sessions
- No bin crops were used in training, validation, or testing
- Complete runtime sessions were kept within a single split

## Result

- Real validation top-1 accuracy: 24/24 (100%)
- Real test top-1 accuracy: 19/20 (95%)
- Frame-00000-only test accuracy: 9/10 (90%)
- Best epoch: 6 (early stopping at epoch 16)
- Training time: 110.1 seconds

The one crop-only error was an M8 screw classified as M4 with only 0.510
confidence. All battery and pill test crops were classified correctly. The
full-frame evaluation below is the more representative result for policy use.

## Full-frame result

The selected full-frame proposal parameters were fitted on train plus
validation only. On six held-out clear scan frames:

- Object proposals: 20/20 targets found (100% recall)
- Extra proposals: 0
- Raw six-class classification: 19/20 (95%)
- Classification after restricting to the known task pair: 20/20 (100%)
- Accepted by the conservative 0.75 pairwise-confidence gate: 18/20 (90%)
- Accepted false positives: 0

The two rejected correct detections were one AA and one AAA battery. Rejection
is deliberately fail-safe: the policy should take another clear scan instead
of attempting a low-margin grasp.

A camera-clear safety guard was also checked on six clear scan frames and seven
post-sort frames. It left all six clear frames enabled and blocked all seven
frames where the robot wrist/gripper occupied the loose-object workspace.

These results cover only the collected CSI layouts and adjacent frame pairs.
They are a strong integration check, not a deployment-level generalization
claim. A future genuinely new CSI run must remain untouched until final
evaluation.

## Artifacts

- Checkpoint: `artifacts/run/weights/best.pt`
- Detailed predictions: `classifier_predictions.csv`
- Structured evaluation: `classifier_evaluation.json`
- Full-frame evaluation: `full_frame_evaluation/guarded_full_frame_report.json`
- Occluded-frame audit: `full_frame_evaluation/empty_frame_audit.json`
- Full-frame still-image annotations: `full_frame_evaluation/guarded_test_annotated/`
- Test confusion matrix: `artifacts/real_test/confusion_matrix.png`

Checkpoint SHA-256:

`e88b2cb43dc00edf9d7872cd5aaac45fc6d2ef864d3d3879d2c36a754a924673`
