# Astra six-class real-object classifier

This experiment excludes bins and classifies only verified object crops:

- M4 screw
- M8 screw
- round pill
- long pill
- AA battery
- AAA battery

Real crop labels come from complete Policy 8/9/10 runtime sessions. Screw and
pill centres use the calibrated metric targets recorded by the policies.
Battery centres are independently recovered from their dark-body geometry and
checked against the successful Policy 9 metric targets.

Complete runtime sessions are assigned to only one of train, validation, or
test. This prevents adjacent-frame leakage. Synthetic object crops are used
only for training; validation and test contain real CSI crops only. Bins are
not present in any classifier sample.
