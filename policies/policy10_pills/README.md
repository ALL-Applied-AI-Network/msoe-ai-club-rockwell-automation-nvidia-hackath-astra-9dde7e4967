# Astra Policy 10

Variable-count pill-size sorting for the CSI FANUC cell.

- round (smaller) pills -> red bin
- long (larger) pills -> green bin
- trained six-class YOLO11s plus local chromatic-component proposals
- proposal labels are non-authoritative; calibrated pill geometry decides size class
- two-frame plan confirmation and two clean residual scans
- custom fingers, -334 mm command depth

See `POLICY10_RUNBOOK.md`. This folder intentionally has no `.request.json`;
submission remains a manual user action.
