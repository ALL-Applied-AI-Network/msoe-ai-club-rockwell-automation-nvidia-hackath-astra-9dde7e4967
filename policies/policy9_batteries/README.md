# Astra Policy 9

Variable-count AA/AAA battery sorting for the CSI FANUC cell.

- AAA (smaller) -> red bin
- AA (larger) -> green bin
- Cosmos3-Nano full-frame battery proposals
- Cosmos labels are non-authoritative; calibrated geometry decides AA versus AAA
- local calibrated target windows handle confirmation and relocalization without repeated Cosmos calls
- two-frame plan confirmation and two clean residual scans
- custom fingers, -334 mm command depth

See `POLICY9_RUNBOOK.md`. This folder intentionally has no `.request.json`;
submission remains a manual user action.
