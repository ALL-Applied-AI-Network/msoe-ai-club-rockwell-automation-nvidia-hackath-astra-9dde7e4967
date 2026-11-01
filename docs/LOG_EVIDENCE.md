# Log evidence and counting rules

## Evidence inspected

The repository report was prepared from:

- all 79 available Policy 8–11 JSONL runtime logs copied from the final DGX
  policy workspaces;
- the three downloaded CSI physical-run archives for screw, battery, and pill
  development runs;
- final October 31 run IDs for Policies 8–11;
- simulation execution/validation reports;
- synthetic detector and real crop-classifier training/evaluation reports;
- policy integration and replay-test reports.

Raw CSI logs, frames, archives, result links, and request tokens are not
included in the public source repository. The normalized inventory is in
`runtime_log_summary.csv` and `runtime_log_summary.json`. It can be regenerated
from a private log export with:

```bash
python tools/summarize_runtime_logs.py PRIVATE_LOG_ROOT \
  --json-out docs/runtime_log_summary.json \
  --csv-out docs/runtime_log_summary.csv
```

## Counting definitions

### Official trial ID

The CSI run assigns IDs such as `r20261031-0f36a5-t02`. Restarting the local
policy can create multiple JSONL attempt segments with that same official trial
ID. Therefore, official trial IDs and attempt segments are reported separately.

### Completed trial

A trial is counted complete only if at least one attempt segment contains
`event="complete"`. The controller emits this only after all recorded targets
have verified source disappearance and two consecutive final scans find no
remaining supported targets.

### Verified object

An object is counted from `pickup_verified` / the completed inventory of a
completed attempt. This is a vision-based verification that the selected
source object disappeared after the bin motion. It is not a force or tactile
measurement.

### Partial attempt

An attempt without `complete` is retained as partial even if it contains
several verified pickups. A file ending during motion or after only one clean
final scan may have been manually stopped; it is not silently promoted to a
success.

## Final October 31 evidence ledger

### Policy 8 — `r20261031-0f36a5`

| Trial | Segment outcome sequence | Final completed inventory |
| --- | --- | --- |
| `t00` | abort → abort → complete | 2 M4 |
| `t01` | complete | 2 M4, 1 M8 |
| `t02` | abort → complete | 3 M4, 3 M8 |
| `t03` | complete | 3 M4, 3 M8 |

### Policy 9 — `r20261031-36d31f`

| Trial | Segment outcome sequence | Verified work before stop/abort |
| --- | --- | --- |
| `t00` | final audit, no complete | 1 AAA, 1 AA |
| `t01` | complete | 2 AAA, 1 AA |
| `t02` | abort → stopped during verification | second segment: 3 AAA |
| `t03` | stopped during verification | 1 AAA, 1 AA |
| `t04` | no JSONL present | none evidenced |
| `t05` | final audit found residual AA | 2 AAA, 1 AA |
| `t06` | one clean final audit only | 2 AAA, 1 AA |
| `t07` | stopped → abort → stopped | up to 2 AAA in final segment |

Only `t01` meets the completion definition.

### Policy 10 — `r20261031-72fefc`

| Trial | Segment outcome | Completed inventory |
| --- | --- | --- |
| `t00` | complete | 2 round, 2 long |
| `t01` | complete | 2 round, 2 long |

### Policy 11 — `r20261031-ef6221`

| Trial | Segment outcome sequence | Partial verified inventory |
| --- | --- | --- |
| `t00` | perception failure(s) | none |
| `t01` | one clean final audit, no complete | M4, AAA, round pill |
| `t02` | perception failure → residual M8 at final audit | M4, AAA, round pill in retry |

## Development evidence used to explain design changes

Earlier physical logs showed that the first controller descended to only about
`-326` to `-327 mm` while the verified CSI tool/table contact limit was
`-334 mm`. That explains the early missed small-object grasps and led to the
final `-334 mm` grasp depth. An attempted `-336 mm` setting was abandoned after
the CSI limit was confirmed.

Cosmos proposals were useful for semantic inventory, but early screw logs
showed mean XY error around 2.69 mm, maximum around 4.58 mm, and unreliable
axis yaw. The final screw controller therefore used calibrated local silhouette
geometry for shaft-center and yaw, keeping Cosmos only as a fallback proposal
path.

Repeated stalls after the first placement were traced to returning to a pose
that did not reliably clear the camera. Final controllers return to the known
initial camera-clear home pose, reacquire the whole remaining scene, and do not
rely on stale object coordinates.

These development measurements explain the architecture. They are not added
to the final success counts.

## Reproducibility note

The normalized summaries deliberately retain relative private source paths so
the team can map a row back to its internal evidence archive. They do not
contain CSI result URLs, request credentials, or raw images. Before publishing
any additional log excerpt, check it for private keys and personal data.

