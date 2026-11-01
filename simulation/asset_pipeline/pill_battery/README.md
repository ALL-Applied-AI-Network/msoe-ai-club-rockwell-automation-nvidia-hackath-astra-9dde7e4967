# New pill and battery Isaac assets

This package was built only from the four current STEP files in `design files/`.
The previous pill geometry and previous synthetic dataset were not reused.

## Measured CAD

| Class | Measured dimensions | Simulation mass |
|---|---:|---:|
| `round_pill` | 11.5 x 11.5 x 5.5 mm | 0.520 g |
| `long_pill` | 18.7 x 8.5 x 7.2 mm | 1.074 g |
| `aa_battery` | 49.0 x 14.0 x 14.0 mm | 23.0 g |
| `aaa_battery` | 44.0 x 10.2 x 10.2 mm | 11.5 g |

The battery masses use representative physical values. Pill masses are derived
from measured CAD volume at 1200 kg/m^3.

## Package contents

- `converted/`: direct visual USD conversion of each new STEP solid
- `packaged/`: centered metre-scale rigid-body wrappers and collision proxies
- `inspection/object_asset_manifest.json`: dimensions, bounds, hashes and mass
- `inspection/isaac_asset_validation.json`: full structural validation
- `inspection/new_pill_battery_assets.png`: static calibrated-cell preview

Pills have red, yellow, blue and black material variants. Batteries have black,
blue, copper and silver body variants plus metallic terminal overlays. Colour is
appearance variation; the semantic classes remain shape/size classes.

All sixteen materialized assets pass validation: one collision proxy, positive
mass, rigid-body API, resolved dependencies and material bindings. The supplied
long-pill visual tessellation has a render-only open seam, while its convex
physics collision is watertight.

The custom finger opening is 16.4 mm. All four objects fit when grasped across their short
axis. AA is the tightest at 14.0 mm diameter, leaving 2.4 mm total entry
clearance, so it requires accurate centering.

## DGX location

`/home/yt-astraf0a5/workspace/imports/new_pill_battery_assets_candidate_20261029`

## Combined six-class training dataset

The final dataset is on DGX at:

`/home/yt-astraf0a5/workspace/imports/new_pill_battery_assets_candidate_20261029/dataset_combined_1500_seed20261030`

- 1,500 calibrated 1920 x 1080 scenes: 1,200 train, 150 validation, 150 test
- six semantic classes: M4/M8 screws, round/long pills, and AA/AAA batteries
- 14,426 annotated instances with tight 2D boxes and semantic masks
- randomized 0-3 instances per class, while every scene contains at least one
  screw, pill and battery
- randomized pose, appearance colour, CSI-style multi-light shadows, exposure,
  white balance, blur, sensor grain and vignette
- four bins, table artifacts, the FANUC cell and custom fingers retained as context
- colours are appearance variants and are not semantic classes
- COCO exports: `annotations_train.json`, `annotations_validation.json`, and
  `annotations_test.json`
- full audit: `dataset_validation.json` (`PASS`, zero errors/warnings)
- no videos generated

The deterministic generator is `generate_combined_six_class_dataset.py`. The
integrity checker and COCO exporter is `finalize_combined_dataset.py`.
