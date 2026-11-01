# Five-class sorting candidate

Prepared for the next CSI task with the custom fingers. This package is isolated
from the submitted policies and does not modify policy 1–7.

## Shape classes

- `m4_screw`: delivered black-steel M4 CAD, 7 x 7 x 24 mm
- `m8_screw`: delivered black-steel M8 CAD, 13 x 13 x 28 mm
- `cube`: 10 x 10 x 10 mm, red/blue/yellow
- `long_pill`: delivered STEP, 19.5 x 6.5 x 6.5 mm, red/yellow
- `round_pill`: delivered STEP, 10 x 10 x 5 mm, red/yellow

Colour is appearance metadata, not a separate shape class.

## Isaac-ready pill assets

The `packaged/` directory contains red and yellow simulation wrappers for both
pill shapes. Each has a watertight visual mesh, a generated collision mesh,
mass, rigid-body properties, and material binding. Structural validation is in
`inspection/isaac_asset_validation.json`.

## Synthetic dataset

DGX location:

`/home/yt-astraf0a5/workspace/imports/five_class_assets_candidate_20261029/dataset_500_seed20261029`

Contents:

- 500 calibrated overhead RGB images at 1920 x 1080
- tight 2D boxes and class mappings
- semantic masks and class mappings
- exact pose, colour, dimensions, seed, count, and light level per scene
- one to three instances of every class per scene
- the FANUC cell, custom fingers, and four current bins as background context
- no videos

`dataset_500_summary/dataset_validation.json` records the full PASS audit.

## Reproduction

The generator is `generate_five_class_dataset.py`. The validator is
`validate_five_class_dataset.py`. Both are also installed in the corresponding
DGX import directory.

## Still required before a five-way robot run

1. Define the destination for each of the five shape classes. The current scene
   has four bins, so either add a fifth bin or explicitly group two classes.
2. Add the two pill classes and their grasp dimensions to a new policy derived
   from the latest stable policy; do not overwrite a submitted policy.
3. Train/evaluate a local five-class detector using this dataset, then combine
   its geometry with Cosmos semantic intent. This dataset does not directly
   retrain the CSI-hosted Cosmos service.
4. Run headless simulation grasp trials for both pills before a physical slot.
