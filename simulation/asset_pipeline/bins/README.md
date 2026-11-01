# Red and green simulation bins

This isolated candidate package converts `Red-bin.STEP` into one shared visual
USD and two colour-specific simulation assets. It does not modify any existing
scene or policy.

- `packaged/red_bin_sim.usda`: explicit red material and hollow collision
- `packaged/green_bin_sim.usda`: explicit green material and hollow collision
- `converted/red_green_bin_visual.usdc`: shared tessellated STEP visual
- `source/red_green_bin.step`: unchanged SolidWorks STEP export

The STEP is Z-up, centred in X/Y, and rests at Z=0. Its outer dimensions are
90 x 90 x 60 mm. Each simulation asset uses five static box colliders (floor
and four walls), leaving the 80 x 80 mm top opening clear.
