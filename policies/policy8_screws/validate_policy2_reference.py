from collections import Counter
from pathlib import Path
import sys

import cv2

ROOT = Path("/home/yt-astraf0a5/workspace/physical_ai_submission_policy4")
sys.path.insert(0, str(ROOT))

from sort4_policy.instruction import default_task
from sort4_policy.perception import PerceptionError, Sort4CosmosClient


instruction = (
    "Sort all M4 screws into the red bin, all M8 screws into the yellow bin, "
    "all red cubes into the green bin, and all blue cubes into the blue bin."
)
bgr = cv2.imread(str(ROOT / "sort4_sim_reference.png"))
if bgr is None:
    raise SystemExit("reference image is missing")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
client = Sort4CosmosClient()
try:
    scene = client.infer(rgb, default_task(instruction))
    counts = Counter(item["sort_class"] for item in scene["objects"])
    client.to_targets(scene, default_task(instruction))
except PerceptionError as exc:
    print("PASS_EXPECTED_REJECTION", str(exc))
    raise SystemExit(0)
raise SystemExit(f"ERROR: policy accepted the deliberately wrong reference inventory: {dict(counts)}")
