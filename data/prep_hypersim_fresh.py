"""Build datasets/hypersim_fresh (same processing as Hypersim in prep_sun_hypersim.py: no crop, center square, 512).

Origin: experiments/task11/prep_fresh.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA
import sys, os, json, collections, numpy as np
from PIL import Image

from vp_reward import center_square_resize
F = DATA + "/datasets/hypersim_fresh"; RAW = DATA + "/datasets/hypersim/raw"
os.makedirs(f"{F}/rgb512", exist_ok=True); os.makedirs(f"{F}/native_sq", exist_ok=True)
sel = json.load(open(f"{F}/selection.json")); meta = []
for i, x in enumerate(sel):
    p = f"{RAW}/{x['rel']}"; im = np.array(Image.open(p).convert("RGB")); h, w = im.shape[:2]; s = min(h, w)
    Image.fromarray(im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).save(f"{F}/native_sq/{i:04d}.png")
    Image.fromarray(center_square_resize(im)).save(f"{F}/rgb512/{i:04d}.png")
    meta.append(dict(id=i, src=p, scene=x["scene"], cam=x["cam"], frame=x["frame"], subset=x["subset"], split="FRESH"))
json.dump(dict(crop=0, note="Task 11 fresh Hypersim split: frames never used in datasets/hypersim; subsets newscene (scenes never used) and reportscene (new frames of REPORT scenes)", native_square=s, items=meta), open(f"{F}/meta.json", "w"), indent=1)
print(len(meta), collections.Counter(m["subset"] for m in meta), "scenes", {k: len({m["scene"] for m in meta if m["subset"] == k}) for k in ("newscene", "reportscene")})
