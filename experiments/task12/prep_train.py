"""Build datasets/hypersim_train (same processing as task3/prep_refs.py for Hypersim: crop 0, center square, 512 resize)."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, json, collections, numpy as np
from PIL import Image

from vp_reward import center_square_resize
F = DATA + "/datasets/hypersim_train"; RAW = DATA + "/datasets/hypersim/raw"
os.makedirs(f"{F}/rgb512", exist_ok=True); os.makedirs(f"{F}/native_sq", exist_ok=True)
sel = json.load(open(f"{F}/selection.json")); meta = []
for i, x in enumerate(sel):
    p = f"{RAW}/{x['rel']}"; im = np.array(Image.open(p).convert("RGB")); h, w = im.shape[:2]; s = min(h, w)
    Image.fromarray(im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).save(f"{F}/native_sq/{i:04d}.png")
    Image.fromarray(center_square_resize(im)).save(f"{F}/rgb512/{i:04d}.png")
    meta.append(dict(id=i, src=p, scene=x["scene"], cam=x["cam"], frame=x["frame"], split="TRAIN"))
json.dump(dict(crop=0, note="Task 12 RL training prompts: never-used frames of SELECT scenes, >=15 frames apart", native_square=s, items=meta), open(f"{F}/meta.json", "w"), indent=1)
print(len(meta), "scenes", len({m["scene"] for m in meta}))
