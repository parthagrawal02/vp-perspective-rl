# NYUv2 labeled -> border-cropped, center-square, 512x512 PNGs + scene-level SELECT/REPORT split.
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import h5py, numpy as np, json, re, os, sys
from PIL import Image

from vp_reward import center_square_resize
OUT = DATA + "/datasets/nyu/rgb512"; os.makedirs(OUT, exist_ok=True)
f = h5py.File(DATA + "/datasets/nyu/raw/nyu_depth_v2_labeled.mat", "r")
scenes = ["".join(chr(c) for c in f[r][:].ravel()) for r in f["scenes"][0]]
stypes = ["".join(chr(c) for c in f[r][:].ravel()) for r in f["sceneTypes"][0]]
I = f["images"]
CROP = (8, 472, 8, 632)  # rows, cols: removes 5-6px white registration pad + blended px (8px margin)
for i in range(I.shape[0]):
    im = I[i].transpose(2, 1, 0)[CROP[0]:CROP[1], CROP[2]:CROP[3]]
    Image.fromarray(center_square_resize(im)).save(f"{OUT}/{i:04d}.png")
uniq = sorted(set(scenes)); rng = np.random.default_rng(0); perm = rng.permutation(len(uniq))
sel_scenes = set(uniq[j] for j in perm[: len(uniq) // 2])
meta = [dict(id=i, scene=s, scene_type=t, split="SELECT" if s in sel_scenes else "REPORT") for i, (s, t) in enumerate(zip(scenes, stypes))]
json.dump(dict(crop_rows_cols=CROP, note="already undistorted (toolbox pipeline + straightness test); 8px border crop; center square 464 -> 512 bicubic", items=meta),
          open(DATA + "/datasets/nyu/meta.json", "w"), indent=1)
from collections import Counter
print(Counter(m["split"] for m in meta), "scenes SELECT", len(sel_scenes), "REPORT", len(uniq) - len(sel_scenes))
print(Counter(m["scene_type"] for m in meta).most_common(8))
