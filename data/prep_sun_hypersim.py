"""SUN RGB-D kv2 and Hypersim reference sets -> datasets/{sun_kv2,hypersim}/{rgb512,native_sq}/ + meta.json.
SUN: kv2 color images (730x530, registered), grouped into capture sessions (15-min chaining), <= 12 per session, 1000 sampled,
6 px border crop. Hypersim: <= 3 tonemap.jpg frames per scene (data/hypersim_dl.py), 1000 sampled, no crop. Session/scene-level
SELECT/REPORT split (seed 0). Input: $VP_DATA/datasets/sunrgbd/SUNRGBD/kv2/..., $VP_DATA/datasets/hypersim/raw/...

Origin: experiments/task03/prep_refs.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA
import glob, json, os, re, sys, collections, numpy as np
from PIL import Image

from vp_reward import center_square_resize
D = DATA + "/datasets"
def build(name, items, crop, note):
    """items: list of (path, scene_group). crop: px border removed on all sides."""
    out = f"{D}/{name}"; os.makedirs(f"{out}/rgb512", exist_ok=True); os.makedirs(f"{out}/native_sq", exist_ok=True)
    rng = np.random.default_rng(0)
    groups = sorted(set(g for _, g in items)); perm = rng.permutation(len(groups)); sel = set(groups[j] for j in perm[: len(groups) // 2])
    meta = []
    for i, (p, g) in enumerate(items):
        im = np.array(Image.open(p).convert("RGB")); im = im[crop:im.shape[0] - crop, crop:im.shape[1] - crop]
        h, w = im.shape[:2]; s = min(h, w); sq = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        Image.fromarray(sq).save(f"{out}/native_sq/{i:04d}.png"); Image.fromarray(center_square_resize(im)).save(f"{out}/rgb512/{i:04d}.png")
        meta.append(dict(id=i, src=p, scene=g, split="SELECT" if g in sel else "REPORT"))
    json.dump(dict(crop=crop, note=note, native_square=s, items=meta), open(f"{out}/meta.json", "w"), indent=1)
    c = collections.Counter(m["split"] for m in meta); print(name, len(meta), dict(c), "groups", len(groups), "native square", s)
rng = np.random.default_rng(0)
# SUN RGB-D kv2: group by capture session timestamp; <=12 images per session (100 sessions, 15-min chaining); 1000 total
J = sorted(glob.glob(f"{D}/sunrgbd/SUNRGBD/kv2/*/*/image/*.jpg"))
import datetime as dt
def ts(p): return dt.datetime.strptime(re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", p).group(1), "%Y-%m-%d_%H-%M-%S")
_t = sorted(set(ts(p) for p in J)); _g = {}; gid = 0
for a, b in zip([None] + _t[:-1], _t):
    if a is None or (b - a).total_seconds() > 900: gid += 1
    _g[b] = f"sess{gid:04d}"
def sess(p): return _g[ts(p)]
by = collections.defaultdict(list)
for p in J: by[sess(p)].append(p)
pool = [p for s in sorted(by) for p in list(rng.permutation(by[s]))[:12]]
pool = [pool[j] for j in sorted(rng.choice(len(pool), min(1000, len(pool)), replace=False))]
build("sun_kv2", [(p, sess(p)) for p in pool], 6, "SUN RGB-D kv2 (730x530 registered color, fx=fy=529.5, pp at center); undistorted per sweep; 6px border crop (bottom rows corrupted)")
# Hypersim: <=3 tonemap frames per scene (as downloaded), 1000 total
H = sorted(glob.glob(f"{D}/hypersim/raw/*/images/*/*.tonemap.jpg"))
H = [H[j] for j in sorted(rng.choice(len(H), min(1000, len(H)), replace=False))]
build("hypersim", [(p, p.split("/raw/")[1].split("/")[0]) for p in H], 0, "Hypersim tonemap.jpg 1024x768 (CC BY-SA 3.0), pinhole render; no border")
