"""Task 2 analysis: segment cache -> 9-config fits -> validation stats -> selection.

python analysis.py segs      # LSD segment cache for all image sets (CPU, parallel)
python analysis.py fit       # fit/score every cached image under all 9 configs (+ RANSAC-seed std)
python analysis.py stats     # 5a-5e for every candidate on SELECT and REPORT, selection rule
"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image


sys.path.insert(0, code_dir("task2"))
from vp_reward import VPConfig, detect_segments, fit_and_score  # noqa: E402
from imageops import choose_window, degrade, rotate_region  # noqa: E402

OUT = DATA + "/outputs/task2"
GEN = f"{OUT}/gen"
NYU = DATA + "/datasets/nyu/rgb512"
META = json.load(open(DATA + "/datasets/nyu/meta.json"))["items"]
SPLIT = {m["id"]: m["split"] for m in META}
ALL = [m["id"] for m in META]
REP = [i for i in ALL if SPLIT[i] == "REPORT"]
DEG = json.load(open(f"{OUT}/degrade_params.json"))
THETAS = [0.5, 1, 2, 4, 8]
LMINS, TAUS = [0.03, 0.04, 0.06], [1.0, 1.5, 2.0]
CANDS = ["G", "NM", "RegMean", "RegMax"]
NPROC = int(os.environ.get("NPROC", 96))
SEG_MIN = 0.03 * 512  # cache only segments that can survive the smallest L_min


def jobs():
    J = [("real", i, -1, 0.0) for i in ALL]
    J += [("base", i, s, 0.0) for i in ALL for s in range(4)]
    J += [("base_deg", i, s, 0.0) for i in ALL for s in range(4)]
    J += [("real_rot", i, -1, t) for i in ALL for t in THETAS]
    J += [("real_rotctl", i, -1, 8.0) for i in ALL]  # +8 then -8 deg: resampling/blend control
    for v in ("base_suffix", "flowgrpo", "nft"):
        J += [(v, i, s, 0.0) for i in REP for s in range(2)]
    return J


GDIR = {"base": "base_cfg4.5", "base_deg": "base_cfg4.5", "base_suffix": "base_suffix_cfg4.5", "flowgrpo": "flowgrpo_cfg4.5", "nft": "nft_cfg1.0"}


def load(setname, i, s):
    if setname.startswith("real"):
        return np.array(Image.open(f"{NYU}/{i:04d}.png").convert("RGB"))
    return np.array(Image.open(f"{GEN}/{GDIR[setname]}/{i:04d}_s{s}.png").convert("RGB"))


def _filt(seg):
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return seg[L >= SEG_MIN]


def seg_job(j):
    import cv2
    cv2.setNumThreads(1)
    setname, i, s, t = j
    img = load(setname, i, s)
    extra = {}
    if setname == "base_deg":
        img = degrade(img, DEG["down"], DEG["blur_sigma"], DEG["jpeg_q"])
    elif setname.startswith("real_rot"):
        (x0, y0), wl = choose_window(detect_segments(img))
        img = rotate_region(img, x0, y0, t, back_and_forth=(setname == "real_rotctl"))
        extra = dict(win=(x0, y0), win_len=wl)
    return j, _filt(detect_segments(img)), extra


def stage_segs():
    J = jobs()
    t = time.time()
    with ProcessPoolExecutor(NPROC) as ex:
        res = list(ex.map(seg_job, J, chunksize=16))
    cache = {r[0]: r[1] for r in res}
    extra = {r[0]: r[2] for r in res if r[2]}
    pickle.dump(dict(segs=cache, extra=extra), open(f"{OUT}/segcache.pkl", "wb"))
    print(f"segs: {len(J)} images in {time.time()-t:.0f}s on {NPROC} procs")


def cfg_list():
    return [VPConfig(lmin_frac=l, tau=t) for l in LMINS for t in TAUS]


def fit_job(args):
    keys, segs, cfg, seed = args
    rows = []
    for k, sg in zip(keys, segs):
        r = fit_and_score(sg, cfg, seed)
        d = r.diag
        rows.append(dict(set=k[0], id=k[1], seed=k[2], theta=k[3], cfg=cfg.tag(), rseed=seed, **{c: r.scores[c] for c in CANDS},
                         Lc=d["Lc"], total_len=d["total_len"], n_segs=d["n_segs"], n_vps=len(d["vps"]), R_mean=d["R_mean"], R_max=d["R_max"]))
    return rows


def stage_fit():
    C = pickle.load(open(f"{OUT}/segcache.pkl", "rb"))["segs"]
    keys = list(C.keys())
    tasks = []
    B = 64
    for cfg in cfg_list():
        for b in range(0, len(keys), B):
            kk = keys[b:b + B]
            tasks.append((kk, [C[k] for k in kk], cfg, 0))
    # RANSAC-seed variability: 5 seeds on 100 SELECT real + 100 SELECT base(seed0) images, every config
    sel = [i for i in ALL if SPLIT[i] == "SELECT"][::7][:100]
    rk = [("real", i, -1, 0.0) for i in sel] + [("base", i, 0, 0.0) for i in sel]
    for cfg in cfg_list():
        for sd in range(1, 5):
            tasks.append(([(f"seedstd_{k[0]}",) + k[1:] for k in rk], [C[k] for k in rk], cfg, sd))
        tasks.append(([(f"seedstd_{k[0]}",) + k[1:] for k in rk], [C[k] for k in rk], cfg, 0))
    t = time.time()
    with ProcessPoolExecutor(NPROC) as ex:
        rows = [r for rr in ex.map(fit_job, tasks, chunksize=1) for r in rr]
    df = pd.DataFrame(rows)
    df["split"] = df["id"].map(SPLIT)
    df.to_pickle(f"{OUT}/scores.pkl")
    print(f"fit: {len(df)} rows in {time.time()-t:.0f}s")


if __name__ == "__main__":
    {"segs": stage_segs, "fit": stage_fit}[sys.argv[1]]()
