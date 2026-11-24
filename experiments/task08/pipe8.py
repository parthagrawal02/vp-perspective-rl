"""Task 8 scoring (CPU). python pipe8.py  -> outputs/task8/scores8_raw.pkl  (pre in {none, sf1.5})"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, pickle, time, numpy as np, pandas as pd
sys.path.insert(0, code_dir("task6")); 
from pipe6 import all_keys, get_img, load_lines, DS, O3
from vp_reward.v4 import preprocess
O = DATA + "/outputs/task8"; os.makedirs(O, exist_ok=True)


def _job(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t8
    key, L_none, L_pre, seeds = a
    img = get_img(key); pre = preprocess(img, 1.5); rows = []
    for tag, src, L in (("none", img, L_none), ("sf1.5", pre, L_pre)):
        for sd in seeds:
            rows.append(dict(pre=tag, key=key, rseed=sd, **compute_t8(src, L, seed=sd)))
    return rows


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    D6 = load_lines(); D0 = {}
    for ds in DS:
        for f in glob.glob(f"{O3}/dlsd/{ds}_*.pkl"): D0.update({("m", ds, k): v for k, v in pickle.load(open(f, "rb")).items()})
    for f in glob.glob(DATA + "/outputs/task4/exdlsd_*.pkl"): D0.update({("x",) + k: v for k, v in pickle.load(open(f, "rb")).items()})
    seeds = {}
    for ds in DS:
        t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"]
        for r in t3.itertuples(): seeds.setdefault(("m", ds, (r.set, r.id, r.seed, r.theta)), []).append(r.rseed)
    T = [(k, D0[k], D6[(1.5,) + k], seeds.get(k, [0])) for k in all_keys()]
    t0 = time.time()
    with ProcessPoolExecutor(80) as ex: rows = [r for rr in ex.map(_job, T, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_pickle(f"{O}/scores8_raw.pkl"); print("SCORED8", len(rows), f"{time.time()-t0:.0f}s", flush=True)
