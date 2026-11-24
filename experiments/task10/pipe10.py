"""Task 10: contrast stretch (1-99 pct, no blur) before DeepLSD; SELECT-side Task-3 jobs + exploit images.
  python pipe10.py dlsd <shard> <nshard>   (deeplsd env, GPU)  -> outputs/task10/dlsd10_<shard>.pkl
  python pipe10.py g                       (nft env, CPU)      -> outputs/task10/g10.pkl  (G_p1/G_p2 sig + Lc, incl. rseeds)"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, pickle, time, numpy as np
sys.path.insert(0, code_dir("task6")); sys.path.insert(0, code_dir("task4")); 
from pipe6 import all_keys, get_img, DS, O3
from score4 import meta
from vp_reward.v4 import stretch
O = DATA + "/outputs/task10"; os.makedirs(O, exist_ok=True)


def keys():
    sel = {ds: {m["id"] for m in meta(ds) if m["split"] == "SELECT"} for ds in DS}
    return [k for k in all_keys() if k[0] == "x" or k[2][1] in sel[k[1]]]


def stage_dlsd(shard, nshard):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from deeplsd.models.deeplsd_inference import DeepLSD
    conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
    net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval()
    J = keys()[shard::nshard]; out = {}; t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for n, (key, pre) in enumerate(zip(J, ex.map(lambda k: stretch(get_img(k)), J))):
            x = torch.tensor(pre[None, None] / 255, dtype=torch.float32, device="cuda")
            with torch.no_grad():
                out[key] = net({"image": x})["lines"][0].reshape(-1, 4).astype(np.float32)
            if n % 2000 == 0: print(shard, n, len(J), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"{O}/dlsd10_{shard}.pkl", "wb")); print("DLSD10_DONE", shard, len(out), f"{time.time()-t0:.0f}s", flush=True)


def _g(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t8
    k, L, seeds = a; pre = stretch(get_img(k))
    return [dict(key=k, rseed=sd, **{c: v for c, v in compute_t8(pre, L, seed=sd).items() if c in ("C_p1_sig", "C_p2_sig", "Lc_v2")}) for sd in seeds]


def stage_g():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in sorted(glob.glob(f"{O}/dlsd10_*.pkl")): DL.update(pickle.load(open(f, "rb")))
    seeds = {}
    for ds in DS:
        t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"]
        for r in t3.itertuples(): seeds.setdefault(("m", ds, (r.set, r.id, r.seed, r.theta)), []).append(r.rseed)
    K = keys(); T = [(k, DL[k], seeds.get(k, [0])) for k in K]; t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 80))) as ex: rows = [r for rr in ex.map(_g, T, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_pickle(f"{O}/g10.pkl"); print("G10_DONE", len(rows), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "keys": print(len(keys()))
    elif a[1] == "dlsd": stage_dlsd(int(a[2]), int(a[3]))
    elif a[1] == "g": stage_g()
