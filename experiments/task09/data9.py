"""Task 9 data. python data9.py keys            -> list of image keys needed
                python data9.py q <shard> <n>    (nft env, GPU) PickScore + aesthetic
                python data9.py g                (CPU) G (p1/p2 sig, none) + Lc_v2 incl. rseeds"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, json, glob, pickle, time, numpy as np, pandas as pd
sys.path.insert(0, code_dir("task6")); 
from pipe6 import all_keys, get_img, DS, O3
from score4 import meta
O = DATA + "/outputs/task9"; os.makedirs(O, exist_ok=True)
D = DATA + "/datasets"


def keys():
    sel = {ds: {m["id"] for m in meta(ds) if m["split"] == "SELECT"} for ds in DS}
    K = []
    for k in all_keys():
        if k[0] == "x": K.append(k); continue
        ds, (st, i, s, t) = k[1], k[2]
        if i not in sel[ds]: continue
        if st == "base" or (ds == "hypersim" and st in ("real", "real_rot", "real_rotctl")): K.append(k)
    return K


def caption(ds, i, _C={}):
    if ds not in _C: _C[ds] = json.load(open(f"{D}/{ds}/captions.json"))["captions"]
    return _C[ds][str(i)]


def stage_q(shard, n):
    import torch
    sys.path.insert(0, code_dir("DiffusionNFT"))
    from flow_grpo.pickscore_scorer import PickScoreScorer
    from flow_grpo.aesthetic_scorer import AestheticScorer
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    ps = PickScoreScorer(device="cuda"); ae = AestheticScorer(dtype=torch.float32, device="cuda")
    K = keys()[shard::n]; rows = []
    with ThreadPoolExecutor(8) as ex:
        for b in range(0, len(K), 64):
            B = K[b:b + 64]; ims = list(ex.map(get_img, B))
            prompts = [caption(k[1], k[2] if k[0] == "x" else k[2][1]) for k in B]
            with torch.no_grad():
                p = ps(prompts, [Image.fromarray(x) for x in ims]).cpu().numpy() * 26
                a = ae(torch.tensor(np.stack(ims)).permute(0, 3, 1, 2)).cpu().numpy()
            rows += [dict(key=k, pickscore=float(x), aesthetic=float(y)) for k, x, y in zip(B, p, a)]
    pd.DataFrame(rows).to_pickle(f"{O}/q_{shard}.pkl"); print("Q_DONE", shard, len(rows), flush=True)


def _g(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t8
    k, L, seeds = a; img = get_img(k)
    return [dict(key=k, rseed=sd, **{c: v for c, v in compute_t8(img, L, seed=sd).items() if c in ("C_p1_sig", "C_p2_sig", "Lc_v2")}) for sd in seeds]


def stage_g():
    from concurrent.futures import ProcessPoolExecutor
    D0 = {}
    for ds in DS:
        for f in glob.glob(f"{O3}/dlsd/{ds}_*.pkl"): D0.update({("m", ds, k): v for k, v in pickle.load(open(f, "rb")).items()})
    for f in glob.glob(DATA + "/outputs/task4/exdlsd_*.pkl"): D0.update({("x",) + k: v for k, v in pickle.load(open(f, "rb")).items()})
    seeds = {}
    for ds in DS:
        t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"]
        for r in t3.itertuples(): seeds.setdefault(("m", ds, (r.set, r.id, r.seed, r.theta)), []).append(r.rseed)
    K = keys(); K += [("m", ds, ("real", i, -1, 0.0)) for ds in DS for i in sorted({m["id"] for m in meta(ds) if m["split"] == "SELECT"}) if ds != "hypersim"]
    T = [(k, D0[k], seeds.get(k, [0])) for k in K]
    with ProcessPoolExecutor(80) as ex: rows = [r for rr in ex.map(_g, T, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_pickle(f"{O}/g9.pkl"); print("G_DONE", len(rows), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "keys": print(len(keys()))
    elif a[1] == "q": stage_q(int(a[2]), int(a[3]))
    elif a[1] == "g": stage_g()
