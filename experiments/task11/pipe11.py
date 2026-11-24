"""Task 11 pipeline on the fresh Hypersim split (ds tag 'hf'). Frozen reward pieces from vp_reward/final.py.
  python pipe11.py dlsd <shard> <nshard>   (deeplsd env, GPU)  -> outputs/task11/dlsd11_<shard>.pkl
  python pipe11.py g                       (nft env, CPU)      -> outputs/task11/g11.pkl   (G p2 sig raw, Lc; rseeds for learnability)
  python pipe11.py q <shard> <nshard>      (nft env, GPU)      -> outputs/task11/q11_<shard>.pkl (PickScore x26, aesthetic)
Keys: ("m", set, id, seed, theta) Task-3-style jobs; ("b", variant, id, seed) baseline rows; ("x", id, kind, p) exploit (seed-0 twins)."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, json, pickle, time, numpy as np
for p in ["vp_reward", "task2", "task3", "task4"]: sys.path.insert(0, code_dir(p))
from PIL import Image
O = DATA + "/outputs/task11"; os.makedirs(O, exist_ok=True)
D = DATA + "/datasets/hypersim_fresh"; GEN = f"{O}/gen"
THETAS = [0.5, 1, 2, 4, 8]
PERT = [("orig", None), ("blur", 0.5), ("blur", 1.0), ("blur", 2.0), ("noise", 2), ("noise", 5), ("noise", 10),
        ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("rescale", 0.75), ("rescale", 0.5), ("contrast", 0.7)]
BVARS = ["base_cfg1.0", "base_suffix_cfg4.5", "flowgrpo_cfg4.5", "nft_cfg1.0"]
DEG = json.load(open(DATA + "/outputs/task3/degrade_hypersim.json"))  # same degradation params as Task 3 Hypersim (not re-tuned)


def ids():
    return sorted(int(k) for k in json.load(open(f"{D}/captions.json"))["captions"])


def caption(i, _C={}):
    if not _C: _C.update(json.load(open(f"{D}/captions.json"))["captions"])
    return _C[str(i)]


def ex_ids():  # 500 exploit twins: same hash order rule as task3 gen.py --max_ids
    return sorted(sorted(ids(), key=lambda i: (i * 2654435761) % 2**32)[:500])


def rs_ids():  # 100 images for RANSAC-seed std (same rule as task3 pipe.stage_score)
    I = ids(); return I[::max(1, len(I) // 100)][:100]


def keys():
    I = ids(); K = [("m", "real", i, -1, 0.0) for i in I] + [("m", st, i, s, 0.0) for st in ("base", "base_deg") for i in I for s in range(4)]
    K += [("m", "real_rot", i, -1, float(t)) for i in I for t in THETAS] + [("m", "real_rotctl", i, -1, 8.0) for i in I]
    K += [("b", v, i, s) for v in BVARS for i in I for s in (0, 1)]
    return K + [("x", i, k, p) for i in ex_ids() for k, p in PERT]


def qkeys():
    return [k for k in keys() if (k[0] == "m" and k[1] in ("real", "base")) or k[0] in ("b", "x")]


def rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def get_img(k):
    from imageops import choose_window, degrade, rotate_region
    from score4 import perturb
    if k[0] == "b": return rgb(f"{GEN}/{k[1]}/{k[2]:04d}_s{k[3]}.png")
    if k[0] == "x": return perturb(rgb(f"{GEN}/base_cfg4.5/{k[1]:04d}_s0.png"), k[2], k[3], np.random.default_rng(k[1] * 10 + 3))
    _, st, i, s, t = k
    if st.startswith("real"):
        img = rgb(f"{D}/rgb512/{i:04d}.png")
        if st.startswith("real_rot"):
            (x0, y0), _ = choose_window(lsd_segs(img)); img = rotate_region(img, x0, y0, t, back_and_forth=(st == "real_rotctl"))
        return img
    img = rgb(f"{GEN}/base_cfg4.5/{i:04d}_s{s}.png")
    return degrade(img, DEG["down"], DEG["blur_sigma"], DEG["jpeg_q"]) if st == "base_deg" else img


def lsd_segs(img):
    import cv2
    l = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0]
    return np.zeros((0, 4), np.float32) if l is None else l.reshape(-1, 4)


def winlen(i):
    from imageops import choose_window
    return choose_window(lsd_segs(rgb(f"{D}/rgb512/{i:04d}.png")))[1]


def stage_dlsd(shard, nshard, kinds="mbx"):
    from concurrent.futures import ThreadPoolExecutor
    from vp_reward.final import load_deeplsd, detect_lines
    from vp_reward.v4 import stretch
    net = load_deeplsd("cuda"); J = [k for k in keys() if k[0] in kinds][shard::nshard]; out = {}; t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for n, (k, pre) in enumerate(zip(J, ex.map(lambda k: stretch(get_img(k)), J))):
            out[k] = detect_lines(net, pre)
            if n % 2000 == 0: print(shard, n, len(J), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"{O}/dlsd11_{kinds}_{shard}.pkl", "wb")); print("DLSD11_DONE", shard, len(out), f"{time.time()-t0:.0f}s", flush=True)


def _g(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import stretch, compute_t8
    from vp_reward.final import LMIN_FRAC, SIGMA0
    k, L, seeds = a; pre = stretch(get_img(k)); rows = []
    for sd in seeds:
        o = compute_t8(pre, L, seed=sd, lmin_frac=LMIN_FRAC, s0=SIGMA0); rows.append(dict(key=k, rseed=sd, G=o["C_p2_sig"], Lc=o["Lc_v2"]))
    return rows


def stage_g(kinds="mbx"):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in sorted(glob.glob(f"{O}/dlsd11_*.pkl")): DL.update(pickle.load(open(f, "rb")))
    R = set(rs_ids())
    seeds = lambda k: [0, 1, 2, 3, 4] if k[0] == "m" and k[2] in R and (k[1] == "real" or (k[1] == "base" and k[3] == 0)) else [0]
    T = [(k, DL[k], seeds(k)) for k in keys() if k[0] in kinds]; t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 80))) as ex: rows = [r for rr in ex.map(_g, T, chunksize=4) for r in rr]
    if "m" in kinds:
        with ProcessPoolExecutor(64) as ex: wl = dict(zip(ids(), ex.map(winlen, ids(), chunksize=8)))
        pickle.dump(wl, open(f"{O}/winlen11.pkl", "wb"))
    pd.DataFrame(rows).to_pickle(f"{O}/g11_{kinds}.pkl"); print("G11_DONE", len(rows), f"{time.time()-t0:.0f}s", flush=True)


def stage_q(shard, n, kinds="mbx"):
    import torch, pandas as pd
    sys.path.insert(0, code_dir("DiffusionNFT"))
    from concurrent.futures import ThreadPoolExecutor
    from flow_grpo.pickscore_scorer import PickScoreScorer
    from flow_grpo.aesthetic_scorer import AestheticScorer
    from vp_reward.final import quality_batch
    ps = PickScoreScorer(device="cuda", dtype=torch.float32); ae = AestheticScorer(dtype=torch.float32, device="cuda")
    K = [k for k in qkeys() if k[0] in kinds][shard::n]; rows = []
    with ThreadPoolExecutor(8) as ex:
        for b in range(0, len(K), 64):
            B = K[b:b + 64]; ims = np.stack(list(ex.map(get_img, B)))
            cid = [k[2] if k[0] in ("m", "b") else k[1] for k in B]  # PickScore always uses the plain caption (no suffix)
            p, a = quality_batch(ps, ae, ims, [caption(i) for i in cid])
            rows += [dict(key=k, pickscore=float(x), aesthetic=float(y)) for k, x, y in zip(B, p, a)]
    pd.DataFrame(rows).to_pickle(f"{O}/q11_{kinds}_{shard}.pkl"); print("Q11_DONE", shard, len(rows), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "keys": print(len(keys()), len(qkeys()))
    elif a[1] == "dlsd": stage_dlsd(int(a[2]), int(a[3]), a[4] if len(a) > 4 else "mbx")
    elif a[1] == "g": stage_g(a[2] if len(a) > 2 else "mbx")
    elif a[1] == "q": stage_q(int(a[2]), int(a[3]), a[4] if len(a) > 4 else "mbx")
