"""Task 3 pipeline.
  python pipe.py degtune <ds>                 # tune degradation (diagnostic) to the reference's Laplacian variance
  python pipe.py dlsd <ds> <shard> <nshard>   # DeepLSD lines for all job images (deeplsd env, GPU)
  python pipe.py score <ds>                   # all variants x {lsd, dlsd} (nft env, CPU, parallel)
"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json, os, pickle, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, code_dir("task2"))
from imageops import choose_window, degrade, rotate_region, laplacian_var  # noqa

D = DATA + "/datasets"; O = DATA + "/outputs/task3"
THETAS = [0.5, 1, 2, 4, 8]
REAL = {"nyu": f"{D}/nyu/rgb512", "sun_kv2": f"{D}/sun_kv2/rgb512", "hypersim": f"{D}/hypersim/rgb512"}
GEN = {"nyu": DATA + "/outputs/task2/gen/base_cfg4.5", "sun_kv2": f"{O}/gen_sun_kv2_512/base_cfg4.5", "hypersim": f"{O}/gen_hypersim_512/base_cfg4.5"}


def meta(ds):
    return json.load(open(f"{D}/{ds}/meta.json"))["items"]


def jobs(ds):
    ids = [m["id"] for m in meta(ds)]
    J = [("real", i, -1, 0.0) for i in ids] + [("base", i, s, 0.0) for i in ids for s in range(4)]
    J += [("base_deg", i, s, 0.0) for i in ids for s in range(4)]
    J += [("real_rot", i, -1, t) for i in ids for t in THETAS] + [("real_rotctl", i, -1, 8.0) for i in ids]
    return J


import threading
_TL = threading.local()


def lsd_segs(img):
    import cv2
    if not hasattr(_TL, "lsd"):
        _TL.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)  # one detector per thread (not thread-safe)
    l = _TL.lsd.detect(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0]
    return np.zeros((0, 4), np.float32) if l is None else l.reshape(-1, 4)


def load(ds, key, deg=None):
    st, i, s, t = key
    if st.startswith("real"):
        img = np.array(Image.open(f"{REAL[ds]}/{i:04d}.png").convert("RGB"))
    else:
        img = np.array(Image.open(f"{GEN[ds]}/{i:04d}_s{s}.png").convert("RGB"))
    if st == "base_deg":
        img = degrade(img, deg["down"], deg["blur_sigma"], deg["jpeg_q"])
    elif st.startswith("real_rot"):
        (x0, y0), _ = choose_window(lsd_segs(img))  # window chosen from LSD on the original (same for both detectors)
        img = rotate_region(img, x0, y0, t, back_and_forth=(st == "real_rotctl"))
    return img


def degparams(ds):
    return json.load(open(f"{O}/degrade_{ds}.json"))


def stage_degtune(ds):
    import glob
    r = [laplacian_var(np.array(Image.open(p).convert("RGB"))) for p in sorted(glob.glob(f"{REAL[ds]}/*.png"))]
    g = [np.array(Image.open(p).convert("RGB")) for p in sorted(glob.glob(f"{GEN[ds]}/*_s0.png"))[::3][:250]]
    target = float(np.mean(r)); f = lambda d: np.mean([laplacian_var(degrade(x, d)) for x in g])
    lo, hi = 1.0, 4.0
    blur = 0.5
    if f(1.001) <= target:  # any resample is already too soft: no resize, tune blur sigma instead
        d = 1.0
        fb = lambda b: np.mean([laplacian_var(degrade(x, 1.0, b)) for x in g])
        blo, bhi = 0.0, 1.5
        for _ in range(10):
            mid = (blo + bhi) / 2; blo, bhi = (mid, bhi) if fb(mid) > target else (blo, mid)
        blur = round((blo + bhi) / 2, 3); f = lambda d: fb(blur)
    else:
        for _ in range(12):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
        d = (lo + hi) / 2
    out = dict(down=round(d, 3), blur_sigma=blur, jpeg_q=75, target_lapvar=target, gen_lapvar=float(np.mean([laplacian_var(x) for x in g])), achieved=float(f(d)))
    json.dump(out, open(f"{O}/degrade_{ds}.json", "w")); print(ds, out)


def stage_dlsd(ds, shard, nshard):
    import torch, cv2
    from deeplsd.models.deeplsd_inference import DeepLSD
    from concurrent.futures import ThreadPoolExecutor
    conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
    net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval()
    J = jobs(ds)[shard::nshard]; deg = degparams(ds); out = {}
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        imgs = ex.map(lambda k: load(ds, k, deg), J)
        for n, (k, img) in enumerate(zip(J, imgs)):
            x = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[None, None], dtype=torch.float32, device="cuda") / 255
            with torch.no_grad():
                L = net({"image": x})["lines"][0]
            out[k] = L.reshape(-1, 4).astype(np.float32)  # [x1,y1,x2,y2]; NO line_refiner / VP optimisation
            if n % 2000 == 0: print(ds, shard, n, len(J), f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(f"{O}/dlsd", exist_ok=True)
    pickle.dump(out, open(f"{O}/dlsd/{ds}_{shard}.pkl", "wb")); print("DLSD_SHARD_DONE", ds, shard, flush=True)


_DL = None


def _score_job(args):
    import cv2
    cv2.setNumThreads(1)
    from vp_reward.v2 import compute_all
    ds, key, seed, dl = args
    img = load(ds, key, degparams(ds))
    rows = []
    for det, seg in (("lsd", lsd_segs(img)), ("dlsd", dl)):
        if seg is None:
            continue
        o = compute_all(img, seg, seed=seed)
        rows.append(dict(ds=ds, det=det, set=key[0], id=key[1], seed=key[2], theta=key[3], rseed=seed, **o))
    return rows


def stage_score(ds):
    import glob
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in sorted(glob.glob(f"{O}/dlsd/{ds}_*.pkl")):
        DL.update(pickle.load(open(f, "rb")))
    J = jobs(ds)
    M = {m["id"]: m["split"] for m in meta(ds)}
    sel = [i for i in M if M[i] == "SELECT"][::max(1, len([i for i in M if M[i] == "SELECT"]) // 100)][:100]
    extra = [(("real", i, -1, 0.0), sd) for i in sel for sd in range(1, 5)] + [(("base", i, 0, 0.0), sd) for i in sel for sd in range(1, 5)]
    tasks = [(ds, k, 0, DL.get(k)) for k in J] + [(ds, k, sd, DL.get(k)) for k, sd in extra]
    print(ds, "tasks", len(tasks), "dlsd cached", len(DL), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 100))) as ex:
        rows = [r for rr in ex.map(_score_job, tasks, chunksize=4) for r in rr]
    df = pd.DataFrame(rows); df["split"] = df["id"].map(M)
    df.to_pickle(f"{O}/scores_{ds}.pkl"); print(ds, "scored", len(df), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "degtune": stage_degtune(sys.argv[2])
    elif st == "dlsd": stage_dlsd(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif st == "score": stage_score(sys.argv[2])
