"""Task 6.  python pipe6.py dlsd <shard> <nshard>   (deeplsd env)   |   python pipe6.py score   |   python pipe6.py decomp
Keys: ("m", ds, pipe_key) for Task-3 jobs, ("x", ds, id, kind, p) for exploit images. Lines stored per sigma_f."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, pickle, time, numpy as np
sys.path.insert(0, code_dir("task4")); 
from score4 import load, degparams, ex_img, ex_ids, jobs, PERT, DS, O3  # noqa
from vp_reward.v4 import preprocess, SIGF
O = DATA + "/outputs/task6"; os.makedirs(O, exist_ok=True)
DEG = {ds: degparams(ds) for ds in DS}


def all_keys():
    K = [("m", ds, k) for ds in DS for k in jobs(ds)]
    return K + [("x", ds, i, k, p) for ds, i in ex_ids() for k, p in PERT]


def get_img(key):
    return load(key[1], key[2], DEG[key[1]]) if key[0] == "m" else ex_img(*key[1:])


def stage_dlsd(shard, nshard):
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from deeplsd.models.deeplsd_inference import DeepLSD
    conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
    net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval()
    J = all_keys()[shard::nshard]; out = {}; t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for n, (key, pres) in enumerate(zip(J, ex.map(lambda k: [preprocess(get_img(k), s) for s in SIGF], J))):
            for s, pre in zip(SIGF, pres):
                x = torch.tensor(pre[None, None] / 255, dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    out[(s,) + key] = net({"image": x})["lines"][0].reshape(-1, 4).astype(np.float32)
            if n % 2000 == 0: print(shard, n, len(J), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"{O}/dlsd6_{shard}.pkl", "wb")); print("DLSD6_DONE", shard, f"{time.time()-t0:.0f}s", flush=True)


def load_lines():
    DL = {}
    for f in sorted(glob.glob(f"{O}/dlsd6_*.pkl")): DL.update(pickle.load(open(f, "rb")))
    return DL


def _job(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t6
    key, sfs_lines, seeds = a  # one image, all sigma_f, all rseeds
    img = get_img(key); rows = []
    for s, L in sfs_lines:
        pre = preprocess(img, s)
        for sd in seeds:
            rows.append(dict(sf=s, key=key, rseed=sd, **compute_t6(pre, L, seed=sd)))
    return rows


def stage_score():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = load_lines(); seeds = {}
    for ds in DS:
        t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"]
        for r in t3.itertuples(): seeds.setdefault(("m", ds, (r.set, r.id, r.seed, r.theta)), []).append(r.rseed)
    T = [(k, [(s, DL[(s,) + k]) for s in SIGF], seeds.get(k, [0])) for k in all_keys()]
    t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 80))) as ex:
        rows = [r for rr in ex.map(_job, T, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_pickle(f"{O}/scores6_raw.pkl"); print("SCORED6", len(rows), f"{time.time()-t0:.0f}s", flush=True)


def _djob(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t6
    s, key, Lp, Lo = a
    pre_p = preprocess(ex_img(*key[1:]), s); pre_o = preprocess(ex_img(key[1], key[2], "orig", None), s)
    return [dict(sf=s, key=key, mode="refine_only", **compute_t6(pre_p, Lo)), dict(sf=s, key=key, mode="detect_only", **compute_t6(pre_o, Lp))]


def stage_decomp():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = load_lines()
    T = [(s, k, DL[(s,) + k], DL[(s, "x", k[1], k[2], "orig", None)]) for k in all_keys() if k[0] == "x" and k[3] != "orig" for s in SIGF]
    with ProcessPoolExecutor(80) as ex: rows = [r for rr in ex.map(_djob, T, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_pickle(f"{O}/decomp6.pkl"); print("DECOMP6", len(rows), flush=True)


if __name__ == "__main__":
    a = sys.argv
    {"dlsd": lambda: stage_dlsd(int(a[2]), int(a[3])), "score": stage_score, "decomp": stage_decomp}[a[1]]()
