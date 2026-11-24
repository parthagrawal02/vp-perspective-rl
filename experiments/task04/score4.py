"""Task 4 scoring.
  python score4.py score <ds>                     (nft env, CPU) all Task-3 jobs (cached DeepLSD lines) x V0..V3
  python score4.py exdlsd <shard> <nshard>        (deeplsd env, GPU) DeepLSD on perturbed twins
  python score4.py exscore                        (nft env, CPU) exploit-test scores
"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, json, glob, pickle, time, io
import numpy as np
sys.path.insert(0, code_dir("task3")); 
from pipe import load, degparams, jobs, meta, GEN  # noqa
O3 = DATA + "/outputs/task3"; O = DATA + "/outputs/task4"
os.makedirs(O, exist_ok=True)
DS = ["nyu", "sun_kv2", "hypersim"]
PERT = [("orig", None), ("blur", 0.5), ("blur", 1.0), ("blur", 2.0), ("noise", 2), ("noise", 5), ("noise", 10),
        ("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("rescale", 0.75), ("rescale", 0.5), ("contrast", 0.7)]


def _job(a):
    import cv2
    cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t4
    ds, key, seed, dl = a
    o = compute_t4(load(ds, key, degparams(ds)), dl, seed=seed)
    return dict(ds=ds, det="dlsd", set=key[0], id=key[1], seed=key[2], theta=key[3], rseed=seed, **o)


def stage_score(ds):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in sorted(glob.glob(f"{O3}/dlsd/{ds}_*.pkl")): DL.update(pickle.load(open(f, "rb")))
    t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"]
    tasks = [(ds, (r.set, r.id, r.seed, r.theta), r.rseed, DL[(r.set, r.id, r.seed, r.theta)]) for r in t3.itertuples()]
    print(ds, "tasks", len(tasks), flush=True); t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 96))) as ex:
        rows = list(ex.map(_job, tasks, chunksize=8))
    df = pd.DataFrame(rows)
    df = df.merge(t3[["set", "id", "seed", "theta", "rseed", "Lc", "total_len", "split", "s_A1"]], on=["set", "id", "seed", "theta", "rseed"])
    assert len(df) == len(t3), (len(df), len(t3))
    print(ds, "V0 vs Task3 A1 max abs diff", float(np.nanmax(np.abs(df.V0 - df.s_A1))), flush=True)
    df.to_pickle(f"{O}/scores4_{ds}.pkl"); print(ds, "scored", len(df), f"{time.time()-t0:.0f}s", flush=True)


def ex_ids():
    """~500 SELECT base twins (seed 0): 167/167/166 spread over the three sets, deterministic."""
    out = []
    for ds, n in zip(DS, (167, 167, 166)):
        sel = sorted(m["id"] for m in meta(ds) if m["split"] == "SELECT")
        out += [(ds, i) for i in sel[:: max(1, len(sel) // n)][:n]]
    return out


def perturb(img, kind, p, rng):
    import cv2
    if kind == "orig": return img
    if kind == "blur": return cv2.GaussianBlur(img, (0, 0), p)
    if kind == "noise": return np.clip(img + rng.normal(0, p, img.shape), 0, 255).round().astype(np.uint8)
    if kind == "jpeg":
        ok, b = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, p]); return cv2.imdecode(b, 1)[..., ::-1].copy()
    if kind == "rescale":
        h = img.shape[0]; m = int(round(h * p))
        return cv2.resize(cv2.resize(img, (m, m), interpolation=cv2.INTER_AREA), (h, h), interpolation=cv2.INTER_CUBIC)
    if kind == "contrast":
        x = img.astype(np.float32); return np.clip((x - x.mean()) * p + x.mean(), 0, 255).round().astype(np.uint8)


def ex_img(ds, i, kind, p):
    from PIL import Image
    img = np.array(Image.open(f"{GEN[ds]}/{i:04d}_s0.png").convert("RGB"))
    return perturb(img, kind, p, np.random.default_rng(i * 10 + DS.index(ds)))


def stage_exdlsd(shard, nshard):
    import torch, cv2
    from deeplsd.models.deeplsd_inference import DeepLSD
    conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
    net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval(); out = {}
    J = [(ds, i, k, p) for ds, i in ex_ids() for k, p in PERT][shard::nshard]
    for key in J:
        x = torch.tensor(cv2.cvtColor(ex_img(*key), cv2.COLOR_RGB2GRAY)[None, None], dtype=torch.float32, device="cuda") / 255
        with torch.no_grad():
            out[key] = net({"image": x})["lines"][0].reshape(-1, 4).astype(np.float32)
    pickle.dump(out, open(f"{O}/exdlsd_{shard}.pkl", "wb")); print("EXDLSD_DONE", shard, len(out), flush=True)


def _exjob(a):
    import cv2
    cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t4
    key, dl = a
    return dict(ds=key[0], id=key[1], pert=key[2], p=key[3], **compute_t4(ex_img(*key), dl, seed=0))


def stage_exscore():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in glob.glob(f"{O}/exdlsd_*.pkl"): DL.update(pickle.load(open(f, "rb")))
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 96))) as ex:
        rows = list(ex.map(_exjob, list(DL.items()), chunksize=4))
    pd.DataFrame(rows).to_pickle(f"{O}/exploit_scores.pkl"); print("EX scored", len(rows), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "score": stage_score(a[2])
    elif a[1] == "exdlsd": stage_exdlsd(int(a[2]), int(a[3]))
    elif a[1] == "exscore": stage_exscore()
