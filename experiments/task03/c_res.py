"""Part C: 1024 vs 512 on ~300 captions per B set (same ids, seeds 0-1).
  python c_res.py dlsd <ds> <shard> <nshard>   (deeplsd env)
  python c_res.py score <ds>                    (nft env)
"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, json, os, pickle, numpy as np
from PIL import Image
D = DATA + "/datasets"; O = DATA + "/outputs/task3"


def ids300(ds):
    caps = json.load(open(f"{D}/{ds}/captions.json"))["captions"]
    ALL = [m["id"] for m in json.load(open(f"{D}/{ds}/meta.json"))["items"] if str(m["id"]) in caps]
    return [i for k, i in enumerate(sorted(ALL, key=lambda i: (i * 2654435761) % 2**32)) if k < 300]  # same rule as gen.py --max_ids


def jobs(ds):
    J = []
    for i in ids300(ds):
        for res in (512, 1024):
            J.append((res, "real", i, -1))
            J += [(res, "base", i, s) for s in (0, 1)]
    return J


def load(ds, key):
    import cv2
    res, st, i, s = key
    if st == "real":
        im = np.array(Image.open(f"{D}/{ds}/native_sq/{i:04d}.png").convert("RGB"))
        return cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA if im.shape[0] > res else cv2.INTER_CUBIC)
    return np.array(Image.open(f"{O}/gen_{ds}_{res}/base_cfg4.5/{i:04d}_s{s}.png").convert("RGB"))


def stage_dlsd(ds, shard, nshard):
    import torch, cv2
    from deeplsd.models.deeplsd_inference import DeepLSD
    conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
    net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval(); out = {}
    for k in jobs(ds)[shard::nshard]:
        x = torch.tensor(cv2.cvtColor(load(ds, k), cv2.COLOR_RGB2GRAY)[None, None], dtype=torch.float32, device="cuda") / 255
        with torch.no_grad():
            out[k] = net({"image": x})["lines"][0].reshape(-1, 4).astype(np.float32)
    pickle.dump(out, open(f"{O}/dlsd/C_{ds}_{shard}.pkl", "wb")); print("C_DLSD_DONE", ds, shard)


def _job(a):
    import cv2
    cv2.setNumThreads(1)

    from vp_reward.v2 import compute_all
    ds, k, dl = a
    img = load(ds, k)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[0]
    lsd = np.zeros((0, 4)) if lsd is None else lsd.reshape(-1, 4)
    rows = []
    for det, seg in (("lsd", lsd), ("dlsd", dl)):
        if seg is None: continue
        o = compute_all(img, seg, seed=0, do_null=False)
        rows.append(dict(ds=ds, det=det, res=k[0], set=k[1], id=k[2], seed=k[3], **o))
    return rows


def stage_score(ds):
    import glob, pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in glob.glob(f"{O}/dlsd/C_{ds}_*.pkl"): DL.update(pickle.load(open(f, "rb")))
    with ProcessPoolExecutor(90) as ex:
        rows = [r for rr in ex.map(_job, [(ds, k, DL.get(k)) for k in jobs(ds)], chunksize=2) for r in rr]
    pd.DataFrame(rows).to_pickle(f"{O}/C_scores_{ds}.pkl"); print("C scored", ds, len(rows))


if __name__ == "__main__":
    if sys.argv[1] == "dlsd": stage_dlsd(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else: stage_score(sys.argv[2])
