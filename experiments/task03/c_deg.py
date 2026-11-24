"""Part C diagnostic: 1024 twins blurred to the Laplacian variance of the upsampled 1024 reals, rescored.
  python c_deg.py tune|dlsd <ds> [shard nshard]|score <ds>"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, json, pickle, glob, numpy as np, cv2
sys.path.insert(0, code_dir("task3")); sys.path.insert(0, code_dir("task2"))
import c_res
from imageops import laplacian_var
O = c_res.O
def deg(img, ds):
    p = json.load(open(f"{O}/C_deg_{ds}.json")); x = cv2.GaussianBlur(img, (0, 0), p["blur"]) if p["blur"] > 0 else img
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]); return cv2.cvtColor(cv2.imdecode(b, 1), cv2.COLOR_BGR2RGB)
def keys(ds): return [(1024, "base", i, s) for i in c_res.ids300(ds) for s in (0, 1)]
if sys.argv[1] == "tune":
    ds = sys.argv[2]; ids = c_res.ids300(ds)
    tgt = np.mean([laplacian_var(c_res.load(ds, (1024, "real", i, -1))) for i in ids])
    G = [c_res.load(ds, (1024, "base", i, 0)) for i in ids[:150]]
    def f(b):
        json.dump(dict(blur=b), open(f"{O}/C_deg_{ds}.json", "w")); return np.mean([laplacian_var(deg(g, ds)) for g in G])
    lo, hi = 0.0, 4.0
    for _ in range(12):
        m = (lo + hi) / 2; lo, hi = (m, hi) if f(m) > tgt else (lo, m)
    b = (lo + hi) / 2; a = f(b); json.dump(dict(blur=b, target=tgt, achieved=a), open(f"{O}/C_deg_{ds}.json", "w")); print(ds, "blur", round(b, 3), "target", round(tgt, 1), "achieved", round(a, 1))
elif sys.argv[1] == "dlsd":
    import torch
    from deeplsd.models.deeplsd_inference import DeepLSD
    ds, sh, n = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    conf = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}
    ck = torch.load(DATA + "/models/deeplsd/deeplsd_wireframe.tar", map_location="cpu", weights_only=False)
    net = DeepLSD(conf); net.load_state_dict(ck["model"]); net = net.cuda().eval(); out = {}
    for k in keys(ds)[sh::n]:
        x = torch.tensor(cv2.cvtColor(deg(c_res.load(ds, k), ds), cv2.COLOR_RGB2GRAY)[None, None], dtype=torch.float32, device="cuda") / 255
        with torch.no_grad(): out[k] = net({"image": x})["lines"][0].reshape(-1, 4).astype(np.float32)
    pickle.dump(out, open(f"{O}/dlsd/Cdeg_{ds}_{sh}.pkl", "wb")); print("done")
else:
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    ds = sys.argv[2]; DL = {}
    for f in glob.glob(f"{O}/dlsd/Cdeg_{ds}_*.pkl"): DL.update(pickle.load(open(f, "rb")))
    def job(k):
        from vp_reward.v2 import compute_all
        cv2.setNumThreads(1); o = compute_all(deg(c_res.load(ds, k), ds), DL[k], seed=0, do_null=False); return dict(ds=ds, det="dlsd", res=1024, set="base_deg", id=k[2], seed=k[3], **o)
    with ProcessPoolExecutor(90) as ex: rows = list(ex.map(job, keys(ds), chunksize=2))
    pd.DataFrame(rows).to_pickle(f"{O}/C_scores_deg_{ds}.pkl"); print("scored", len(rows))
