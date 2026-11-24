"""Task 5 scoring (top-k per cluster). python score5.py score <ds> | exscore | decomp <variant>"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, pickle, time, numpy as np
sys.path.insert(0, code_dir("task4")); 
from score4 import load, degparams, ex_img, O3, DS, PERT  # noqa
O4 = DATA + "/outputs/task4"; O = DATA + "/outputs/task5"; os.makedirs(O, exist_ok=True)


def _job(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t5
    ds, key, seed, dl = a
    return dict(ds=ds, det="dlsd", set=key[0], id=key[1], seed=key[2], theta=key[3], rseed=seed, **compute_t5(load(ds, key, degparams(ds)), dl, seed=seed))


def stage_score(ds):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = {}
    for f in sorted(glob.glob(f"{O3}/dlsd/{ds}_*.pkl")): DL.update(pickle.load(open(f, "rb")))
    t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"]
    tasks = [(ds, (r.set, r.id, r.seed, r.theta), r.rseed, DL[(r.set, r.id, r.seed, r.theta)]) for r in t3.itertuples()]
    t0 = time.time()
    with ProcessPoolExecutor(80) as ex: rows = list(ex.map(_job, tasks, chunksize=8))
    df = pd.DataFrame(rows).merge(t3[["set", "id", "seed", "theta", "rseed", "Lc", "total_len", "split"]], on=["set", "id", "seed", "theta", "rseed"])
    assert len(df) == len(t3)
    df.to_pickle(f"{O}/scores5_{ds}.pkl"); print(ds, "scored", len(df), f"{time.time()-t0:.0f}s", flush=True)


def _exjob(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t5
    key, dl, mode, dlo = a
    if mode == "full": return dict(ds=key[0], id=key[1], pert=key[2], p=key[3], mode=mode, **compute_t5(ex_img(*key), dl))
    if mode == "refine_only": return dict(ds=key[0], id=key[1], pert=key[2], p=key[3], mode=mode, **compute_t5(ex_img(*key), dlo))
    return dict(ds=key[0], id=key[1], pert=key[2], p=key[3], mode=mode, **compute_t5(ex_img(key[0], key[1], "orig", None), dl))


def load_ex():
    DL = {}
    for f in glob.glob(f"{O4}/exdlsd_*.pkl"): DL.update(pickle.load(open(f, "rb")))
    return DL


def stage_ex(modes):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    DL = load_ex(); T = []
    for k, v in DL.items():
        for m in modes:
            if m != "full" and k[2] == "orig": continue
            T.append((k, v, m, DL[(k[0], k[1], "orig", None)]))
    with ProcessPoolExecutor(80) as ex: rows = list(ex.map(_exjob, T, chunksize=4))
    tag = "exploit" if modes == ["full"] else "decomp"
    pd.DataFrame(rows).to_pickle(f"{O}/{tag}_scores.pkl"); print(tag, "scored", len(rows), flush=True)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "score": stage_score(a[2])
    elif a[1] == "exscore": stage_ex(["full"])
    elif a[1] == "decomp": stage_ex(["refine_only", "detect_only"])
