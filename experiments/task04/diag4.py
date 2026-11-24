"""Decompose the perturbation effect: (a) refinement only = original-image DeepLSD lines on the perturbed image;
(b) detection only = perturbed-image lines refined/scored on the original image."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, glob, pickle, numpy as np, pandas as pd, os
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import wilcoxon
from score4 import ex_img, O
PS = [("blur", 1.0), ("blur", 2.0), ("noise", 10), ("contrast", 0.7), ("rescale", 0.5)]
def job(a):
    import cv2; cv2.setNumThreads(1)
    from vp_reward.v4 import compute_t4
    ds, i, k, p, lo, lp = a
    return [dict(ds=ds, id=i, pert=f"{k} {p:g}", mode="refine_only", **compute_t4(ex_img(ds, i, k, p), lo)),
            dict(ds=ds, id=i, pert=f"{k} {p:g}", mode="detect_only", **compute_t4(ex_img(ds, i, "orig", None), lp)),
            dict(ds=ds, id=i, pert=f"{k} {p:g}", mode="segstats", n_orig=len(lo), n_pert=len(lp),
                 L_orig=float(np.hypot(lo[:,2]-lo[:,0], lo[:,3]-lo[:,1]).sum()), L_pert=float(np.hypot(lp[:,2]-lp[:,0], lp[:,3]-lp[:,1]).sum()))]
if __name__ == "__main__":
    DL = {}
    for f in glob.glob(f"{O}/exdlsd_*.pkl"): DL.update(pickle.load(open(f, "rb")))
    T = [(ds, i, k, p, DL[(ds, i, "orig", None)], DL[(ds, i, k, p)]) for (ds, i, kk, pp) in DL if kk == "orig" for k, p in PS]
    with ProcessPoolExecutor(96) as ex: R = pd.DataFrame([r for rr in ex.map(job, T, chunksize=4) for r in rr])
    E = pd.read_pickle(f"{O}/exploit_scores.pkl"); o = E[E.pert == "orig"].set_index(["ds", "id"])
    E["pert"] = E.pert + " " + E.p.map(lambda x: f"{x:g}" if x == x and x is not None else "")
    E["pert"] = E.pert.str.strip(); E["mode"] = "full"
    A = pd.concat([E[E.pert.isin([f"{k} {p:g}" for k, p in PS])], R[R["mode"] != "segstats"]])
    out = []
    for (pt, m), g in A.groupby(["pert", "mode"]):
        g = g.set_index(["ds", "id"])
        for v in ["V0", "V2_sig0.5", "V2_sig1.0"]:
            d = (g[v] - o[v].reindex(g.index)).dropna()
            out.append(dict(pert=pt, mode=m, var=v, mean_change=round(d.mean(), 4), p_up=wilcoxon(d, alternative="greater").pvalue))
    X = pd.DataFrame(out); X.to_csv(f"{O}/diag_decompose.csv", index=False)
    print(X.pivot_table(index=["pert", "var"], columns="mode", values="mean_change")[["full", "refine_only", "detect_only"]].to_string())
    S = R[R["mode"] == "segstats"]
    print(S.groupby("pert")[["n_orig", "n_pert", "L_orig", "L_pert"]].mean().round(1).to_string())
