"""python eval5.py SELECT|REPORT"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, code_dir("task3")); sys.path.insert(0, code_dir("task4")); 
from evalv import evaluate, winlens
from eval4 import deg_p
from score5 import O, DS
from vp_reward.v4 import VARIANTS5 as V


def exploit():
    E = pd.read_pickle(f"{O}/exploit_scores.pkl"); o = E[E.pert == "orig"].set_index(["ds", "id"]); rows = []
    for (k, p), g in E[E.pert != "orig"].groupby(["pert", "p"], sort=False):
        g = g.set_index(["ds", "id"])
        for v in V:
            d = (g[v] - o[v].reindex(g.index)).dropna().values
            pv = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
            rows.append(dict(pert=f"{k} {p:g}", var=v, n=len(d), mean_change=float(d.mean()), p_up=pv, cov=float(g[v].notna().mean()),
                             sig_up=bool(d.mean() > 0 and pv < 0.01)))
    for v in V: rows.append(dict(pert="orig", var=v, cov=float(o[v].notna().mean())))
    X = pd.DataFrame(rows); X.to_csv(f"{O}/exploit.csv", index=False); return X


def wrong_dir_p(df, v, split):
    d = df[(df.rseed == 0) & (df.split == split)]
    r = d[d.set == "real"].set_index("id")[v].dropna(); g = d[d.set == "base"].dropna(subset=[v]).groupby("id")[v].mean()
    i = r.index.intersection(g.index); x = (r[i] - g[i]).values
    return float(wilcoxon(x, alternative="less").pvalue)


if __name__ == "__main__":
    split = sys.argv[1]; X = exploit()
    exp = {v: not X[(X["var"] == v) & (X.pert != "orig")].sig_up.any() for v in V}
    excov = {v: float(X[X["var"] == v]["cov"].min()) for v in V}
    rows = []
    for ds in DS:
        df = pd.read_pickle(f"{O}/scores5_{ds}.pkl"); wl = winlens(ds, sorted(df[df.set == "real"].id.unique()))
        for v in V:
            r = evaluate(df, ds, "dlsd", v, split, wl); r["p_less"] = wrong_dir_p(df, v, split)
            if ds == "hypersim": r["p_deg"] = deg_p(df, v, split)
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{O}/eval5_{split}.csv", index=False)
    S = []
    for v in V:
        h = T[(T.ds == "hypersim") & (T["var"] == v)].iloc[0]; o = {ds: T[(T.ds == ds) & (T["var"] == v)].iloc[0] for ds in DS}
        c = dict(var=v, hyp_core=bool(h.wr > .5 and h.p < .01 and h.q_above >= 4 and h.rot_pass and h.learn >= 3),
                 hyp_deg=bool(h.wr_deg > .5 and h.p_deg < .01), exploit=exp[v],
                 nyu_sun_ok=bool(o["nyu"].p_less >= .01 and o["sun_kv2"].p_less >= .01),
                 cov_ok=bool(min(o[d].cov_gen for d in DS) >= .9 and excov[v] >= .9), min_cov_twins=min(o[d].cov_gen for d in DS), min_cov_pert=excov[v])
        c["PASSES"] = all([c["hyp_core"], c["hyp_deg"], c["exploit"], c["nyu_sun_ok"], c["cov_ok"]]); S.append(c)
    S = pd.DataFrame(S); S.to_csv(f"{O}/pass5_{split}.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["ds", "var", "cov_real", "cov_gen", "wr", "p", "p_less", "wr_deg", "p_deg", "q_above", "rot_mono", "p_rot2", "rot_pass", "learn", "rho_len"]].round(4).to_string(index=False))
    print(S.round(3).to_string(index=False))
    P = X[X.pert != "orig"]
    print((P.pivot(index="pert", columns="var", values="mean_change")[V].round(4).astype(str) + P.pivot(index="pert", columns="var", values="sig_up")[V].replace({True: "*", False: ""})).loc[P.pert.unique()].to_string())
    print(X.pivot(index="pert", columns="var", values="cov")[V].round(3).to_string())
