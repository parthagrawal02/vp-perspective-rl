"""Task 4 evaluation. python eval4.py [SELECT|REPORT]  -> outputs/task4/eval4_<split>.csv, exploit.csv"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, code_dir("task3"))
from evalv import evaluate, winlens
from score4 import O, DS

from vp_reward.v4 import VARIANTS


def exploit():
    E = pd.read_pickle(f"{O}/exploit_scores.pkl")
    o = E[E.pert == "orig"].set_index(["ds", "id"])
    rows = []
    for (k, p), g in E[E.pert != "orig"].groupby(["pert", "p"], sort=False):
        g = g.set_index(["ds", "id"])
        for v in VARIANTS:
            d = (g[v] - o[v].reindex(g.index)).dropna().values
            pv = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
            rows.append(dict(pert=f"{k} {p:g}", var=v, n=len(d), mean_change=float(d.mean()), p_up=pv,
                             cov_pert=float(g[v].notna().mean()), cov_orig=float(o[v].notna().mean()), sig_up=bool(d.mean() > 0 and pv < 0.01)))
    X = pd.DataFrame(rows); X.to_csv(f"{O}/exploit.csv", index=False)
    return X


def deg_p(df, v, split):
    d = df[(df.rseed == 0) & (df.split == split)]
    r = d[d.set == "real"].set_index("id")[v].dropna()
    g = d[d.set == "base_deg"].dropna(subset=[v]).groupby("id")[v].mean()
    i = r.index.intersection(g.index); x = (r[i] - g[i]).values
    return float(wilcoxon(x).pvalue)


if __name__ == "__main__":
    split = sys.argv[1]
    X = exploit(); ex_pass = {v: not X[X["var"] == v].sig_up.any() for v in VARIANTS}
    rows = []
    for ds in DS:
        df = pd.read_pickle(f"{O}/scores4_{ds}.pkl")
        wl = winlens(ds, sorted(df[df.set == "real"].id.unique()))
        for v in VARIANTS:
            r = evaluate(df, ds, "dlsd", v, split, wl)
            r["p_deg"] = deg_p(df, v, split); r["deg_ok"] = bool(r["wr_deg"] > 0.5 and r["p_deg"] < 0.01)
            r["exploit_pass"] = ex_pass[v]; r["WORKS_T3"] = r["WORKS"]
            r["WORKS"] = bool(r["WORKS_T3"] and r["deg_ok"] and r["exploit_pass"])
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{O}/eval4_{split}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(T[["ds", "var", "cov_real", "wr", "p", "wr_deg", "p_deg", "q_above", "rot_mono", "p_rot2", "rot_pass", "learn", "rho_len", "exploit_pass", "WORKS_T3", "WORKS"]].round(4).to_string(index=False))
    P = X.pivot(index="pert", columns="var", values="mean_change")[VARIANTS]
    S = X.pivot(index="pert", columns="var", values="sig_up")[VARIANTS]
    print((P.round(4).astype(str) + S.replace({True: "*", False: ""})).loc[X.pert.unique()].to_string())
    print(X.pivot(index="pert", columns="var", values="cov_pert")[VARIANTS].round(3).loc[X.pert.unique()].to_string())
