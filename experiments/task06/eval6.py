"""python eval6.py SELECT|REPORT [--decomp] -> outputs/task6/{eval6_<split>.csv, pass6_<split>.csv, exploit6.csv}"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
for p in ["task3", "task4", "task5", "task6"]: sys.path.insert(0, code_dir(p))

from evalv import evaluate, winlens
from eval4 import deg_p
from eval5 import wrong_dir_p
from rot6 import rot_new
from vp_reward.v4 import SCORES6, SIGF
O = DATA + "/outputs/task6"; O3 = DATA + "/outputs/task3"; DS = ["nyu", "sun_kv2", "hypersim"]
V = [f"{s}@sf{f:g}" for s in SCORES6 for f in SIGF]


def wide(R):
    R = R.copy(); R["k"] = R.key.map(str) + "|" + R.rseed.map(str)
    W = R.pivot_table(index=["k"], columns="sf", values=SCORES6, aggfunc="first", dropna=False)
    W.columns = [f"{a}@sf{b:g}" for a, b in W.columns]
    first = R.drop_duplicates("k").set_index("k")[["key", "rseed"]]
    return first.join(W)


def main_df(R, ds):
    M = wide(R[R.key.map(lambda k: k[0] == "m" and k[1] == ds)])
    M["set"] = M.key.map(lambda k: k[2][0]); M["id"] = M.key.map(lambda k: k[2][1]); M["seed"] = M.key.map(lambda k: k[2][2]); M["theta"] = M.key.map(lambda k: k[2][3])
    t3 = pd.read_pickle(f"{O3}/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"][["set", "id", "seed", "theta", "rseed", "Lc", "total_len", "split"]]
    M = M.drop(columns="key").merge(t3, on=["set", "id", "seed", "theta", "rseed"]); M["det"] = "dlsd"; M["ds"] = ds
    return M


def exploit(X, label="exploit6"):
    o = X[X.pert == "orig"].set_index(["ds", "id"]); rows = []
    for pt, g in X[X.pert != "orig"].groupby("pert", sort=False):
        g = g.set_index(["ds", "id"])
        for v in V:
            d = (g[v] - o[v].reindex(g.index)).dropna().values
            pv = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
            rows.append(dict(pert=pt, var=v, mean_change=float(d.mean()), p_up=pv, cov=float(g[v].notna().mean()), sig_up=bool(d.mean() > 0 and pv < 0.01)))
    for v in V: rows.append(dict(pert="orig", var=v, cov=float(o[v].notna().mean())))
    E = pd.DataFrame(rows); E.to_csv(f"{O}/{label}.csv", index=False); return E


def ex_df(R):
    X = wide(R[R.key.map(lambda k: k[0] == "x")])
    X["ds"] = X.key.map(lambda k: k[1]); X["id"] = X.key.map(lambda k: k[2]); X["pert"] = X.key.map(lambda k: k[3] if k[3] == "orig" else f"{k[3]} {k[4]:g}")
    return X


if __name__ == "__main__":
    split = sys.argv[1]; R = pd.read_pickle(f"{O}/scores6_raw.pkl")
    E = exploit(ex_df(R)); exp = {v: not E[(E["var"] == v) & (E.pert != "orig")].sig_up.any() for v in V}; excov = {v: E[E["var"] == v]["cov"].min() for v in V}
    rows = []
    for ds in DS:
        df = main_df(R, ds); wl = winlens(ds, sorted(df[df.set == "real"].id.unique()))
        for v in V:
            r = evaluate(df, ds, "dlsd", v, split, wl); r.update(rot_new(df, v, split, wl)); r["p_less"] = wrong_dir_p(df, v, split)
            if ds == "hypersim": r["p_deg"] = deg_p(df, v, split)
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{O}/eval6_{split}.csv", index=False); S = []
    for v in V:
        o = {ds: T[(T.ds == ds) & (T["var"] == v)].iloc[0] for ds in DS}; h = o["hypersim"]
        c = dict(var=v, hyp_core=bool(h.wr > .5 and h.p < .01 and h.q_above >= 4 and h.rot_new and h.learn >= 3), hyp_deg=bool(h.wr_deg > .5 and h.p_deg < .01),
                 exploit=exp[v], nyu_sun_ok=bool(o["nyu"].p_less >= .01 and o["sun_kv2"].p_less >= .01),
                 min_cov_twins=min(o[d].cov_gen for d in DS), min_cov_pert=excov[v])
        c["cov_ok"] = bool(c["min_cov_twins"] >= .9 and c["min_cov_pert"] >= .9)
        c["PASSES"] = all([c["hyp_core"], c["hyp_deg"], c["exploit"], c["nyu_sun_ok"], c["cov_ok"]]); S.append(c)
    S = pd.DataFrame(S); S.to_csv(f"{O}/pass6_{split}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(T[["ds", "var", "cov_gen", "wr", "p", "p_less", "wr_deg", "p_deg", "q_above", "p_rot2", "p_4lt2", "p_8lt2", "rot_new", "learn", "rho_len"]].round(4).to_string(index=False))
    print(S.round(3).to_string(index=False))
    P = E[E.pert != "orig"]
    print((P.pivot(index="pert", columns="var", values="mean_change")[V].round(4).astype(str) + P.pivot(index="pert", columns="var", values="sig_up")[V].replace({True: "*", False: ""})).loc[P.pert.unique()].T.to_string())
