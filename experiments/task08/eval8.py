"""python eval8.py  (SELECT only) -> outputs/task8/{eval8_SELECT.csv, table8.csv, exploit8.csv}"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
for p in ["task3", "task4", "task5", "task6"]: sys.path.insert(0, code_dir(p))

from evalv import evaluate, winlens
from eval4 import deg_p
from eval5 import wrong_dir_p
from rot6 import rot_new
import eval6
from vp_reward.v4 import NAMES8
O = DATA + "/outputs"; DS = ["nyu", "sun_kv2", "hypersim"]; SPLIT = "SELECT"
V = [f"{n}@{pre}" for pre in ("none", "sf1.5") for n in NAMES8]
REF = {"REF adopted K10 s1 frac @sf1.5": "K10_sig1.0_frac@sf1.5", "REF A1 (V0)": "V0"}
ALL = V + list(REF)


def wide8(R):
    R = R.copy(); R["k"] = R.key.map(str) + "|" + R.rseed.map(str)
    W = R.pivot_table(index="k", columns="pre", values=NAMES8, aggfunc="first", dropna=False)
    W.columns = [f"{a}@{b}" for a, b in W.columns]
    return R.drop_duplicates("k").set_index("k")[["key", "rseed"]].join(W)


def main_df(R8, R6, ds):
    M = wide8(R8[R8.key.map(lambda k: k[0] == "m" and k[1] == ds)])
    for c, f in (("set", 0), ("id", 1), ("seed", 2), ("theta", 3)): M[c] = M.key.map(lambda k: k[2][f])
    M = M.drop(columns="key")
    t3 = pd.read_pickle(f"{O}/task3/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"][["set", "id", "seed", "theta", "rseed", "Lc", "total_len", "split"]]
    a6 = eval6.main_df(R6, ds)[["set", "id", "seed", "theta", "rseed", "K10_sig1.0_frac@sf1.5"]].rename(columns={"K10_sig1.0_frac@sf1.5": "REF adopted K10 s1 frac @sf1.5"})
    v0 = pd.read_pickle(f"{O}/task4/scores4_{ds}.pkl")[["set", "id", "seed", "theta", "rseed", "V0"]].rename(columns={"V0": "REF A1 (V0)"})
    K = ["set", "id", "seed", "theta", "rseed"]
    M = M.merge(t3, on=K).merge(a6, on=K).merge(v0, on=K); M["det"] = "dlsd"; M["ds"] = ds
    return M


def ex_df(R8, R6):
    X = wide8(R8[R8.key.map(lambda k: k[0] == "x")])
    X["ds"] = X.key.map(lambda k: k[1]); X["id"] = X.key.map(lambda k: k[2]); X["pert"] = X.key.map(lambda k: k[3] if k[3] == "orig" else f"{k[3]} {k[4]:g}")
    X6 = eval6.ex_df(R6)[["ds", "id", "pert", "K10_sig1.0_frac@sf1.5"]].rename(columns={"K10_sig1.0_frac@sf1.5": "REF adopted K10 s1 frac @sf1.5"})
    E4 = pd.read_pickle(f"{O}/task4/exploit_scores.pkl"); E4["pert"] = [k if k == "orig" else f"{k} {p:g}" for k, p in zip(E4.pert, E4.p)]
    E4 = E4[["ds", "id", "pert", "V0"]].rename(columns={"V0": "REF A1 (V0)"})
    return X.merge(X6, on=["ds", "id", "pert"]).merge(E4, on=["ds", "id", "pert"])


def exploit(X):
    o = X[X.pert == "orig"].set_index(["ds", "id"]); rows = []
    for pt, g in X[X.pert != "orig"].groupby("pert", sort=False):
        g = g.set_index(["ds", "id"])
        for v in ALL:
            d = (g[v] - o[v].reindex(g.index)).dropna().values
            pv = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
            rows.append(dict(pert=pt, var=v, mean_change=float(d.mean()), p_up=pv, cov=float(g[v].notna().mean()), sig_up=bool(d.mean() > 0 and pv < .01)))
    for v in ALL: rows.append(dict(pert="orig", var=v, cov=float(o[v].notna().mean())))
    E = pd.DataFrame(rows); E.to_csv(f"{O}/task8/exploit8.csv", index=False); return E


def untied(df, v):
    d = df[(df.rseed == 0) & (df.split == SPLIT)]
    r = d[d.set == "real"].set_index("id")[v]; g = d[d.set == "base"][["id", v]].dropna()
    rr = g.id.map(r); ok = rr.notna(); w = (rr[ok] > g[v][ok]).sum(); l = (rr[ok] < g[v][ok]).sum(); t = (rr[ok] == g[v][ok]).sum()
    return w / (w + l), t / ok.sum()


def sens(df, v, wl):
    d = df[(df.rseed == 0) & (df.split == SPLIT)]
    keep = [i for i in d[d.set == "real"].id if wl.get(i, 0) >= 100]
    o = d[d.set == "real"].set_index("id")[v].reindex(keep); out = {}
    for nm, sub in (("r2", d[(d.set == "real_rot") & (d.theta == 2.0)]), ("r8", d[(d.set == "real_rot") & (d.theta == 8.0)]), ("ctl", d[d.set == "real_rotctl"])):
        x = (sub.set_index("id")[v].reindex(keep) - o).dropna()
        out[f"{nm}_lower"] = float((x < 0).mean()); out[f"{nm}_tied"] = float((x == 0).mean()); out[f"{nm}_higher"] = float((x > 0).mean())
    return out


if __name__ == "__main__":
    R8 = pd.read_pickle(f"{O}/task8/scores8_raw.pkl"); R6 = pd.read_pickle(f"{O}/task6/scores6_raw.pkl"); R6 = R6[R6.sf == 1.5]
    E = exploit(ex_df(R8, R6)); exp = {v: not E[(E["var"] == v) & (E.pert != "orig")].sig_up.any() for v in ALL}; excov = {v: E[E["var"] == v]["cov"].min() for v in ALL}
    rows = []
    for ds in DS:
        df = main_df(R8, R6, ds); wl = winlens(ds, sorted(df[df.set == "real"].id.unique()))
        for v in ALL:
            r = evaluate(df, ds, "dlsd", v, SPLIT, wl); r.update(rot_new(df, v, SPLIT, wl)); r["p_less"] = wrong_dir_p(df, v, SPLIT)
            r["wr_untied"], r["tie_frac"] = untied(df, v)
            if ds == "hypersim": r["p_deg"] = deg_p(df, v, SPLIT); r.update(sens(df, v, wl))
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(f"{O}/task8/eval8_SELECT.csv", index=False); S = []
    for v in ALL:
        o = {ds: T[(T.ds == ds) & (T["var"] == v)].iloc[0] for ds in DS}; h = o["hypersim"]
        c = dict(var=v, H_wr_untied=h.wr_untied, H_tie=h.tie_frac, H_wr=h.wr, H_deg=h.wr_deg, H_q=h.q_above, H_rot=h.rot_new, H_learn=h.learn, exploit=exp[v],
                 N_wr=o["nyu"].wr, S_wr=o["sun_kv2"].wr, N_pless=o["nyu"].p_less, S_pless=o["sun_kv2"].p_less,
                 cov_tw=min(o[d].cov_gen for d in DS), cov_pert=excov[v],
                 r2_lo=h.r2_lower, r2_hi=h.r2_higher, r8_lo=h.r8_lower, r8_hi=h.r8_higher, ctl_lo=h.ctl_lower, ctl_tie=h.ctl_tied, ctl_hi=h.ctl_higher)
        hc = bool(h.wr > .5 and h.p < .01 and h.q_above >= 4 and h.rot_new and h.learn >= 3); hd = bool(h.wr_deg > .5 and h.p_deg < .01)
        ns = bool(o["nyu"].p_less >= .01 and o["sun_kv2"].p_less >= .01); cv = bool(c["cov_tw"] >= .9 and c["cov_pert"] >= .9)
        c["fails"] = ",".join(n for n, ok in (("hyp_core", hc), ("hyp_deg", hd), ("exploit", exp[v]), ("nyu_sun", ns), ("cov", cv)) if not ok)
        c["PASSES"] = c["fails"] == ""; S.append(c)
    S = pd.DataFrame(S); S.to_csv(f"{O}/task8/table8.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(S.round(3).to_string(index=False))
    print(T[T.ds == "hypersim"][["var", "wr", "p", "q_above", "p_rot2", "p_4lt2", "p_8lt2", "learn", "wr_deg", "p_deg"]].round(4).to_string(index=False))
    P = E[(E.pert != "orig")]
    print((P.pivot(index="pert", columns="var", values="mean_change")[ALL].round(4).astype(str) + P.pivot(index="pert", columns="var", values="sig_up")[ALL].astype(object).replace({True: "*", False: ""})).T.to_string())
