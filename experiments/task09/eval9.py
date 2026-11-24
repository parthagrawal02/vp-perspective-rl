"""Task 9: guards + exploit test on combined reward R = zG - lf*P + lq*zQ  (SELECT only)."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, glob, itertools, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
for p in ["task3", "task4", "task6"]: sys.path.insert(0, code_dir(p))
from evalv import winlens
from rot6 import rot_new
from score4 import meta
O = DATA + "/outputs/task9"; DS = ["nyu", "sun_kv2", "hypersim"]
LF = [0, 0.5, 1, 2, 4]; LQ = [0, 0.25, 0.5, 1, 2]; AL = [0.7, 0.8, 0.9]; GS = ["p2", "p1"]; QS = ["pick", "aes", "sum"]


def build():
    G = pd.read_pickle(f"{O}/g9.pkl"); Q = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{O}/q_*.pkl")])
    Q2 = pd.read_pickle(DATA + "/outputs/task2/pick_aes.pkl"); Q2 = Q2[Q2.set == "real"]  # NYU reals (unused for criteria)
    G["k"] = G.key.map(str); Q["k"] = Q.key.map(str)
    T = G.merge(Q[["k", "pickscore", "aesthetic"]], on="k", how="left")
    T["kind"] = T.key.map(lambda k: k[0]); T["ds"] = T.key.map(lambda k: k[1])
    T["set"] = T.key.map(lambda k: k[2][0] if k[0] == "m" else "ex"); T["id"] = T.key.map(lambda k: k[2][1] if k[0] == "m" else k[2])
    T["seed"] = T.key.map(lambda k: k[2][2] if k[0] == "m" else 0); T["theta"] = T.key.map(lambda k: k[2][3] if k[0] == "m" else 0.0)
    T["pert"] = T.key.map(lambda k: None if k[0] == "m" else (k[3] if k[3] == "orig" else f"{k[3]} {k[4]:g}"))
    sp = {(ds, m["id"]): m["split"] for ds in DS for m in meta(ds)}; T["split"] = [sp[(d, i)] for d, i in zip(T.ds, T.id)]
    T = T.rename(columns={"C_p1_sig": "G_p1", "C_p2_sig": "G_p2", "Lc_v2": "Lc", "pickscore": "Q_pick", "aesthetic": "Q_aes"})
    # Lc_base: mean Lc of the caption's base twins (rseed 0), excluding the image itself if it is one (exploit images = seed-0 twin)
    B = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)].set_index(["ds", "id", "seed"]).Lc
    S4 = B.groupby(level=[0, 1]).sum(); N4 = B.groupby(level=[0, 1]).count()
    own = [(B.get((d, i, s), np.nan) if st in ("base", "ex") else np.nan) for d, i, s, st in zip(T.ds, T.id, T.seed, T.set)]
    tot = np.array([S4.get((d, i), np.nan) for d, i in zip(T.ds, T.id)]); cnt = np.array([N4.get((d, i), 0) for d, i in zip(T.ds, T.id)], float)
    own = np.array(own, float); ex = ~np.isnan(own)
    T["Lc_base"] = np.where(ex, (tot - np.nan_to_num(own)) / (cnt - 1), tot / cnt)
    # standardization on base twins (SELECT, rseed 0, all sets)
    ref = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)]
    for c in ["G_p1", "G_p2", "Q_pick", "Q_aes"]:
        T["z" + c] = (T[c] - ref[c].mean()) / ref[c].std()
    s = T.zQ_pick + T.zQ_aes; rs = s[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)]
    T["zQ_sum"] = (s - rs.mean()) / rs.std()
    return T


def R_of(T, g, a, lf, q, lq):
    P = np.maximum(0, 1 - T.Lc / (a * T.Lc_base))
    return T["zG_" + g] - lf * P + lq * (T["zQ_" + q] if lq else 0), P


def exploit(X, col):
    o = X[X.pert == "orig"].set_index(["ds", "id"])[col]; out = {}
    for pt, g in X[X.pert != "orig"].groupby("pert"):
        d = (g.set_index(["ds", "id"])[col] - o).dropna().values
        pv = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
        out[pt] = (float(d.mean()), float(pv), bool(d.mean() > 0 and pv < .01))
    return out


def learn(H, col):
    d0 = H[(H.rseed == 0) & (H.set == "base")]
    bs = d0.groupby("id")[col].std(ddof=0).median()
    ids = H[H.rseed > 0].id.unique()
    sd = H[H.id.isin(ids) & (((H.set == "real") & (H.seed == -1)) | ((H.set == "base") & (H.seed == 0)))]
    return float(bs / max(sd.groupby(["set", "id"])[col].std(ddof=0).mean(), 1e-9))


def geo_corr(M, col, gcol):
    d = M[(M.rseed == 0) & (M.set == "base")]
    r = [spearmanr(g[col], g[gcol]).statistic for _, g in d.groupby(["ds", "id"]) if len(g) == 4 and g[gcol].nunique() > 1]
    return float(np.nanmedian(r))


if __name__ == "__main__":
    T = build(); T.to_pickle(f"{O}/table9_data.pkl")
    print("missing Q:", int(T[(T.rseed == 0) & ((T.kind == "x") | (T.set == "base") | (T.ds == "hypersim"))].Q_pick.isna().sum()), "rows", len(T))
    X = T[T.kind == "x"]; M = T[(T.kind == "m") & (T.split == "SELECT")]; H = M[M.ds == "hypersim"].copy(); H["det"] = "dlsd"
    wl = winlens("hypersim", sorted(H[H.set == "real"].id.unique()))
    rows = []
    for g, a, lf, q, lq in itertools.product(GS, AL, LF, QS, LQ):
        if lf == 0 and a != AL[0]: continue  # alpha irrelevant
        if lq == 0 and q != QS[0]: continue  # Q irrelevant
        Xr = X.assign(R=R_of(X, g, a, lf, q, lq)[0]); e = exploit(Xr, "R")
        Hr = H.assign(R=R_of(H, g, a, lf, q, lq)[0]); Mr = M.assign(R=R_of(M, g, a, lf, q, lq)[0])
        ro = rot_new(Hr, "R", "SELECT", wl)
        r = dict(G=g, alpha=a if lf else np.nan, lf=lf, Q=q if lq else "-", lq=lq, closes=not any(v[2] for v in e.values()),
                 n_sig=sum(v[2] for v in e.values()), sig=",".join(k for k, v in e.items() if v[2]), rot=ro["rot_new"], p_rot2=ro["p_rot2"], p_4lt2=ro["p_4lt2"], p_8lt2=ro["p_8lt2"],
                 learn=learn(Hr, "R"), corr_all=geo_corr(Mr, "R", "zG_" + g), corr_hyp=geo_corr(Hr, "R", "zG_" + g))
        r["acceptable"] = bool(r["closes"] and r["rot"] and r["learn"] >= 3 and r["corr_all"] >= 0.8 - 1e-9)
        r["guard"] = "none" if lf == 0 and lq == 0 else "floor" if lq == 0 else "quality" if lf == 0 else "both"
        rows.append(r)
    S = pd.DataFrame(rows); S["wsum"] = S.lf + S.lq; S.to_csv(f"{O}/grid9.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(S.groupby(["G", "guard"]).agg(n=("closes", "size"), n_close=("closes", "sum"), n_acc=("acceptable", "sum")).to_string())
    for (g, gd), s in S.groupby(["G", "guard"]):
        c = s[s.closes].sort_values(["wsum", "corr_all"], ascending=[True, False])
        print(g, gd, "smallest closing:", "NONE" if c.empty else c.iloc[0][["alpha", "lf", "Q", "lq", "acceptable", "rot", "learn", "corr_all", "corr_hyp"]].to_dict())
    print(S.sort_values(["G", "guard", "n_sig", "wsum"]).groupby(["G", "guard"]).head(3)[["G", "guard", "alpha", "lf", "Q", "lq", "n_sig", "sig", "rot", "learn", "corr_all"]].round(3).to_string(index=False))
    for g in GS:
        A = S[(S.G == g) & S.acceptable].sort_values(["wsum", "corr_all"], ascending=[True, False])
        if len(A):
            pk = A.iloc[0]; print("PICK", pk.to_dict()); break
    else:
        pk = None; print("PICK: none acceptable")
    if pk is not None:
        a, lf, q, lq = pk.alpha if pk.lf else 0.8, pk.lf, pk.Q if pk.lq else "sum", pk.lq
        R, P = R_of(X, pk.G, a, lf, q, lq)
        Xc = X.assign(G=X["zG_" + pk.G], F=-lf * P, Qc=lq * X["zQ_" + q] if lq else 0.0, R=R)
        out = {c: exploit(Xc, c) for c in ["G", "F", "Qc", "R"]}
        D = pd.DataFrame({c: {k: f"{v[0]:+.3f}{'*' if v[2] else ''}" for k, v in out[c].items()} for c in out})
        D.to_csv(f"{O}/pick_decomp.csv"); print(D.to_string())
