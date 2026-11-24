"""Task 10 eval (SELECT only).  python eval10.py sanity | grid
sanity -> outputs/task10/sanity10.csv, exploitG10.csv ; grid -> grid10.csv, pick10_decomp.csv, table10_data.pkl"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, glob, itertools, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
for p in ["task3", "task4", "task5", "task6", "task8", "task9"]: sys.path.insert(0, code_dir(p))
from evalv import evaluate, winlens
from eval4 import deg_p
from eval5 import wrong_dir_p
from rot6 import rot_new
from score4 import meta
import eval8
from eval9 import exploit, learn, geo_corr
O = DATA + "/outputs"; O10 = f"{O}/task10"; DS = ["nyu", "sun_kv2", "hypersim"]; SPLIT = "SELECT"
eval8.SPLIT = SPLIT
AL = [0.7, 0.8, 0.9]; LF = [0, 0.5, 1, 2, 4]; QS = ["pick", "sum"]; LQ = [0, 0.25, 0.5, 1]


def unpack(T):
    T["kind"] = T.key.map(lambda k: k[0]); T["ds"] = T.key.map(lambda k: k[1])
    T["set"] = T.key.map(lambda k: k[2][0] if k[0] == "m" else "ex"); T["id"] = T.key.map(lambda k: k[2][1] if k[0] == "m" else k[2])
    T["seed"] = T.key.map(lambda k: k[2][2] if k[0] == "m" else 0); T["theta"] = T.key.map(lambda k: k[2][3] if k[0] == "m" else 0.0)
    T["pert"] = T.key.map(lambda k: None if k[0] == "m" else (k[3] if k[3] == "orig" else f"{k[3]} {k[4]:g}"))
    return T


def g_both():
    G10 = pd.read_pickle(f"{O10}/g10.pkl").rename(columns={"C_p2_sig": "G_st", "C_p1_sig": "G1_st", "Lc_v2": "Lc_st"})
    R8 = pd.read_pickle(f"{O}/task8/scores8_raw.pkl"); R8 = R8[R8.pre == "none"][["key", "rseed", "C_p2_sig"]].rename(columns={"C_p2_sig": "G_none"})
    G10["k"] = G10.key.map(str) + "|" + G10.rseed.map(str); R8["k"] = R8.key.map(str) + "|" + R8.rseed.map(str)
    return unpack(G10.merge(R8.drop(columns=["key", "rseed"]), on="k", how="left"))


def sanity():
    T = g_both(); V = ["G_none", "G_st"]; rows = []
    for ds in DS:
        df = T[(T.kind == "m") & (T.ds == ds)][["set", "id", "seed", "theta", "rseed"] + V]
        t3 = pd.read_pickle(f"{O}/task3/scores_{ds}.pkl"); t3 = t3[t3.det == "dlsd"][["set", "id", "seed", "theta", "rseed", "Lc", "total_len", "split"]]
        df = df.merge(t3, on=["set", "id", "seed", "theta", "rseed"]); df["det"] = "dlsd"; df["ds"] = ds
        wl = winlens(ds, sorted(df[df.set == "real"].id.unique()))
        for v in V:
            r = evaluate(df, ds, "dlsd", v, SPLIT, wl); r.update(rot_new(df, v, SPLIT, wl)); r["p_less"] = wrong_dir_p(df, v, SPLIT)
            r["wr_untied"], r["tie_frac"] = eval8.untied(df, v)
            if ds == "hypersim": r["p_deg"] = deg_p(df, v, SPLIT); r.update(eval8.sens(df, v, wl))
            rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f"{O10}/sanity10.csv", index=False)
    X = T[T.kind == "x"]; E = {v: exploit(X[X.rseed == 0], v) for v in V}
    EX = pd.DataFrame({v: {k: f"{a:+.3f}{'*' if s else ''} (p={p:.3g})" for k, (a, p, s) in E[v].items()} for v in V}); EX.to_csv(f"{O10}/exploitG10.csv")
    pd.set_option("display.width", 300, "display.max_columns", 50)
    print(S[["ds", "var", "cov_gen", "wr", "p", "wr_untied", "tie_frac", "p_less", "wr_deg", "p_deg", "q_above", "p_rot2", "p_4lt2", "p_8lt2", "rot_new", "learn", "r2_lower", "r2_tied", "r2_higher"]].round(4).to_string(index=False))
    print(EX.to_string())
    print("cov exploit:", {v: float(X[v].notna().mean()) for v in V})


def build():
    G = pd.read_pickle(f"{O10}/g10.pkl"); Q = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{O}/task9/q_*.pkl")])
    G["k"] = G.key.map(str); Q["k"] = Q.key.map(str)
    T = unpack(G.merge(Q[["k", "pickscore", "aesthetic"]], on="k", how="left"))
    sp = {(ds, m["id"]): m["split"] for ds in DS for m in meta(ds)}; T["split"] = [sp[(d, i)] for d, i in zip(T.ds, T.id)]
    T = T.rename(columns={"C_p2_sig": "G_p2", "Lc_v2": "Lc", "pickscore": "Q_pick", "aesthetic": "Q_aes"})
    B = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)].set_index(["ds", "id", "seed"]).Lc
    S4 = B.groupby(level=[0, 1]).sum(); N4 = B.groupby(level=[0, 1]).count()
    own = np.array([(B.get((d, i, s), np.nan) if st in ("base", "ex") else np.nan) for d, i, s, st in zip(T.ds, T.id, T.seed, T.set)], float)
    tot = np.array([S4.get((d, i), np.nan) for d, i in zip(T.ds, T.id)]); cnt = np.array([N4.get((d, i), 0) for d, i in zip(T.ds, T.id)], float)
    T["Lc_base"] = np.where(~np.isnan(own), (tot - np.nan_to_num(own)) / (cnt - 1), tot / cnt)
    ref = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)]; C = {}
    for c in ["G_p2", "Q_pick", "Q_aes"]:
        C[c] = (float(ref[c].mean()), float(ref[c].std())); T["z" + c] = (T[c] - C[c][0]) / C[c][1]
    s = T.zQ_pick + T.zQ_aes; rs = s[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)]
    C["Q_sum(zpick+zaes)"] = (float(rs.mean()), float(rs.std())); T["zQ_sum"] = (s - C["Q_sum(zpick+zaes)"][0]) / C["Q_sum(zpick+zaes)"][1]
    C["n_ref"] = len(ref); C["G_p2_nan_frac_ref"] = float(ref.G_p2.isna().mean())
    return T, C


def R_of(T, a, lf, q, lq):
    P = np.maximum(0, 1 - T.Lc / (a * T.Lc_base))
    return T["zG_p2"] - lf * P + lq * (T["zQ_" + q] if lq else 0), P


def grid():
    T, C = build(); T.to_pickle(f"{O10}/table10_data.pkl"); print("constants", C)
    X = T[(T.kind == "x") & (T.rseed == 0)]; M = T[(T.kind == "m") & (T.split == SPLIT)]; H = M[M.ds == "hypersim"].copy(); H["det"] = "dlsd"
    print("missing Q (needed rows):", int(T[(T.rseed == 0) & ((T.kind == "x") | (T.set == "base") | ((T.ds == "hypersim") & T.set.isin(["real", "real_rot", "real_rotctl"])))].Q_pick.isna().sum()))
    BT = M[(M.set == "base") & (M.rseed == 0)]
    wl = winlens("hypersim", sorted(H[H.set == "real"].id.unique())); rows = []
    for a, lf, q, lq in itertools.product(AL, LF, QS, LQ):
        if lf == 0 and a != AL[0]: continue
        if lq == 0 and q != QS[0]: continue
        e = exploit(X.assign(R=R_of(X, a, lf, q, lq)[0]), "R")
        Hr = H.assign(R=R_of(H, a, lf, q, lq)[0]); Mr = M.assign(R=R_of(M, a, lf, q, lq)[0]); ro = rot_new(Hr, "R", SPLIT, wl)
        Pb = R_of(BT, a, lf, q, lq)[1]
        worst = max(v[0] for v in e.values())
        r = dict(alpha=a if lf else np.nan, lf=lf, Q=q if lq else "-", lq=lq, worst_dR=worst, worst_pert=max(e, key=lambda k: e[k][0]),
                 n_sig=sum(v[2] for v in e.values()), n_pos=sum(v[0] > 0 for v in e.values()), pos=",".join(k for k, v in e.items() if v[0] > 0),
                 rot=ro["rot_new"], p_rot2=ro["p_rot2"], p_4lt2=ro["p_4lt2"], p_8lt2=ro["p_8lt2"], learn=learn(Hr, "R"),
                 corr_all=geo_corr(Mr, "R", "zG_p2"), corr_hyp=geo_corr(Hr, "R", "zG_p2"), floor_act=float((Pb > 0).mean()) if lf else 0.0,
                 floor_act_hyp=float((Pb[BT.ds == "hypersim"] > 0).mean()) if lf else 0.0)
        r["margin"] = bool(r["n_sig"] == 0 and worst <= 0)
        crit = dict(margin=r["margin"], rot=r["rot"], learn=r["learn"] >= 3, corr=r["corr_all"] >= 0.8 - 1e-9, floor=r["floor_act"] <= 0.20)
        r["fails"] = ",".join(k for k, ok in crit.items() if not ok); r["acceptable"] = r["fails"] == ""
        r["guard"] = "none" if lf == 0 and lq == 0 else "floor" if lq == 0 else "quality" if lf == 0 else "both"
        rows.append(r)
    S = pd.DataFrame(rows); S["wsum"] = S.lf + S.lq; S["qrank"] = S.Q.map({"pick": 0, "-": 0, "sum": 1}); S.to_csv(f"{O10}/grid10.csv", index=False)
    pd.set_option("display.width", 300, "display.max_rows", 500, "display.max_columns", 40)
    cols = ["alpha", "lf", "Q", "lq", "worst_dR", "worst_pert", "n_sig", "n_pos", "rot", "learn", "corr_all", "floor_act", "fails"]
    print(S.sort_values(["wsum", "worst_dR"])[cols].round(3).to_string(index=False))
    A = S[S.acceptable].sort_values(["wsum", "floor_act", "qrank", "corr_all"], ascending=[True, True, True, False])
    print("n acceptable:", len(A))
    for q in ["-", "pick", "sum"]:
        a = A[A.Q == q]; print("smallest acceptable Q=", q, "NONE" if a.empty else a.iloc[0][cols].to_dict())
    if A.empty:
        print("NO PICK. closest 3:"); print(S.sort_values("worst_dR").head(3)[cols + ["pos"]].round(4).to_string(index=False)); return
    pk = A.iloc[0]; print("PICK", pk.to_dict())
    a, lf, q, lq = (pk.alpha if pk.lf else 0.9), pk.lf, (pk.Q if pk.lq else "pick"), pk.lq
    R, P = R_of(X, a, lf, q, lq)
    Xc = X.assign(G=X.zG_p2, F=-lf * P, Qc=lq * X["zQ_" + q] if lq else 0.0, R=R)
    out = {c: exploit(Xc, c) for c in ["G", "F", "Qc", "R"]}
    D = pd.DataFrame({c: {k: f"{v[0]:+.3f}{'*' if v[2] else ''}" for k, v in out[c].items()} for c in out}); D["p_R_up"] = {k: v[1] for k, v in out["R"].items()}
    D.to_csv(f"{O10}/pick10_decomp.csv"); print(D.to_string())
    pd.Series({**{k: v for k, v in pk.to_dict().items()}, **{f"const_{k}": v for k, v in C.items()}}).to_json(f"{O10}/pick10.json", indent=1)


if __name__ == "__main__":
    {"sanity": sanity, "grid": grid}[sys.argv[1]]()
