"""Task 11 evaluation on the fresh Hypersim split.  python eval11.py
-> outputs/task11/{confirm11.csv, subsets11.csv, exploit11.csv, baseline11.csv, paired11.csv, table11_data.pkl}"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, glob, pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
for p in ["vp_reward", "task3", "task4", "task5", "task6", "task8", "task9", "task11"]: sys.path.insert(0, code_dir(p))
from evalv import evaluate
from eval4 import deg_p
from rot6 import rot_new
import eval8
from eval9 import exploit, learn, geo_corr
from vp_reward.final import combine

O = DATA + "/outputs/task11"; FR = DATA + "/datasets/hypersim_fresh"
eval8.SPLIT = "SELECT"


def build():
    import json
    G = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{O}/g11_*.pkl"))]); Q = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{O}/q11_*.pkl")])
    G["k"] = G.key.map(str); Q["k"] = Q.key.map(str)
    T = G.merge(Q[["k", "pickscore", "aesthetic"]], on="k", how="left")
    T["kind"] = T.key.map(lambda k: k[0])
    T["set"] = T.key.map(lambda k: k[1] if k[0] in ("m", "b") else "ex")
    T["id"] = T.key.map(lambda k: k[2] if k[0] in ("m", "b") else k[1])
    T["seed"] = T.key.map(lambda k: k[3] if k[0] in ("m", "b") else 0)
    T["theta"] = T.key.map(lambda k: k[4] if k[0] == "m" else 0.0)
    T["pert"] = T.key.map(lambda k: None if k[0] != "x" else (k[2] if k[2] == "orig" else f"{k[2]} {k[3]:g}"))
    sub = {m["id"]: m["subset"] for m in json.load(open(f"{FR}/meta.json"))["items"]}; T["subset"] = T.id.map(sub)
    # Lc_base: mean Lc of the caption's 4 base twins (rseed 0); leave-one-out when the image is one of them (base twins, exploit = seed-0 twin)
    B = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)].set_index(["id", "seed"]).Lc
    S4 = B.groupby(level=0).sum(); N4 = B.groupby(level=0).count()
    own = np.array([B.get((i, s), np.nan) if (st in ("base", "ex")) else np.nan for i, s, st in zip(T.id, T.seed, T.set)], float)
    tot = T.id.map(S4).values.astype(float); cnt = T.id.map(N4).values.astype(float)
    T["Lc_base"] = np.where(~np.isnan(own), (tot - np.nan_to_num(own)) / (cnt - 1), tot / cnt)
    R, info = combine(T.G.values, T.Lc.values, T.Lc_base.values, T.pickscore.values, T.aesthetic.values)
    T["Gf"] = info["G"]; T["G_nan"] = info["G_nan"]; T["zG"] = info["zG"]; T["P"] = info["P"]; T["zQ"] = info["zQ"]; T["R"] = R
    T["split"] = "SELECT"  # internal label only (evaluate()/learn() select on it); every row is from the fresh split
    T["det"] = "dlsd"; T["total_len"] = T.Lc
    return T


def g_confirm(M, wl, v="G"):
    r = evaluate(M, "hf", "dlsd", v, "SELECT", wl); r.update(rot_new(M, v, "SELECT", wl))
    r["wr_untied"], r["tie_frac"] = eval8.untied(M, v); r["p_deg"] = deg_p(M, v, "SELECT"); r.update(eval8.sens(M, v, wl))
    return r


def mean_se(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; return f"{x.mean():.4f} ± {x.std(ddof=1) / np.sqrt(len(x)):.4f}"


if __name__ == "__main__":
    T = build(); T.to_pickle(f"{O}/table11_data.pkl"); wl = pickle.load(open(f"{O}/winlen11.pkl", "rb"))
    M = T[T.kind == "m"].copy()
    print("rows", len(T), "missing Q (needed):", int(T[(T.rseed == 0) & ((T.kind != "m") | T.set.isin(["real", "base"]))].pickscore.isna().sum()),
          "G NaN frac (rseed0):", T[T.rseed == 0].groupby("set").G.apply(lambda x: round(float(x.isna().mean()), 4)).to_dict())
    # ---- 3a. G criteria (raw G with NaN, exactly as Task 3-10 evaluations) + R-side ----
    g = g_confirm(M, wl, "G")
    X = T[(T.kind == "x") & (T.rseed == 0)].assign(ds="hf"); e = exploit(X, "R")
    pd.DataFrame([dict(pert=k, mean_dR=v[0], p_up=v[1], sig_up=v[2]) for k, v in e.items()]).to_csv(f"{O}/exploit11.csv", index=False)
    eG = exploit(X, "zG"); eP = exploit(X.assign(F=-4 * X.P), "F"); eQ = exploit(X.assign(Qc=0.25 * X.zQ), "Qc")
    BT = M[(M.set == "base") & (M.rseed == 0)]
    MR = M.assign(ds="hf"); lR = learn(MR, "R"); cR = geo_corr(MR, "R", "zG"); fa = float((BT.P > 0).mean())
    worst = max(v[0] for v in e.values()); nsig = sum(v[2] for v in e.values())
    C = [("G: untied win rate > 0.5, p < 0.01", f"{g['wr_untied']:.3f} (ties {g['tie_frac']:.3f}), p = {g['p']:.2g}", g["wr_untied"] > .5 and g["p"] < .01),
         ("G: Lc quintiles with win rate > 0.5 (>= 4)", f"{g['q_above']} of 5: " + ", ".join(f"{g[f'wr_q{k}']:.2f}" for k in range(1, 6)), g["q_above"] >= 4),
         ("G: rotation rule", f"p_rot2 {g['p_rot2']:.2g}, p(4<2) {g['p_4lt2']:.2g}, p(8<2) {g['p_8lt2']:.2g}", g["rot_new"]),
         ("G: learnability >= 3", f"{g['learn']:.2f}", g["learn"] >= 3),
         ("G: degraded-twin win rate > 0.5, p < 0.01", f"{g['wr_deg']:.3f}, p = {g['p_deg']:.2g}", g["wr_deg"] > .5 and g["p_deg"] < .01),
         ("G: 2 deg rotations scored lower / higher (info)", f"{g['r2_lower']:.1%} / {g['r2_higher']:.1%} (tied {g['r2_tied']:.1%})", None),
         ("R: exploit margin (all mean dR <= 0, none sig. up)", f"worst mean dR {worst:+.4f}, n sig {nsig}", worst <= 0 and nsig == 0),
         ("R: learnability (info; >= 3 in Task 10)", f"{lR:.2f}", None),
         ("R: within-caption Spearman(R, G) (info; >= 0.8 in Task 10)", f"{cR:.2f}", None),
         ("R: floor activity on base twins (info; <= 23% agreed)", f"{fa:.1%}", None)]
    CF = pd.DataFrame(C, columns=["criterion", "value", "pass"]); CF.to_csv(f"{O}/confirm11.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120, "display.max_columns", 30); print(CF.to_string(index=False))
    print(pd.DataFrame({c: {k: f"{v[0]:+.3f}{'*' if v[2] else ''}" for k, v in ee.items()} for c, ee in (("zG", eG), ("-4P", eP), ("0.25zQ", eQ), ("R", e))}).assign(p_R=[f"{e[k][1]:.3g}" for k in eG]).to_string())
    print("G coverage twins", g["cov_gen"], "reals", g["cov_real"], "| rotation rule raw:", g["rot_new"], "| G vs R learn", round(g["learn"], 2), round(lR, 2))
    # subsets
    S = []
    for sb in ("newscene", "reportscene"):
        Ms = M[M.subset == sb]; ws = {i: wl[i] for i in Ms[Ms.set == "real"].id.unique()}; r = g_confirm(Ms, ws, "G")
        S.append(dict(subset=sb, n_img=Ms[Ms.set == "real"].id.nunique(), wr_untied=r["wr_untied"], p=r["p"], q_above=r["q_above"], rot=r["rot_new"], p_rot2=r["p_rot2"],
                      learn=r["learn"], wr_deg=r["wr_deg"], p_deg=r["p_deg"], r2_lower=r["r2_lower"], r2_higher=r["r2_higher"],
                      floor_act=float((Ms[(Ms.set == "base") & (Ms.rseed == 0)].P > 0).mean())))
    S = pd.DataFrame(S); S.to_csv(f"{O}/subsets11.csv", index=False); print(S.round(4).to_string(index=False))
    # ---- 4. baseline table (G NaN-filled per the frozen rule; Lc_base from the 4 base twins, leave-one-out for base twins) ----
    b0 = T[T.rseed == 0]
    real = b0[(b0.kind == "m") & (b0.set == "real")].set_index("id")
    base = b0[(b0.kind == "m") & (b0.set == "base")]
    rows = {"real Hypersim": real.reset_index(), "base SD3.5 CFG 4.5": base[base.seed <= 1], "base SD3.5 CFG 1.0": b0[(b0.kind == "b") & (b0.set == "base_cfg1.0")],
            "base + perspective suffix, CFG 4.5": b0[(b0.kind == "b") & (b0.set == "base_suffix_cfg4.5")],
            "best-of-4 by R (base CFG 4.5 twins)": base.loc[base.groupby("id").R.idxmax()],
            "Flow-GRPO PickScore LoRA, CFG 4.5": b0[(b0.kind == "b") & (b0.set == "flowgrpo_cfg4.5")],
            "DiffusionNFT multi-reward LoRA, CFG 1.0": b0[(b0.kind == "b") & (b0.set == "nft_cfg1.0")]}
    bref = base[base.seed <= 1].set_index(["id", "seed"]); edges = np.quantile(bref.Lc, [.2, .4, .6, .8]); TAB, PAIR = [], []
    for name, d in rows.items():
        t = dict(row=name, n=len(d), G=mean_se(d.Gf), G_nan=float(d.G_nan.mean()), Lc=mean_se(d.Lc), floor_act=float((d.P > 0).mean()),
                 PickScore=mean_se(d.pickscore), aesthetic=mean_se(d.aesthetic), R=mean_se(d.R))
        if name != "real Hypersim":
            w = d.groupby("id").Gf.apply(np.array); I = [i for i in w.index if i in real.index]
            t["real_vs_row_wr"] = float(np.mean([np.mean((real.Gf[i] > w[i]) + .5 * (real.Gf[i] == w[i])) for i in I]))
            t["p_real_vs_row"] = float(wilcoxon([real.Gf[i] - w[i].mean() for i in I]).pvalue)
        TAB.append(t)
        if name == "base SD3.5 CFG 4.5": continue
        # paired vs base CFG 4.5 by (caption, seed); real and best-of-4 (one image per caption) are paired with base seeds 0 and 1
        pr = []
        for r in d.itertuples():
            for s in ([0, 1] if name in ("real Hypersim", "best-of-4 by R (base CFG 4.5 twins)") else [r.seed]):
                if (r.id, s) in bref.index:
                    bb = bref.loc[(r.id, s)]; pr.append((r.Gf, bb.Gf, int(np.searchsorted(edges, bb.Lc)), r.Lc, bb.Lc))
        pr = np.array(pr); win = (pr[:, 0] > pr[:, 1]) + .5 * (pr[:, 0] == pr[:, 1])
        p = dict(row=name, n_pairs=len(pr), wr_vs_base=float(win.mean()), p=float(wilcoxon(pr[:, 0] - pr[:, 1]).pvalue), dLc=float((pr[:, 3] - pr[:, 4]).mean()))
        for q in range(5): p[f"wr_q{q+1}"] = float(win[pr[:, 2] == q].mean())
        PAIR.append(p)
    TB = pd.DataFrame(TAB); PR = pd.DataFrame(PAIR); TB.to_csv(f"{O}/baseline11.csv", index=False); PR.to_csv(f"{O}/paired11.csv", index=False)
    print(TB.to_string(index=False)); print(PR.round(3).to_string(index=False)); print("base-Lc quintile edges", edges.round(0))
