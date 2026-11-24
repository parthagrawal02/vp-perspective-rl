"""Task 7 diagnostics. python diag7.py  -> outputs/task7/{ties.csv, sens.csv, sens_rows.pkl}"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, pickle, numpy as np, pandas as pd
for p in ["task2", "task3", "task4", "task6"]: sys.path.insert(0, code_dir(p))

O = DATA + "/outputs"; OUT = f"{O}/task7"; os.makedirs(OUT, exist_ok=True)
DS = ["nyu", "sun_kv2", "hypersim"]; TH = [0.5, 1.0, 2.0, 4.0, 8.0]
SC = {"adopted": "K10 s1 frac @sf1.5", "k10_nopre": "K10 s1 frac (no preproc, T5)", "V0": "A1 / V0 (T3)"}


def saved(ds):
    """real + base (rseed 0, both splits) for the three scores."""
    from eval6 import main_df
    R = pd.read_pickle(f"{O}/task6/scores6_raw.pkl")
    a = main_df(R[R.sf == 1.5], ds)[["set", "id", "seed", "theta", "rseed", "split", "K10_sig1.0_frac@sf1.5"]].rename(columns={"K10_sig1.0_frac@sf1.5": "adopted"})
    b = pd.read_pickle(f"{O}/task5/scores5_{ds}.pkl")[["set", "id", "seed", "theta", "rseed", "K10_sig1.0_frac"]].rename(columns={"K10_sig1.0_frac": "k10_nopre"})
    c = pd.read_pickle(f"{O}/task4/scores4_{ds}.pkl")[["set", "id", "seed", "theta", "rseed", "V0"]]
    return a.merge(b).merge(c)


def ties(D, ds):
    D = D[D.rseed == 0]; rows = []
    for v in SC:
        r = D[D.set == "real"].set_index("id")[v]; g = D[D.set == "base"]
        P = g[["id", v]].rename(columns={v: "g"}).assign(r=lambda x: x.id.map(r)).dropna()
        w, t, l = (P.r > P.g).mean(), (P.r == P.g).mean(), (P.r < P.g).mean()
        rows.append(dict(score=SC[v], ds=ds, n_pairs=len(P), win=w, tie=t, loss=l, wr_half=w + t / 2, wr_untied=w / (w + l),
                         real_at_max=float((r.dropna() >= 1 - 1e-9).mean()), twin_at_max=float((g[v].dropna() >= 1 - 1e-9).mean())))
    return rows


def scored_mask_k10(xy, sth, sys_px, seed=0, k=10):
    from vp_reward import v2
    v2._SYS[0] = sys_px; s, ctx = v2.a1_score(xy, sth, seed); v2._SYS[0] = v2.SIG_SYS
    if not ctx: return np.nan, np.zeros(len(xy), bool)
    L = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]); m = np.zeros(len(xy), bool); fr = []
    for c in range(len(ctx["vps"])):
        idx = np.where(ctx["asg"] & (ctx["arg"] == c))[0]
        if len(idx) >= k:
            top = idx[np.argsort(-L[idx])[:k]]; m[top] = True; fr.append((ctx["z"][top] > 3).mean())
    return (1 - float(np.mean(fr)) if fr else np.nan), m


def _job(a):
    import cv2; cv2.setNumThreads(1)
    from pipe import load, lsd_segs
    from imageops import choose_window, window_mask
    from vp_reward import v2
    from vp_reward.v4 import refine_both, sth_fixed, preprocess
    ds, i, lines = a  # lines[(name, theta)] ; theta 0 = original
    img0 = load(ds, ("real", i, -1, 0.0)); (x0, y0), wl = choose_window(lsd_segs(img0)); W = window_mask(x0, y0)
    out = []
    for th in [0.0] + TH:
        img = img0 if th == 0 else load(ds, ("real_rot", i, -1, th))
        for v in SC:
            src = preprocess(img, 1.5) if v == "adopted" else img
            xy, sth, xyu, _, _, _ = refine_both(src, lines[(v, th)], 0.04 * 512)
            if v == "V0":
                v2._SYS[0] = v2.SIG_SYS; s, ctx = v2.a1_score(xy, sth, 0)
                seg, m = xy, (ctx["asg"] if ctx else np.zeros(len(xy), bool))
            else:
                Lu = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1]); seg = xyu
                s, m = scored_mask_k10(xyu, sth_fixed(Lu, 1.0), v2.SIG_SYS)
            mid = (seg[:, :2] + seg[:, 2:]) / 2 if len(seg) else np.zeros((0, 2))
            inwin = W[np.clip(mid[:, 1].astype(int), 0, 511), np.clip(mid[:, 0].astype(int), 0, 511)] > 0.5 if len(mid) else np.zeros(0, bool)
            out.append(dict(ds=ds, id=i, theta=th, score=v, s=s, looks=bool((m & inwin).any()), winlen=wl))
    return out


def sens_rows():
    from concurrent.futures import ProcessPoolExecutor
    D6 = {}
    for f in glob.glob(f"{O}/task6/dlsd6_*.pkl"):
        D6.update({k[3]: v for k, v in pickle.load(open(f, "rb")).items() if k[0] == 1.5 and k[1] == "m" and k[3][0] in ("real", "real_rot")} if False else
                  {(k[2],) + k[3]: v for k, v in pickle.load(open(f, "rb")).items() if k[0] == 1.5 and k[1] == "m" and k[3][0] in ("real", "real_rot")})
    T = []
    for ds in DS:
        D3 = {}
        for f in glob.glob(f"{O}/task3/dlsd/{ds}_*.pkl"): D3.update({k: v for k, v in pickle.load(open(f, "rb")).items() if k[0] in ("real", "real_rot")})
        ids = sorted({k[1] for k in D3 if k[0] == "real"})
        for i in ids:
            L = {}
            for th in [0.0] + TH:
                key = ("real", i, -1, 0.0) if th == 0 else ("real_rot", i, -1, th)
                L[("adopted", th)] = D6[(ds,) + key]; L[("k10_nopre", th)] = D3[key]; L[("V0", th)] = D3[key]
            T.append((ds, i, L))
    with ProcessPoolExecutor(80) as ex: rows = [r for rr in ex.map(_job, T, chunksize=4) for r in rr]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    rows, SV = [], {}
    for ds in DS:
        SV[ds] = saved(ds); rows += ties(SV[ds], ds)
    TT = pd.DataFrame(rows); TT.to_csv(f"{OUT}/ties.csv", index=False); print(TT.round(3).to_string(index=False))
    S = sens_rows(); S.to_pickle(f"{OUT}/sens_rows.pkl")
    # consistency with saved scores
    for ds in DS:
        sv = SV[ds][(SV[ds].rseed == 0) & SV[ds].set.isin(["real", "real_rot"])]
        for v in SC:
            a = S[(S.ds == ds) & (S.score == v)].set_index(["id", "theta"]).s
            b = sv.assign(theta=sv.theta.where(sv.set == "real_rot", 0.0)).set_index(["id", "theta"])[v].reindex(a.index)
            print("consistency", ds, v, "max|diff|", float(np.nanmax(np.abs(a.values - b.values))), "nan-mismatch", int((a.isna() != b.isna()).sum()))
    out = []
    S = S[S.winlen >= 100]  # same image subset as the rotation test
    for (ds, v), g in S.groupby(["ds", "score"]):
        o = g[g.theta == 0].set_index("id").s
        for th in TH:
            r = g[g.theta == th].set_index("id"); d = (r.s - o.reindex(r.index)); ok = d.notna()
            lk = r.looks & ok
            out.append(dict(score=SC[v], ds=ds, theta=th, n=int(ok.sum()), lower=float((d[ok] < 0).mean()), tied=float((d[ok] == 0).mean()),
                            higher=float((d[ok] > 0).mean()), looks=float(r.looks[ok].mean()), lower_if_looks=float((d[lk] < 0).mean()) if lk.any() else np.nan))
    SS = pd.DataFrame(out); SS.to_csv(f"{OUT}/sens.csv", index=False); print(SS.round(3).to_string(index=False))
