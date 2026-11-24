"""Evaluate every detector x variant on each reference set. python evalv.py [SELECT|REPORT] ds1 ds2 ..."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, code_dir("task3"))
O = DATA + "/outputs/task3"
THETAS = [0.5, 1, 2, 4, 8]


def variants(df):
    v = [c for c in df.columns if c.startswith("s_")]
    return sorted(v)


def _wl(args):
    from pipe import load, lsd_segs
    from imageops import choose_window
    ds, i = args
    return choose_window(lsd_segs(load(ds, ("real", i, -1, 0.0))))[1]


def winlens(ds, ids):
    with ProcessPoolExecutor(64) as ex:
        return dict(zip(ids, ex.map(_wl, [(ds, i) for i in ids], chunksize=8)))


def evaluate(df, ds, det, v, split, winlen):
    d = df[(df.det == det) & (df.rseed == 0)]
    sub = d[d.split == split]
    if "horizon" in v:
        d = d.assign(s_horizon=-d.horizon_err_deg); sub = sub.assign(s_horizon=-sub.horizon_err_deg)
    real = sub[sub.set == "real"].set_index("id")
    base = sub[sub.set == "base"]
    out = dict(ds=ds, det=det, var=v, split=split)
    r = real[v]
    out["cov_real"] = float(r.notna().mean()); out["cov_gen"] = float(base[v].notna().mean())
    # paired win rate
    g = base.dropna(subset=[v]).groupby("id")[v].apply(np.array)
    ids = [i for i in r.dropna().index if i in g.index]
    w = np.array([np.mean((r[i] > g[i]) + 0.5 * (r[i] == g[i])) for i in ids])
    dd = np.array([r[i] - g[i].mean() for i in ids])
    out["n_pairs"] = len(ids)
    out["wr"] = float(w.mean()) if len(w) else np.nan
    out["p"] = float(wilcoxon(dd).pvalue) if len(dd) > 10 and np.any(dd != 0) else np.nan
    rr, gg = r.dropna().values, base[v].dropna().values
    out["auroc"] = float(roc_auc_score(np.r_[np.ones(len(rr)), np.zeros(len(gg))], np.r_[rr, gg])) if len(rr) and len(gg) else np.nan
    # degraded twins (diagnostic)
    gd = sub[sub.set == "base_deg"].dropna(subset=[v]).groupby("id")[v].apply(np.array)
    wd = [np.mean((r[i] > gd[i]) + 0.5 * (r[i] == gd[i])) for i in ids if i in gd.index]
    out["wr_deg"] = float(np.mean(wd)) if wd else np.nan
    # structure-matched: quintiles of pooled Lc (real + twins, this split/detector); compare real only with same-bin twins
    pool = pd.concat([real.reset_index()[["id", "Lc"]], base[["id", "Lc"]]])
    edges = np.quantile(pool.Lc, [0.2, 0.4, 0.6, 0.8])
    q = lambda x: int(np.searchsorted(edges, x))
    qw = {k: [] for k in range(5)}
    for i, gi in base.dropna(subset=[v]).groupby("id"):
        if i not in r.index or np.isnan(r[i]): continue
        qi = q(real.loc[i, "Lc"]); gs = gi[[q(x) == qi for x in gi.Lc]][v].values
        if len(gs): qw[qi].append(np.mean((r[i] > gs) + 0.5 * (r[i] == gs)))
    for k in range(5):
        out[f"wr_q{k+1}"] = float(np.mean(qw[k])) if qw[k] else np.nan; out[f"n_q{k+1}"] = len(qw[k])
    out["q_above"] = int(sum((out[f"wr_q{k+1}"] > 0.5) for k in range(5) if not np.isnan(out[f"wr_q{k+1}"])))
    # rotation test
    keep = [i for i in r.dropna().index if winlen.get(i, 0) >= 100]
    rot = sub[sub.set == "real_rot"]
    drops, ps = [], []
    for t in THETAS:
        rt = rot[rot.theta == t].set_index("id")[v].reindex(keep)
        x = (r.reindex(keep) - rt).dropna().values
        drops.append(float(x.mean()) if len(x) else np.nan)
        ps.append(float(wilcoxon(x, alternative="greater").pvalue) if len(x) > 10 and np.any(x != 0) else np.nan)
    ctl = (r.reindex(keep) - sub[sub.set == "real_rotctl"].set_index("id")[v].reindex(keep)).dropna().values
    out.update({f"drop{t}": x for t, x in zip(THETAS, drops)}, p_rot2=ps[2], drop_ctl=float(ctl.mean()) if len(ctl) else np.nan)
    out["rot_mono"] = bool(all(drops[j + 1] >= drops[j] for j in range(4)) and drops[-1] > drops[0])
    out["rot_pass"] = bool(out["rot_mono"] and ps[2] < 0.01)
    # learnability: caption std over 4 base seeds (median over SELECT captions) / RANSAC-seed std (mean over 200 imgs)
    bs = d[(d.split == "SELECT") & (d.set == "base")].groupby("id")[v].std(ddof=0).median()
    sd = df[(df.det == det) & (df.split == "SELECT") & (((df.set == "real") & (df.seed == -1)) | ((df.set == "base") & (df.seed == 0)))]
    if "horizon" in v: sd = sd.assign(s_horizon=-sd.horizon_err_deg)
    sd = sd[sd.id.isin(sd[sd.rseed > 0].id.unique())]
    rs = sd.groupby(["set", "id"])[v].std(ddof=0)
    out["cap_std"] = float(bs); out["ransac_std"] = float(rs.mean())
    out["learn"] = float(bs / max(rs.mean(), 1e-9)) if not np.isnan(bs) else np.nan
    # 5e
    pp = pd.concat([real.reset_index()[[v, "total_len"]], base[[v, "total_len"]]]).dropna()
    out["rho_len"] = float(spearmanr(pp[v], pp.total_len).statistic) if len(pp) > 10 else np.nan
    out["WORKS"] = bool(out["wr"] > 0.5 and out["p"] < 0.01 and out["q_above"] >= 4 and out["rot_pass"] and out["learn"] >= 3)
    return out


if __name__ == "__main__":
    split = sys.argv[1]; rows = []
    for ds in sys.argv[2:]:
        df = pd.read_pickle(f"{O}/scores_{ds}.pkl")
        wl = winlens(ds, sorted(df[df.set == "real"].id.unique()))
        V = variants(df) + ["s_horizon"]
        for det in sorted(df.det.unique()):
            for v in V:
                rows.append(evaluate(df, ds, det, v, split, wl))
    T = pd.DataFrame(rows); tag = "_".join(sys.argv[2:])
    T.to_csv(f"{O}/eval_{split}_{tag}.csv", index=False)
    pd.set_option("display.width", 260, "display.max_rows", 200, "display.max_columns", 30)
    print(T[["ds", "det", "var", "cov_real", "wr", "p", "auroc", "wr_deg", "wr_q1", "wr_q2", "wr_q3", "wr_q4", "wr_q5", "q_above", "drop2", "p_rot2", "rot_pass", "learn", "rho_len", "WORKS"]].round(3).to_string(index=False))
