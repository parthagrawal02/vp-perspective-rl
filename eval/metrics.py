"""Validation metrics for a candidate reward score (the criteria of Tasks 3-11), collected from the task scripts
(experiments/task03/evalv.py, task04/eval4.py, task06/rot6.py, task08/eval8.py, task09/eval9.py) with the logic unchanged.

All functions take a long table with one row per (image, RANSAC seed):
    id      caption / reference id                    set    real | base | base_deg | real_rot | real_rotctl
    seed    twin seed (-1 for the real image)         rseed  RANSAC seed (0 = the reward; >0 only for learnability)
    theta   rotation in degrees (real_rot rows)       Lc     assigned segment length (structure amount)
    <v>     the score column being evaluated (e.g. G, R); NaN = undefined
'real' is the reference photo/render, 'base' its 4 caption-matched SD3.5-M twins (CFG 4.5, 40 steps, seeds 0-3), 'base_deg'
the twins degraded to the reference's sharpness, 'real_rot' the reference with a small in-plane rotation (theta in
{0.5,1,2,4,8} deg, crop window chosen so no border enters), 'real_rotctl' the same crop without rotation (control).
Win rates count ties as 1/2. Filter the table to one split / detector before calling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

THETAS = [0.5, 1, 2, 4, 8]


def _main(df):
    return df[df.rseed == 0]


def paired_win_rate(df, v):
    """Per reference: fraction of its twins it beats (ties = 1/2); mean over references. p: two-sided Wilcoxon on
    (reference score - mean twin score). -> dict(wr, p, n_pairs, cov_real, cov_gen)."""
    d = _main(df)
    real = d[d.set == "real"].set_index("id"); base = d[d.set == "base"]; r = real[v]
    g = base.dropna(subset=[v]).groupby("id")[v].apply(np.array)
    ids = [i for i in r.dropna().index if i in g.index]
    w = np.array([np.mean((r[i] > g[i]) + 0.5 * (r[i] == g[i])) for i in ids])
    dd = np.array([r[i] - g[i].mean() for i in ids])
    return dict(wr=float(w.mean()) if len(w) else np.nan, n_pairs=len(ids),
                p=float(wilcoxon(dd).pvalue) if len(dd) > 10 and np.any(dd != 0) else np.nan,
                cov_real=float(r.notna().mean()), cov_gen=float(base[v].notna().mean()))


def untied_win_rate(df, v):
    """Pooled over all (reference, twin) pairs: wins / (wins + losses), and the tie fraction. -> (wr_untied, tie_frac).
    This is the headline 'G win rate' (equal to paired_win_rate's wr when there are no ties)."""
    d = _main(df)
    r = d[d.set == "real"].set_index("id")[v]; g = d[d.set == "base"][["id", v]].dropna()
    rr = g.id.map(r); ok = rr.notna()
    w = (rr[ok] > g[v][ok]).sum(); l = (rr[ok] < g[v][ok]).sum(); t = (rr[ok] == g[v][ok]).sum()
    return float(w / (w + l)), float(t / ok.sum())


def lc_quintile_win_rates(df, v):
    """Structure-matched win rates: quintile edges of the pooled Lc (references + twins); each reference is compared only with
    its twins in the same Lc quintile. -> dict(wr_q1..wr_q5, n_q1..n_q5, q_above = #quintiles with wr > 0.5)."""
    d = _main(df)
    real = d[d.set == "real"].set_index("id"); base = d[d.set == "base"]; r = real[v]
    pool = pd.concat([real.reset_index()[["id", "Lc"]], base[["id", "Lc"]]])
    edges = np.quantile(pool.Lc, [0.2, 0.4, 0.6, 0.8])
    q = lambda x: int(np.searchsorted(edges, x))
    qw = {k: [] for k in range(5)}
    for i, gi in base.dropna(subset=[v]).groupby("id"):
        if i not in r.index or np.isnan(r[i]):
            continue
        qi = q(real.loc[i, "Lc"]); gs = gi[[q(x) == qi for x in gi.Lc]][v].values
        if len(gs):
            qw[qi].append(np.mean((r[i] > gs) + 0.5 * (r[i] == gs)))
    out = {}
    for k in range(5):
        out[f"wr_q{k+1}"] = float(np.mean(qw[k])) if qw[k] else np.nan; out[f"n_q{k+1}"] = len(qw[k])
    out["q_above"] = int(sum(out[f"wr_q{k+1}"] > 0.5 for k in range(5) if not np.isnan(out[f"wr_q{k+1}"])))
    out["edges"] = edges.tolist()
    return out


def _p(x, alt):
    x = x[~np.isnan(x)]
    return float(wilcoxon(x, alternative=alt).pvalue) if len(x) > 10 and np.any(x != 0) else np.nan


def rotation_test(df, v, winlen, min_window=100):
    """Rotating the reference must lower the score. Uses references whose rotation crop window is >= min_window px
    (winlen: id -> window length). Rule (Task 6): drop at 2 deg significant (one-sided Wilcoxon, p < 0.01) AND the drops at
    4 and 8 deg not significantly smaller than at 2 deg. Also returns mean drops per angle, the control-crop drop and the
    share of 2-deg rotations scored lower / tied / higher. -> dict(..., rot_pass)."""
    d = _main(df)
    r = d[d.set == "real"].set_index("id")[v]
    keep = [i for i in r.dropna().index if winlen.get(i, 0) >= min_window]
    rot = d[d.set == "real_rot"]
    dr = {t: (r.reindex(keep) - rot[rot.theta == t].set_index("id")[v].reindex(keep)).values for t in THETAS}
    out = {f"drop{t:g}": float(np.nanmean(dr[t])) if np.any(~np.isnan(dr[t])) else np.nan for t in THETAS}
    ctl = (r.reindex(keep) - d[d.set == "real_rotctl"].set_index("id")[v].reindex(keep)).dropna().values
    out["drop_ctl"] = float(ctl.mean()) if len(ctl) else np.nan
    out["p_rot2"] = _p(dr[2], "greater"); out["p_4lt2"] = _p(dr[4] - dr[2], "less"); out["p_8lt2"] = _p(dr[8] - dr[2], "less")
    out["rot_pass"] = bool(out["p_rot2"] < 0.01 and not (out["p_4lt2"] < 0.01) and not (out["p_8lt2"] < 0.01))
    x = -dr[2][~np.isnan(dr[2])]   # rotated - original
    out["r2_lower"], out["r2_tied"], out["r2_higher"] = float((x < 0).mean()), float((x == 0).mean()), float((x > 0).mean())
    out["n_rot"] = len(keep)
    return out


def degraded_twin_test(df, v):
    """Reference vs its twins degraded to the reference's sharpness (blur / resize / JPEG; tuned per dataset in Tasks 2-3):
    win rate (ties 1/2, as paired_win_rate) and two-sided Wilcoxon p on (reference - mean degraded twin). -> dict(wr_deg, p_deg)."""
    d = _main(df)
    r = d[d.set == "real"].set_index("id")[v].dropna()
    gd = d[d.set == "base_deg"].dropna(subset=[v]).groupby("id")[v].apply(np.array)
    wd = [np.mean((r[i] > gd[i]) + 0.5 * (r[i] == gd[i])) for i in r.index if i in gd.index]
    i = r.index.intersection(gd.index); x = (r[i] - gd[i].map(np.mean)).values.astype(float)
    return dict(wr_deg=float(np.mean(wd)) if wd else np.nan, p_deg=float(wilcoxon(x).pvalue))


def exploit_test(X, col, alpha=0.01):
    """Exploit / degradation-margin test. X: one row per (ds, id, pert) with pert == 'orig' for the unperturbed twin and e.g.
    'blur 2', 'noise 2', 'contrast 0.7', 'rescale 0.5', 'jpeg 50' for perturbed copies. A perturbation is an exploit if it
    raises the score on average with one-sided Wilcoxon p < alpha. -> {pert: (mean change, p_up, significant_up)}."""
    o = X[X.pert == "orig"].set_index(["ds", "id"])[col]; out = {}
    for pt, g in X[X.pert != "orig"].groupby("pert"):
        d = (g.set_index(["ds", "id"])[col] - o).dropna().values
        pv = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
        out[pt] = (float(d.mean()), float(pv), bool(d.mean() > 0 and pv < alpha))
    return out


def learnability(df, v):
    """Signal / noise of the score as an RL reward: median over captions of the std of the score across the caption's 4 base
    twins, divided by the mean (over images re-fit with RANSAC seeds > 0) of the std across RANSAC seeds. Criterion: >= 3."""
    d0 = df[(df.rseed == 0) & (df.set == "base")]
    bs = d0.groupby("id")[v].std(ddof=0).median()
    ids = df[df.rseed > 0].id.unique()
    sd = df[df.id.isin(ids) & (((df.set == "real") & (df.seed == -1)) | ((df.set == "base") & (df.seed == 0)))]
    return float(bs / max(sd.groupby(["set", "id"])[v].std(ddof=0).mean(), 1e-9))


def within_caption_spearman(df, v, ref):
    """Median over captions of Spearman(v, ref) across the caption's 4 base twins (e.g. R vs zG: does R rank like geometry?)."""
    d = _main(df); d = d[d.set == "base"]
    keys = ["ds", "id"] if "ds" in d.columns else ["id"]
    r = [spearmanr(g[v], g[ref]).statistic for _, g in d.groupby(keys) if len(g) == 4 and g[ref].nunique() > 1]
    return float(np.nanmedian(r))


def floor_activity(df):
    """Share of base twins (RANSAC seed 0) with a non-zero floor penalty P."""
    d = _main(df)
    return float((d[d.set == "base"].P > 0).mean())
