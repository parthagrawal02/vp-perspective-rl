"""5a-5e for all 36 candidates (4 scores x 9 configs), selection on SELECT, same numbers on REPORT."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score

OUT = DATA + "/outputs/task2"
CANDS = ["G", "NM", "RegMean", "RegMax"]
THETAS = [0.5, 1, 2, 4, 8]
MIN_WIN_LEN = 100.0  # px of segment length inside the rotated window ("region containing lines")

df = pd.read_pickle(f"{OUT}/scores.pkl")
main = df[df.rseed == 0]
extra = __import__("pickle").load(open(f"{OUT}/segcache.pkl", "rb"))["extra"]
winlen = {k[1]: v["win_len"] for k, v in extra.items() if k[0] == "real_rot" and k[3] == 2}


def lref(cfg):
    r = main[(main.set == "real") & (main.cfg == cfg) & (main.split == "SELECT")]
    return float(np.percentile(r.Lc, 25))


def winrate(real, gen):
    """real: Series indexed by id; gen: DataFrame with id, value col 'v'. Returns per-id win, diff."""
    g = gen.groupby("id")["v"].apply(np.array)
    ids = real.index.intersection(g.index)
    w = np.array([np.mean((real[i] > g[i]) + 0.5 * (real[i] == g[i])) for i in ids])
    d = np.array([real[i] - g[i].mean() for i in ids])
    return w, d, ids


def paired_block(sub, k, gen_set):
    r = sub[sub.set == "real"].set_index("id")[k]
    g = sub[sub.set == gen_set][["id", k]].rename(columns={k: "v"})
    w, d, ids = winrate(r, g)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    auc = roc_auc_score(np.r_[np.ones(len(r)), np.zeros(len(g))], np.r_[r.values, g.v.values])
    return dict(wr=float(w.mean()), wilcoxon_p=float(p), median_diff=float(np.median(d)), auroc=float(auc), n=len(ids))


def candidate_stats(cfg, k, split, gated=False):
    sub = main[(main.cfg == cfg) & (main.split == split)].copy()
    if gated:
        L = lref(cfg)
        sub[k] = sub[k] * np.minimum(1.0, sub.Lc / L)
    out = {}
    out["a"] = paired_block(sub, k, "base")
    out["b"] = paired_block(sub, k, "base_deg")
    # 5c rotation
    r0 = sub[sub.set == "real"].set_index("id")[k]
    keep = [i for i in r0.index if winlen.get(i, 0) >= MIN_WIN_LEN]
    drops, pvals = {}, {}
    rot = sub[sub.set == "real_rot"]
    for t in THETAS:
        rt = rot[rot.theta == t].set_index("id")[k].reindex(keep)
        dd = (r0.reindex(keep) - rt).values
        drops[t] = float(dd.mean())
        pvals[t] = float(wilcoxon(dd, alternative="greater").pvalue) if np.any(dd != 0) else 1.0
    ctl = sub[sub.set == "real_rotctl"].set_index("id")[k].reindex(keep)
    out["c"] = dict(drop=drops, p=pvals, ctl_drop=float((r0.reindex(keep) - ctl).values.mean()), n=len(keep),
                    monotone=bool(all(drops[THETAS[j + 1]] >= drops[THETAS[j]] for j in range(len(THETAS) - 1)) and drops[8] > drops[0.5]))
    # 5d learnability
    b = sub[sub.set == "base"]
    cap_std = float(b.groupby("id")[k].std(ddof=0).median())
    sd = df[(df.cfg == cfg) & df.set.str.startswith("seedstd")]
    if gated:
        sd = sd.copy(); sd[k] = sd[k] * np.minimum(1.0, sd.Lc / lref(cfg))
    rs = sd.groupby(["set", "id"])[k].std(ddof=0)
    out["d"] = dict(caption_std_median=cap_std, ransac_std_mean=float(rs.mean()), ransac_std_median=float(rs.median()),
                    learnability=float(cap_std / max(rs.mean(), 1e-9)))
    # 5e
    pool = sub[sub.set.isin(["real", "base"])]
    out["e"] = dict(rho_len_pooled=float(spearmanr(pool[k], pool.total_len).statistic),
                    rho_len_real=float(spearmanr(sub[sub.set == "real"][k], sub[sub.set == "real"].total_len).statistic))
    return out


def eligible(s):
    a, b, c, d = s["a"], s["b"], s["c"], s["d"]
    fails = []
    if not c["monotone"]: fails.append("rot-not-monotone")
    if not c["p"][2] < 0.01: fails.append(f"rot2deg p={c['p'][2]:.2g}")
    if not d["learnability"] >= 3: fails.append(f"learn={d['learnability']:.2f}")
    if not (a["wr"] > 0.5 and (b["wr"] - 0.5) >= 0.5 * (a["wr"] - 0.5)): fails.append(f"deg-margin({a['wr']:.3f}->{b['wr']:.3f})")
    return not fails, fails


if __name__ == "__main__":
    cfgs = sorted(main.cfg.unique())
    res = {}
    for split in ("SELECT", "REPORT"):
        for cfg in cfgs:
            for k in CANDS:
                res[(split, cfg, k)] = candidate_stats(cfg, k, split)
    rows = []
    for (split, cfg, k), s in res.items():
        el, fails = eligible(s)
        rows.append(dict(split=split, cfg=cfg, cand=k, wr=s["a"]["wr"], wilc_p=s["a"]["wilcoxon_p"], auroc=s["a"]["auroc"],
                         wr_deg=s["b"]["wr"], wilc_p_deg=s["b"]["wilcoxon_p"], auroc_deg=s["b"]["auroc"],
                         **{f"drop{t}": s["c"]["drop"][t] for t in THETAS}, p_rot2=s["c"]["p"][2], drop_ctl=s["c"]["ctl_drop"], n_rot=s["c"]["n"],
                         monotone=s["c"]["monotone"], learn=s["d"]["learnability"], cap_std=s["d"]["caption_std_median"],
                         ransac_std=s["d"]["ransac_std_mean"], rho_len=s["e"]["rho_len_pooled"], rho_len_real=s["e"]["rho_len_real"],
                         eligible=el, fails="; ".join(fails)))
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT}/selection_table.csv", index=False)
    sel = T[(T.split == "SELECT") & T.eligible].sort_values("wr_deg", ascending=False)
    pick = None if sel.empty else sel.iloc[0][["cfg", "cand"]].to_dict()
    json.dump(dict(pick=pick, lref={c: lref(c) for c in cfgs}), open(f"{OUT}/selection.json", "w"), indent=1)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
    cols = ["cfg", "cand", "wr", "wilc_p", "auroc", "wr_deg", "auroc_deg", "drop0.5", "drop2", "drop8", "p_rot2", "drop_ctl", "monotone", "learn", "rho_len", "eligible", "fails"]
    print(T[T.split == "SELECT"][cols].round(4).to_string(index=False))
    print("PICK:", pick)
    if pick:
        g = candidate_stats(pick["cfg"], pick["cand"], "REPORT", gated=True)
        print("REPORT, chosen (gated):", json.dumps(g, indent=0, default=float)[:1500])
        json.dump(g, open(f"{OUT}/chosen_report_gated.json", "w"), indent=1, default=float)
