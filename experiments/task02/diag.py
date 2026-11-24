from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, json, numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
sys.path.insert(0, code_dir("task2"))
from vp_reward import VPConfig, detect_segments, fit_and_score
from stats import main, winrate, lref
CFG = "L4_t1.5"
m = main[main.cfg == CFG]
def wr(sub, k, gset="base"):
    r = sub[sub.set == "real"].set_index("id")[k]; g = sub[sub.set == gset][["id", k]].rename(columns={k: "v"})
    return winrate(r, g)[0].mean()
print("== REPORT split, cfg", CFG)
for k in ["G", "NM", "RegMean", "RegMax"]:
    print(f"  {k:8s} wr(base)={wr(m[m.split=='REPORT'], k):.3f} wr(deg)={wr(m[m.split=='REPORT'], k, 'base_deg'):.3f}")
print("== medians real vs base vs base_deg (SELECT+REPORT)")
for c in ["n_segs", "total_len", "Lc", "n_vps", "R_mean", "R_max", "G", "RegMean"]:
    print(f"  {c:9s} real {m[m.set=='real'][c].median():8.3f}  base {m[m.set=='base'][c].median():8.3f}  deg {m[m.set=='base_deg'][c].median():8.3f}")
m2 = m.copy(); m2["unassigned_len"] = m2.total_len - m2.Lc; m2["frac_assigned"] = m2.Lc / m2.total_len
print(f"  frac of length assigned (<15deg): real {m2[m2.set=='real'].frac_assigned.median():.3f} base {m2[m2.set=='base'].frac_assigned.median():.3f}")
# structure-rich subset: both real and twin have Lc >= L_ref (pairs restricted per twin)
L = lref(CFG); sub = m[m.split == "SELECT"]
r = sub[sub.set == "real"].set_index("id"); g = sub[sub.set == "base"]
for k in ["G", "RegMean"]:
    ws = []
    for i, gi in g.groupby("id"):
        if r.loc[i, "Lc"] < L: continue
        gi = gi[gi.Lc >= L]
        if len(gi): ws.append(np.mean((r.loc[i, k] > gi[k]) + 0.5 * (r.loc[i, k] == gi[k])))
    print(f"== structure-rich pairs (Lc>=L_ref={L:.0f}) {k}: wr={np.mean(ws):.3f} over {len(ws)} reals")
# Lc-matched comparison: bin by Lc quintile (pooled), compare mean score within bin
pool = sub[sub.set.isin(["real", "base"])].copy(); pool["q"] = pd.qcut(pool.Lc, 5, labels=False)
print("== mean RegMean / G within Lc quintiles (real vs base):")
print(pool.groupby(["q", "set"])[["RegMean", "G", "R_mean"]].mean().unstack().round(3).to_string())
# sanity: would undistorting the (already undistorted?) NYU images raise their scores?
K = np.array([[518.857901, 0, 325.582449], [0, 519.469611, 253.736166], [0, 0, 1]])
D = np.array([2.0796615318809061e-01, -5.8613825163911781e-01, 7.2231363135888329e-04, 1.0479627195765181e-03, 4.9856986684705107e-01])
import h5py
def job(i):
    cv2.setNumThreads(1)
    f = h5py.File(DATA + "/datasets/nyu/raw/nyu_depth_v2_labeled.mat", "r")
    im = f["images"][i].transpose(2, 1, 0).copy()
    from vp_reward import center_square_resize
    out = {}
    for name, x in [("asis", im), ("undist", cv2.undistort(im, K, D))]:
        x = center_square_resize(x[16:464, 16:624])  # slightly larger crop to keep warped border out
        out[name] = fit_and_score(detect_segments(x), VPConfig(lmin_frac=0.04, tau=1.5), 0).scores
    return out
ids = list(range(0, 1449, 6))
with ProcessPoolExecutor(64) as ex: R = list(ex.map(job, ids))
for k in ["G", "RegMean", "RegMax"]:
    a = np.array([r["asis"][k] for r in R]); u = np.array([r["undist"][k] for r in R])
    print(f"== undistort-again test {k}: as-is {a.mean():.3f}  undistorted {u.mean():.3f}  (undist>asis in {np.mean(u>a)*100:.0f}%)")
