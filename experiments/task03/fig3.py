from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, glob, pickle, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, code_dir("task3"))
from vp_reward.v2 import refine_with_uncertainty, a1_score
O = DATA + "/outputs/task3"; F = f"{O}/figs"; import os; os.makedirs(F, exist_ok=True)
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"; SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 10})
R = pd.read_csv(f"{O}/eval_REPORT_nyu_sun_kv2_hypersim.csv")
TOP = [("dlsd", "s_A1", "DeepLSD · A1 noise-normalized (pick)"), ("dlsd", "s_A3_mean_z", "DeepLSD · A3 mean z"),
       ("lsd", "s_A2_k5_z", "LSD · A2 k=5, z"), ("dlsd", "s_RegMean", "DeepLSD · RegMean (Task 2 baseline)")]
NAMES = {"nyu": "NYU v2", "sun_kv2": "SUN RGB-D Kinect v2", "hypersim": "Hypersim"}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, ds in zip(axs, ["nyu", "sun_kv2", "hypersim"]):
    for (det, v, lab), c in zip(TOP, SER):
        r = R[(R.ds == ds) & (R.det == det) & (R["var"] == v)].iloc[0]
        y = [r[f"wr_q{k}"] for k in range(1, 6)]
        ax.plot(range(1, 6), y, "-o", color=c, lw=2.6 if v == "s_A1" and det == "dlsd" else 1.6, ms=6, markeredgecolor=SURF, markeredgewidth=1.5, label=lab)
    ax.axhline(0.5, color=INK2, lw=1, ls="--"); ax.set_xticks(range(1, 6)); ax.set_xticklabels(["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"])
    ax.set_title(NAMES[ds], loc="left", fontsize=11); ax.set_xlabel("structure quintile (Lc, pooled real+twins)")
axs[0].set_ylabel("real-photo win rate vs same-quintile twins"); axs[0].set_ylim(0.2, 1.0)
axs[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("Structure-matched paired win rate, REPORT split (0.5 = no separation; above = real photos score better)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{F}/winrate_by_quintile.png", dpi=150); plt.close(fig)

# ---- 6 example pairs, DeepLSD + A1, segment colour = z w.r.t. assigned VP
S = {ds: pd.read_pickle(f"{O}/scores_{ds}.pkl") for ds in ["nyu", "sun_kv2", "hypersim"]}
REAL = {"nyu": DATA + "/datasets/nyu/rgb512", "sun_kv2": DATA + "/datasets/sun_kv2/rgb512", "hypersim": DATA + "/datasets/hypersim/rgb512"}
GEN = {"nyu": DATA + "/outputs/task2/gen/base_cfg4.5", "sun_kv2": f"{O}/gen_sun_kv2_512/base_cfg4.5", "hypersim": f"{O}/gen_hypersim_512/base_cfg4.5"}
BINS = [(0, 3, SER[2], "z<3 (consistent)"), (3, 10, SER[3], "3–10"), (10, 1e9, "#e34948", "z>10")]
picks = []
for ds in ["nyu", "sun_kv2", "hypersim"]:
    d = S[ds]; d = d[(d.det == "dlsd") & (d.rseed == 0) & (d.split == "REPORT")]
    r = d[d.set == "real"].set_index("id").s_A1; g = d[(d.set == "base") & (d.seed == 0)].set_index("id").s_A1
    diff = (r - g).dropna().sort_values()
    picks += [(ds, diff.index[int(len(diff) * 0.85)], "real > twin (85th pct)"), (ds, diff.index[int(len(diff) * 0.5)], "median pair")]
DL = {}
for f in glob.glob(f"{O}/dlsd/*.pkl"):
    if "/C_" in f: continue
    ds = os.path.basename(f).rsplit("_", 1)[0]
    for k, v in pickle.load(open(f, "rb")).items():
        if k[0] in ("real", "base") and (k[0] == "real" or k[2] == 0): DL[(ds,) + k] = v
fig, ax = plt.subplots(6, 2, figsize=(7.4, 23))
def draw(a, img, seg, title):
    xy, sth, sd, sn = refine_with_uncertainty(img, seg, 0.04 * 512); s, c = a1_score(xy, sth, 0)
    a.imshow(img); a.axis("off")
    for (x1, y1, x2, y2), z, asg in zip(xy, c["z"], c["asg"]):
        if not asg: a.plot([x1, x2], [y1, y2], color="#ffffff", lw=0.7, alpha=0.6); continue
        col = next(cc for lo, hi, cc, _ in BINS if lo <= z < hi); a.plot([x1, x2], [y1, y2], color=col, lw=1.8)
    a.set_title(f"{title}\nA1={s:.3f}  segs={len(xy)}  median σθ={np.degrees(np.median(sth)):.2f}°", fontsize=8.5)
for row, (ds, i, lab) in enumerate(picks):
    draw(ax[row, 0], np.array(Image.open(f"{REAL[ds]}/{i:04d}.png").convert("RGB")), DL[(ds, "real", i, -1, 0.0)], f"{NAMES[ds]} #{i} real ({lab})")
    draw(ax[row, 1], np.array(Image.open(f"{GEN[ds]}/{i:04d}_s0.png").convert("RGB")), DL[(ds, "base", i, 0, 0.0)], f"SD3.5-M twin, seed 0")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=c, lw=2.5) for *_, c, _ in BINS] + [Line2D([], [], color="#bbbbbb", lw=1)], [l for *_, l in BINS] + ["unassigned (>15°)"],
           loc="upper center", ncol=4, frameon=False, fontsize=9, title="DeepLSD segments; colour = noise-normalized endpoint error z to assigned VP")
fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(f"{F}/example_pairs_A1.jpg", dpi=110); plt.close(fig)
print("figs:", F, picks)
