"""Task 2 figures (REPORT split): histograms before/after degradation, rotation curve, 6 example pairs."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json, pickle, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, code_dir("task2"))
from vp_reward import VPConfig, fit_and_score

OUT = DATA + "/outputs/task2"; FIG = f"{OUT}/figs"
import os; os.makedirs(FIG, exist_ok=True)
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
sel = json.load(open(f"{OUT}/selection.json")); pick = sel["pick"] or {"cfg": "L4_t1.5", "cand": "RegMean"}; PROV = sel["pick"] is None; cfg_tag, cand = pick["cfg"], pick["cand"]; LREF = sel["lref"][cfg_tag]
df = pd.read_pickle(f"{OUT}/scores.pkl"); m = df[(df.rseed == 0) & (df.cfg == cfg_tag) & (df.split == "REPORT")].copy()
m["gated"] = m[cand] * np.minimum(1, m.Lc / LREF)
NAMES = {"G": "G (inlier fraction)", "NM": "1 − near-miss fraction", "RegMean": "Regional exp(−R_mean/2°)", "RegMax": "Regional exp(−R_max/2°)"}

# ---- 1. histograms
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
bins = np.linspace(min(m[cand].min(), 0), m[cand].max(), 31)
for a, gset, title in [(ax[0], "base", "Before degradation"), (ax[1], "base_deg", "After sharpness-matched degradation")]:
    r, g = m[m.set == "real"][cand], m[m.set == gset][cand]
    a.hist(r, bins, density=True, color=SER[0], alpha=0.55, label=f"Real NYU (n={len(r)})", edgecolor=SURF, linewidth=1)
    a.hist(g, bins, density=True, color=SER[1], alpha=0.55, label=f"SD3.5-M twins (n={len(g)})", edgecolor=SURF, linewidth=1)
    a.axvline(r.median(), color=SER[0], lw=2); a.axvline(g.median(), color=SER[1], lw=2)
    a.set_title(title, loc="left", fontsize=11); a.set_xlabel(f"{NAMES[cand]}  [{cfg_tag}]")
ax[0].set_ylabel("density"); ax[0].legend(frameon=False, loc="upper left")
fig.suptitle(("PROVISIONAL (no candidate eligible) – " if PROV else "Chosen ") + "score, REPORT split: real photos vs caption-matched generations (vertical lines = medians)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/hist_real_vs_gen.png", dpi=150); plt.close(fig)

# ---- 2. rotation curve (all candidates at the chosen config), with 95% bootstrap CI
extra = pickle.load(open(f"{OUT}/segcache.pkl", "rb"))["extra"]
winlen = {k[1]: v["win_len"] for k, v in extra.items() if k[0] == "real_rot" and k[3] == 2}
TH = [0.5, 1, 2, 4, 8]; rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.4, 4))
for c, col in zip(["G", "NM", "RegMean", "RegMax"], SER):
    r0 = m[m.set == "real"].set_index("id")[c]; keep = [i for i in r0.index if winlen.get(i, 0) >= 100]
    mu, lo, hi = [], [], []
    for t in TH:
        d = (r0.reindex(keep) - m[(m.set == "real_rot") & (m.theta == t)].set_index("id")[c].reindex(keep)).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]
        mu.append(d.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    ctl = (r0.reindex(keep) - m[m.set == "real_rotctl"].set_index("id")[c].reindex(keep)).values.mean()
    lw = 2.6 if c == cand else 1.6
    ax.plot(TH, mu, "-o", color=col, lw=lw, ms=6, label=NAMES[c] + ((" (provisional)" if PROV else " (chosen)") if c == cand else ""), markeredgecolor=SURF, markeredgewidth=1.5)
    ax.fill_between(TH, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot([11], [ctl], marker="D", color=col, ms=6, markeredgecolor=SURF)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log", base=2); ax.set_xticks(TH + [11]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "ctl"])
ax.set_xlabel("rotation θ of a 256×256 feathered region (deg); ctl = +8° then −8°"); ax.set_ylabel("mean score drop (orig − rotated)")
ax.set_title(("PROVISIONAL – " if PROV else "") + f"Geometry check on real photos (REPORT, n={len(keep)}), config {cfg_tag}", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8.5, loc="upper left"); fig.tight_layout(); fig.savefig(f"{FIG}/rotation_curve.png", dpi=150); plt.close(fig)

# ---- 3. six example pairs, segments colored by LOO tile error
C = pickle.load(open(f"{OUT}/segcache.pkl", "rb"))["segs"]
L, T = cfg_tag.split("_"); cfg = VPConfig(lmin_frac=int(L[1:]) / 100, tau=float(T[1:]))
real = m[m.set == "real"].set_index("id")[cand]; twin = m[(m.set == "base") & (m.seed == 0)].set_index("id")[cand]
diff = (real - twin).dropna().sort_values()
picks = list(diff.index[-3:][::-1]) + [diff.index[len(diff) // 2], diff.index[len(diff) // 3]] + [diff.index[0]]
labels = ["real ≫ twin", "real ≫ twin", "real ≫ twin", "typical", "typical", "twin ≫ real"]
BINS = [(0, 1, SER[2], "<1°"), (1, 3, SER[3], "1–3°"), (3, 1e9, "#e34948", ">3°")]
def draw(a, img, key, title):
    r = fit_and_score(C[key], cfg, 0); d = r.diag
    a.imshow(img); a.axis("off")
    for (x1, y1, x2, y2), e in zip(d["seg_xy"], d["seg_tile_err"]):
        if np.isnan(e): a.plot([x1, x2], [y1, y2], color="#ffffff", lw=0.8, alpha=0.6); continue
        col = next(c for lo, hi, c, _ in BINS if lo <= e < hi); a.plot([x1, x2], [y1, y2], color=col, lw=1.8)
    a.set_title(f"{title}\n{cand}={r.scores[cand]:.3f}  R_mean={d['R_mean']:.2f}°  Lc={d['Lc']:.0f}", fontsize=8.5)
fig, ax = plt.subplots(6, 2, figsize=(7.2, 22))
for row, (i, lab) in enumerate(zip(picks, labels)):
    draw(ax[row, 0], np.array(Image.open(DATA + f"/datasets/nyu/rgb512/{i:04d}.png")), ("real", i, -1, 0.0), f"#{i} real ({lab})")
    draw(ax[row, 1], np.array(Image.open(f"{OUT}/gen/base_cfg4.5/{i:04d}_s0.png")), ("base", i, 0, 0.0), f"#{i} SD3.5-M twin, seed 0")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=c, lw=2.5) for *_, c, _ in BINS] + [Line2D([], [], color="#bbbbbb", lw=1)], [l for *_, l in BINS] + ["not in a scored tile"],
           loc="upper center", ncol=4, frameon=False, title="segment colour = leave-one-tile-out angular error", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(f"{FIG}/example_pairs.jpg", dpi=110); plt.close(fig)
print("FIGS in", FIG, "examples:", picks)
