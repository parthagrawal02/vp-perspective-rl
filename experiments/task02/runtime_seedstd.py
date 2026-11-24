from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, time, json, numpy as np

from PIL import Image
from vp_reward import VPConfig, center_square_resize, detect_segments, fit_and_score
meta = json.load(open(DATA + "/datasets/nyu/meta.json"))["items"]
ids = [m["id"] for m in meta][::15][:100]
imgs = [np.array(Image.open(DATA + f"/datasets/nyu/rgb512/{i:04d}.png")) for i in ids]
cfg = VPConfig()
t = time.perf_counter(); segs = [detect_segments(center_square_resize(im)) for im in imgs]; t_lsd = (time.perf_counter() - t) / len(imgs)
t = time.perf_counter(); R = [fit_and_score(s, cfg, 0) for s in segs]; t_fit = (time.perf_counter() - t) / len(imgs)
print(f"single core: LSD {t_lsd*1000:.1f} ms/img, fit+score {t_fit*1000:.1f} ms/img, total {(t_lsd+t_fit)*1000:.1f} ms/img")
print("n_segs (L>=4%) median", np.median([r.diag["n_segs"] for r in R]), " n_vps distribution", np.bincount([len(r.diag["vps"]) for r in R]))
keys = ["G", "NM", "RegMean", "RegMax"]
S = {k: np.array([[fit_and_score(s, cfg, sd).scores[k] for sd in range(5)] for s in segs]) for k in keys}
for k in keys: print(f"{k:8s} RANSAC-seed std (5 seeds) per image: median {np.median(S[k].std(1)):.4f} mean {S[k].std(1).mean():.4f}; score mean {S[k].mean():.3f}")
