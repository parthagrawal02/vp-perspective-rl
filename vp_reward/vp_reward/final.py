"""FROZEN perspective reward (Task 11, 2026-11-17). Do not change without explicit approval. Spec: vp_reward/FINAL.md.

R = z(G) - 4 * P + 0.25 * zQ
  G  : p2-sig continuous VP-consistency score on DeepLSD lines detected on the contrast-stretched image
  P  : max(0, 1 - Lc / (0.8 * Lc_base))        structure floor
  zQ : (z(PickScore) + z(aesthetic) - Q_SUM_MEAN) / Q_SUM_STD

Entry point for DiffusionNFT:  fn = perspective_reward(device, lc_base=...);  scores, info = fn(images, prompts, metadata)
"""
from __future__ import annotations

import numpy as np

from .v4 import stretch, compute_t8

# ---- frozen constants (Task 10: SELECT base twins, all 3 sets, RANSAC seed 0, n = 6800) ----
G_MEAN, G_STD = 0.6762756750467267, 0.12087393883907192
PICK_MEAN, PICK_STD = 22.55044470282162, 0.989595540429443          # PickScore logit (scorer output * 26)
AES_MEAN, AES_STD = 5.452365678899429, 0.3416999366620726
Q_SUM_MEAN, Q_SUM_STD = 5.684341886080802e-16, 1.5849401901583253   # of z(pick) + z(aes)
G_NAN_FILL = 0.37950405006078053   # 1st percentile of the same 6800 G values (3 NaN excluded)
ALPHA, LAMBDA_F, LAMBDA_Q = 0.8, 4.0, 0.25
LMIN_FRAC, SIGMA0, RANSAC_SEED = 0.04, 1.0, 0
from .paths import DEEPLSD_CKPT  # $VP_DEEPLSD_CKPT, default $VP_DATA/models/deeplsd/deeplsd_wireframe.tar (wireframe weights)
DEEPLSD_CONF = {"detect_lines": True, "line_detection_params": {"merge": False, "filtering": True, "grad_thresh": 3, "grad_nfa": True}}


def load_deeplsd(device="cuda"):
    import torch
    from deeplsd.models.deeplsd_inference import DeepLSD
    ck = torch.load(DEEPLSD_CKPT, map_location="cpu", weights_only=False)
    net = DeepLSD(DEEPLSD_CONF); net.load_state_dict(ck["model"])
    return net.to(device).eval()


def detect_lines(net, pre, device="cuda"):
    """pre: stretched grey float32 [0,255] (512x512). Returns (N,4) x1,y1,x2,y2 px. No line refiner / VP optimisation."""
    import torch
    x = torch.tensor(pre[None, None] / 255, dtype=torch.float32, device=device)
    with torch.no_grad():
        return net({"image": x})["lines"][0].reshape(-1, 4).astype(np.float32)


def geometry(pre, lines):
    """-> (G or nan, Lc). V2 fit on the stretched image (sigma0 = 1 px, sigma_sys = 0.1 px, z<3 inliers, 15 deg assignment)."""
    o = compute_t8(pre, lines, seed=RANSAC_SEED, lmin_frac=LMIN_FRAC, s0=SIGMA0)
    return float(o["C_p2_sig"]), float(o["Lc_v2"])


def _geo_job(a):
    import cv2; cv2.setNumThreads(1)
    return geometry(*a)


def floor_P(Lc, Lc_base):
    Lc, Lc_base = np.asarray(Lc, float), np.asarray(Lc_base, float)
    return np.where(Lc_base > 0, np.maximum(0.0, 1.0 - Lc / (ALPHA * np.where(Lc_base > 0, Lc_base, 1.0))), 0.0)


def combine(G, Lc, Lc_base, pick, aes):
    """Pure function of the raw parts -> (R, info). G may contain NaN (filled per the NaN rule)."""
    G = np.asarray(G, float); nan = np.isnan(G); Gf = np.where(nan, G_NAN_FILL, G)
    zG = (Gf - G_MEAN) / G_STD
    zP, zA = (np.asarray(pick, float) - PICK_MEAN) / PICK_STD, (np.asarray(aes, float) - AES_MEAN) / AES_STD
    zQ = (zP + zA - Q_SUM_MEAN) / Q_SUM_STD
    P = floor_P(Lc, Lc_base)
    R = zG - LAMBDA_F * P + LAMBDA_Q * zQ
    return R, dict(G=Gf, G_raw=G, G_nan=nan, zG=zG, Lc=np.asarray(Lc, float), Lc_base=np.asarray(Lc_base, float), P=P,
                   pickscore=np.asarray(pick, float), aesthetic=np.asarray(aes, float), zQ=zQ, R=R)


def to_uint8_nhwc(images):
    import torch
    if isinstance(images, torch.Tensor):  # NCHW float in [0,1] (DiffusionNFT convention)
        return (images * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    a = np.asarray(images)
    return a if a.dtype == np.uint8 else (a * 255).round().clip(0, 255).astype(np.uint8)


def geometry_batch(net, ims, device="cuda", pool=None):
    """uint8 NHWC 512x512 -> (G with NaN, Lc). Needs DeepLSD (deeplsd env)."""
    pres = [stretch(x) for x in ims]
    lines = [detect_lines(net, p, device) for p in pres]
    geo = list(pool.map(_geo_job, zip(pres, lines))) if pool else [geometry(p, l) for p, l in zip(pres, lines)]
    return np.array([g for g, _ in geo], float), np.array([l for _, l in geo], float)


def quality_batch(ps, ae, ims, prompts):
    """uint8 NHWC -> (PickScore logit, aesthetic). Needs flow_grpo scorers (nft env)."""
    import torch
    from PIL import Image
    with torch.no_grad():
        pick = (ps(list(prompts), [Image.fromarray(x) for x in ims]) * 26).float().cpu().numpy()
        aes = ae(torch.tensor(ims).permute(0, 3, 1, 2)).float().cpu().numpy()
    return pick.astype(float), aes.astype(float)


def perspective_reward(device="cuda", lc_base=None, n_workers=16):
    """DiffusionNFT-style reward fn. lc_base: dict prompt -> Lc_base (mean Lc of base SD3.5 samples of that prompt),
    or None to require metadata[k]['lc_base'] per sample. Images must be 512x512 RGB."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from flow_grpo.pickscore_scorer import PickScoreScorer
    from flow_grpo.aesthetic_scorer import AestheticScorer
    net = load_deeplsd(device)
    ps = PickScoreScorer(device=device, dtype=torch.float32); ae = AestheticScorer(dtype=torch.float32, device=device)
    pool = ProcessPoolExecutor(n_workers) if n_workers > 1 else None

    def _fn(images, prompts, metadata=None):
        ims = to_uint8_nhwc(images)
        assert ims.shape[1:3] == (512, 512), f"reward is defined at 512x512, got {ims.shape}"
        G, Lc = geometry_batch(net, ims, device, pool)
        if lc_base is not None:
            Lb = np.array([lc_base[p] for p in prompts], float)
        else:
            Lb = np.array([m["lc_base"] for m in metadata], float)
        pick, aes = quality_batch(ps, ae, ims, prompts)
        R, info = combine(G, Lc, Lb, pick, aes)
        return R.astype(np.float32), info

    return _fn
