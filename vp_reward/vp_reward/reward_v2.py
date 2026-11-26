"""EXPERIMENTAL v2 perspective reward (NOT frozen; the frozen reward is final.py / FINAL.md). Same pipeline, fit and G as
final.py (stretch -> DeepLSD -> refine -> V2 z-fit, RANSAC seed 0), plus an upper structure cap (2.). Default BETA = 0, i.e.
v2 = frozen G + cap. The monotone penalty (1.) is kept as an option but REJECTED by Task 15: it lowers the reference-vs-twin win
rate at every BETA > 0 (SELECT 0.621 -> 0.592 at 0.5 -> 0.569 at 1.0; SUN falls below 0.5) and barely changes the synthetic
bend sweep, whose recovery at large angles is mostly detection loss, not the 15 deg gate. See experiments/task15/NOTES.md.

1. (option, off) Monotone line penalty.  Frozen G drops segments > 15 deg from every VP, so a line that is very wrong costs nothing while a
   slightly wrong one costs a lot (G vs. bend angle bottoms out near 8-16 deg and recovers by 45 deg). v2 counts each kept
   segment that is unassigned (or every segment, when no VP is found) as a saturated violator with weight BETA * w:
       G_b = 1 - (sum_asg w v(z) + BETA sum_unasg w) / (sum_asg w + BETA sum_unasg w),   w = (L/100)^2
   BETA = 0 reproduces the frozen G exactly.
2. Upper structure cap.  The floor P only punishes too little assigned structure; nothing stopped "densification" (adding
   extra lines toward the VPs raises Lc and usually G). v2 adds
       P_hi = min(1, max(0, Lc / (GAMMA * Lc_base) - 1))
   GAMMA = 1.75 is active on 2.9% of base twins (1.0% of references) of vp_pairs_v1 (twins: leave-one-out Lc_base).

R_v2 = z(G) - LAMBDA_F * P - LAMBDA_HI * P_hi + LAMBDA_Q * zQ      (with BETA = 0, z(G) is exactly the frozen z(G))

Undefined-G rule as in final.py (fill + Lc = 0 -> P = 1).
"""
from __future__ import annotations

import numpy as np

from . import final as F
from . import v2
from .core import ang_err
from .v4 import refine_both, sth_fixed, stretch

BETA = 0.0
GAMMA = 1.75
LAMBDA_HI = 4.0
# (mean, std, nan_fill) of G_b per BETA. BETA=0: the frozen constants (PNG twins, n = 6,800). Others: Task 15, JPEG copies of
# 960 SELECT base twins (provisional; the JPEG BETA=0 values are 0.6743 / 0.1233 / 0.3902 vs the frozen 0.6763 / 0.1209 / 0.3795).
G_STATS = {0.0: (F.G_MEAN, F.G_STD, F.G_NAN_FILL),
           0.25: (0.6614468044274336, 0.1250017598199874, 0.35549967217995104),
           0.5: (0.6501122603090724, 0.1276843659337315, 0.32217250766426136),
           1.0: (0.6303660568426753, 0.1340437520010989, 0.29273394554118565)}


def fit(pre, lines):
    """Replays compute_t8's fit (FINAL.md section 2). Returns per-segment arrays: L, z, asg (assigned < 15 deg), n_vps."""
    _, _, xyu, _, _, _ = refine_both(pre, np.asarray(lines, np.float64).reshape(-1, 4), F.LMIN_FRAC * pre.shape[0])
    out = dict(L=np.zeros(0), z=np.zeros(0), asg=np.zeros(0, bool), n_vps=0)
    if len(xyu) < 2:
        out["L"] = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1]) if len(xyu) else np.zeros(0)
        return out
    L = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1])
    sth = sth_fixed(L, F.SIGMA0)
    v2._SYS[0] = v2.SIG_SYS
    sg, Lpx, vps = v2.fit_z(xyu, sth, F.RANSAC_SEED)
    out.update(L=L, n_vps=len(vps))
    if vps:
        E = np.atleast_2d(ang_err(sg.m, sg.d, np.stack(vps)))
        best = E.min(0)
        out.update(z=v2.z_of(best, Lpx, sth), asg=best < v2.ASSIGN_DEG)
    else:
        out.update(z=np.full(len(L), np.inf), asg=np.zeros(len(L), bool))
    return out


def _v(z):
    return 1.0 / (1.0 + np.exp(-np.clip((z - 3.0) / 0.5, -50, 50)))


def score(f, beta=BETA):
    """-> (G_b or nan, Lc). beta=0 -> frozen (C_p2_sig, Lc_v2)."""
    L, z, asg = f["L"], f["z"], f["asg"]
    if f["n_vps"] == 0 or not asg.any():
        return np.nan, 0.0
    w = (L / 100.0) ** 2
    num = (w[asg] * _v(z[asg])).sum() + beta * w[~asg].sum()
    den = w[asg].sum() + beta * w[~asg].sum()
    return float(1 - num / den), float(L[asg].sum())


def geometry(pre, lines, beta=BETA):
    return score(fit(pre, lines), beta)


def _geo_job(a):
    import cv2
    cv2.setNumThreads(1)
    return geometry(*a)


def cap_P(Lc, Lc_base, gamma=GAMMA):
    Lc, Lb = np.asarray(Lc, float), np.asarray(Lc_base, float)
    return np.where(Lb > 0, np.clip(Lc / (gamma * np.where(Lb > 0, Lb, 1.0)) - 1.0, 0.0, 1.0), 0.0)


def combine(G, Lc, Lc_base, pick, aes, beta=BETA, gamma=GAMMA, lambda_hi=LAMBDA_HI):
    """Pure function -> (R_v2, info). Uses G_STATS[beta] for z(G_b)."""
    if beta not in G_STATS:
        raise KeyError(f"no standardization constants for beta={beta}; known: {sorted(G_STATS)}")
    mu, sd, fill = G_STATS[beta]
    G = np.asarray(G, float); nan = np.isnan(G); Gf = np.where(nan, fill, G)
    zG = (Gf - mu) / sd
    zP, zA = (np.asarray(pick, float) - F.PICK_MEAN) / F.PICK_STD, (np.asarray(aes, float) - F.AES_MEAN) / F.AES_STD
    zQ = (zP + zA - F.Q_SUM_MEAN) / F.Q_SUM_STD
    P = F.floor_P(Lc, Lc_base)
    Ph = cap_P(Lc, Lc_base, gamma)
    R = zG - F.LAMBDA_F * P - lambda_hi * Ph + F.LAMBDA_Q * zQ
    return R, dict(G=Gf, G_raw=G, G_nan=nan, zG=zG, Lc=np.asarray(Lc, float), Lc_base=np.asarray(Lc_base, float), P=P, P_hi=Ph,
                   pickscore=np.asarray(pick, float), aesthetic=np.asarray(aes, float), zQ=zQ, R=R)


def geometry_batch(net, ims, device="cuda", pool=None, beta=BETA):
    pres = [stretch(x) for x in ims]
    lines = [F.detect_lines(net, p, device) for p in pres]
    args = [(p, l, beta) for p, l in zip(pres, lines)]
    geo = list(pool.map(_geo_job, args)) if pool else [geometry(*a) for a in args]
    return np.array([g for g, _ in geo], float), np.array([l for _, l in geo], float)


def perspective_reward(device="cuda", lc_base=None, n_workers=16, beta=BETA, gamma=GAMMA, lambda_hi=LAMBDA_HI):
    """Drop-in for final.perspective_reward (same signature and info keys, plus P_hi)."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from flow_grpo.pickscore_scorer import PickScoreScorer
    from flow_grpo.aesthetic_scorer import AestheticScorer
    net = F.load_deeplsd(device)
    ps = PickScoreScorer(device=device, dtype=torch.float32); ae = AestheticScorer(dtype=torch.float32, device=device)
    pool = ProcessPoolExecutor(n_workers) if n_workers > 1 else None

    def _fn(images, prompts, metadata=None):
        ims = F.to_uint8_nhwc(images)
        assert ims.shape[1:3] == (512, 512), f"reward is defined at 512x512, got {ims.shape}"
        G, Lc = geometry_batch(net, ims, device, pool, beta)
        Lb = np.array([lc_base[p] for p in prompts] if lc_base is not None else [m["lc_base"] for m in metadata], float)
        pick, aes = F.quality_batch(ps, ae, ims, prompts)
        R, info = combine(G, Lc, Lb, pick, aes, beta, gamma, lambda_hi)
        return R.astype(np.float32), info

    return _fn
