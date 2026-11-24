"""Task 4: blur-proof variants of A1 (DeepLSD + noise-normalized z fit).

V0  : A1 as in Task 3 (noise-propagated, contrast-weighted line fit, chi2-inflated sigma_theta, SIG_SYS 0.1 px).
V1  : V0 with SIG_SYS in {0.25, 0.5, 1.0} px.
V2  : fixed per-bin edge noise sigma0 in {0.25, 0.5, 1.0} px. Line = UNWEIGHTED LS fit of the per-1px-bin gradient
      centroids (same bins as V0). sigma_theta depends only on segment length:
          n = 0.9 L bins (t in [0.05L, 0.95L], 1 px apart),  sigma_theta = sigma0 * sqrt(12 / (n^3 - n))
      (exact LS slope variance for n equally spaced unit-weight samples). No noise/contrast estimate, no chi2.
      SIG_SYS stays 0.1 px.
V3  : V2 + chi2 inflation: sigma_theta *= sqrt(max(1, RSS / sigma0^2 / (n_bins - 2))) using the unweighted fit residuals.
The VP fit's inlier rule and the score use the same z (as in A1).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import v2
from .core import S, make_segs

SYS_V1 = (0.25, 0.5, 1.0)
SIG0 = (0.25, 0.5, 1.0)


def refine_both(img, seg_px, lmin_px):
    """Per segment: V0 refined line + sigma_theta (identical to v2.refine_with_uncertainty), and the unweighted refit
    (xy_u, RSS, n_bins) for V2/V3. Returns arrays over kept segments (length >= lmin on the V0 line)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    sn = v2.noise_sigma(gray)
    sg = sn * np.sqrt(12) / 8
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    seg_px = np.asarray(seg_px, np.float64).reshape(-1, 4)
    H, W = gray.shape
    rows = []
    for x1, y1, x2, y2 in seg_px:
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        L = np.linalg.norm(p2 - p1)
        if L < lmin_px * 0.9:
            continue
        d = (p2 - p1) / L
        n = np.array([-d[1], d[0]])
        xs = np.arange(int(max(0, min(x1, x2) - 3)), int(min(W - 1, max(x1, x2) + 3)) + 1)
        ys = np.arange(int(max(0, min(y1, y2) - 3)), int(min(H - 1, max(y1, y2) + 3)) + 1)
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float64)
        t = (P - p1) @ d
        s = (P - p1) @ n
        keep = (t >= 0.05 * L) & (t <= 0.95 * L) & (np.abs(s) <= 2.5)
        r0 = ru = None
        if keep.sum() >= 10:
            P, t, s = P[keep], t[keep], s[keep]
            g = np.stack([gx[P[:, 1].astype(int), P[:, 0].astype(int)], gy[P[:, 1].astype(int), P[:, 0].astype(int)]], 1)
            gn = g @ n
            gm = np.linalg.norm(g, axis=1) + 1e-9
            al = (np.abs(gn) / gm > np.cos(np.radians(22.5))) & (gm > 2 * sg)
            if al.sum() >= 10:
                t, s, w = t[al], s[al], np.abs(gn[al])
                b = np.floor(t).astype(int)
                ub, inv = np.unique(b, return_inverse=True)
                if len(ub) >= 5:
                    Sw = np.bincount(inv, w)
                    sstar = np.bincount(inv, w * s) / Sw
                    var = sg**2 * np.bincount(inv, (s - sstar[inv]) ** 2) / Sw**2 + 0.02**2
                    tc = np.bincount(inv, w * t) / Sw

                    def fit(ww):
                        t0 = np.average(tc, weights=ww)
                        A = np.stack([np.ones_like(tc), tc - t0], 1)
                        C = np.linalg.inv(A.T @ (A * ww[:, None]))
                        coef = C @ (A.T @ (ww * sstar))
                        r = sstar - A @ coef
                        q = p1 + t0 * d + coef[0] * n
                        dn = d + coef[1] * n
                        dn /= np.linalg.norm(dn)
                        e = np.r_[q + ((p1 - q) @ dn) * dn, q + ((p2 - q) @ dn) * dn]
                        return e, C, r
                    ww = 1 / var
                    e, C, r = fit(ww)
                    chi = (ww * r**2).sum() / max(len(tc) - 2, 1)
                    r0 = (e, np.sqrt(C[1, 1] * max(1.0, chi)))
                    eu, _, rr = fit(np.ones_like(tc))
                    ru = (eu, float((rr**2).sum()), len(tc))
        if r0 is None:
            r0 = (np.r_[p1, p2], 2.0 / L)
        if ru is None:
            ru = (np.r_[p1, p2], np.nan, 0)
        rows.append((r0[0], r0[1], ru[0], ru[1], ru[2]))
    if not rows:
        z = np.zeros((0, 4))
        return z, np.zeros(0), z, np.zeros(0), np.zeros(0), sn
    xy = np.array([r[0] for r in rows]); sth = np.array([r[1] for r in rows])
    xyu = np.array([r[2] for r in rows]); rss = np.array([r[3] for r in rows]); nb = np.array([r[4] for r in rows])
    Ln = lambda a: np.hypot(a[:, 2] - a[:, 0], a[:, 3] - a[:, 1])
    k = Ln(xy) >= lmin_px  # same kept set for all variants (V0 rule)
    return xy[k], sth[k], xyu[k], rss[k], nb[k], sn


def sth_fixed(L, sig0):
    n = np.maximum(0.9 * L, 3.0)
    return sig0 * np.sqrt(12.0 / (n**3 - n))


def _a1(xy, sth, sys_px, seed):
    v2._SYS[0] = sys_px
    s, _ = v2.a1_score(xy, sth, seed)
    return s


def compute_t4(img, seg_px, seed=0, lmin_frac=0.04):
    N = img.shape[0]
    xy, sth, xyu, rss, nb, sn = refine_both(img, seg_px, lmin_frac * N)
    f = S / N
    xy, xyu = xy * f, xyu * f
    out = dict(noise_sigma=sn, n_segs=len(xy))
    out["V0"] = _a1(xy, sth, v2.SIG_SYS * f, seed)
    for sy in SYS_V1:
        out[f"V1_sys{sy}"] = _a1(xy, sth, sy * f, seed)
    Lu = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1]) / f if len(xyu) else np.zeros(0)  # native px
    for s0 in SIG0:
        st = sth_fixed(Lu, s0)
        out[f"V2_sig{s0}"] = _a1(xyu, st, v2.SIG_SYS * f, seed)
        chi = np.where(nb > 2, rss / s0**2 / np.maximum(nb - 2, 1), 1.0)
        out[f"V3_sig{s0}"] = _a1(xyu, st * np.sqrt(np.maximum(1.0, np.nan_to_num(chi, nan=1.0))), v2.SIG_SYS * f, seed)
    v2._SYS[0] = v2.SIG_SYS
    return out


VARIANTS = ["V0"] + [f"V1_sys{s}" for s in SYS_V1] + [f"V2_sig{s}" for s in SIG0] + [f"V3_sig{s}" for s in SIG0]


# ---------------- Task 5: k longest segments per VP cluster (V2 fit, fixed sigma0) ----------------
K5 = (5, 10)
SIG5 = (0.5, 1.0)
VARIANTS5 = [f"K{k}_sig{s}_{st}" for k in K5 for s in SIG5 for st in ("zcap", "frac")]


def topk_scores(xyu, Lu_native, s0, sys_px, seed=0):
    """V2 fit + assignment; for each VP cluster with >= k assigned segments take its k longest.
    zcap: -mean(min(z,10)); frac: 1 - fraction(z>3); averaged over qualifying clusters (NaN if none)."""
    out = {f"K{k}_sig{s0}_{st}": np.nan for k in K5 for st in ("zcap", "frac")}
    if len(xyu) < 2:
        return out
    v2._SYS[0] = sys_px
    _, ctx = v2.a1_score(xyu, sth_fixed(Lu_native, s0), seed)
    if not ctx:
        return out
    L = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1])
    z, asg, arg = ctx["z"], ctx["asg"], ctx["arg"]
    for k in K5:
        zc, fr = [], []
        for c in range(len(ctx["vps"])):
            idx = np.where(asg & (arg == c))[0]
            if len(idx) >= k:
                top = idx[np.argsort(-L[idx])[:k]]
                zc.append(np.minimum(z[top], 10).mean()); fr.append((z[top] > 3).mean())
        if zc:
            out[f"K{k}_sig{s0}_zcap"] = -float(np.mean(zc)); out[f"K{k}_sig{s0}_frac"] = 1 - float(np.mean(fr))
    return out


def compute_t5(img, seg_px, seed=0, lmin_frac=0.04):
    N = img.shape[0]
    _, _, xyu, _, _, _ = refine_both(img, seg_px, lmin_frac * N)
    f = S / N
    xyu = xyu * f
    Lu = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1]) / f if len(xyu) else np.zeros(0)
    out = {}
    for s0 in SIG5:
        out.update(topk_scores(xyu, Lu, s0, v2.SIG_SYS * f, seed))
    v2._SYS[0] = v2.SIG_SYS
    return out


# ---------------- Task 6: fixed preprocessing before detection ----------------
SIGF = (1.0, 1.5, 2.0)
SCORES6 = ["K10_sig0.5_frac", "K10_sig1.0_frac", "K5_sig1.0_zcap", "V2_sig0.5"]


def preprocess(img, sigma_f):
    """Grayscale -> Gaussian blur sigma_f -> stretch 1st..99th percentile to [0, 255] (float32, clipped)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigma_f)
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255).astype(np.float32)


def stretch(img):
    """Task 10: grayscale -> map 1st..99th percentile to [0, 255], clip (float32). No blur."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255).astype(np.float32)


def compute_t6(pre, seg_px, seed=0, lmin_frac=0.04):
    """Scores on an already-preprocessed grey image (refinement also on it)."""
    N = pre.shape[0]
    _, _, xyu, _, _, _ = refine_both(pre, seg_px, lmin_frac * N)
    f = S / N
    xyu = xyu * f
    Lu = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1]) / f if len(xyu) else np.zeros(0)
    out = {}
    for s0 in (0.5, 1.0):
        o = topk_scores(xyu, Lu, s0, v2.SIG_SYS * f, seed)
        out.update({k: o[k] for k in SCORES6 if k in o})
    out["V2_sig0.5"] = _a1(xyu, sth_fixed(Lu, 0.5), v2.SIG_SYS * f, seed)
    v2._SYS[0] = v2.SIG_SYS
    out["n_segs"] = len(xyu)
    return out


# ---------------- Task 8: continuous pooled score ----------------
P8 = (1, 2, 3)
V8 = {"sig": lambda z: 1.0 / (1.0 + np.exp(-np.clip((z - 3.0) / 0.5, -50, 50))), "cap": lambda z: np.minimum(z, 10.0) / 10.0}
NAMES8 = [f"C_p{p}_{v}" for p in P8 for v in V8]


def compute_t8(src, seg_px, seed=0, lmin_frac=0.04, s0=1.0):
    """src: RGB image or preprocessed grey (refinement runs on it). V2 fit (sigma0=1, z<3 inliers, 15 deg assignment);
    score = 1 - sum_i w_i v(z_i) / sum_i w_i over all assigned segments (pooled), w_i = L_i^p (512-frame px)."""
    N = src.shape[0]
    _, _, xyu, _, _, _ = refine_both(src, seg_px, lmin_frac * N)
    f = S / N
    xyu = xyu * f
    out = {k: np.nan for k in NAMES8}
    out["Lc_v2"] = 0.0
    if len(xyu) < 2:
        return out
    L = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1])
    v2._SYS[0] = v2.SIG_SYS * f
    _, ctx = v2.a1_score(xyu, sth_fixed(L / f, s0), seed)
    v2._SYS[0] = v2.SIG_SYS
    if not ctx or not ctx["asg"].any():
        return out
    z, a = ctx["z"][ctx["asg"]], L[ctx["asg"]]
    out["Lc_v2"] = float(a.sum())
    for p in P8:
        w = (a / 100.0) ** p
        for vn, vf in V8.items():
            out[f"C_p{p}_{vn}"] = float(1 - (w * vf(z)).sum() / w.sum())
    return out
