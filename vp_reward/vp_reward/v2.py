"""Task 3 variants: per-segment uncertainty, noise-normalized (z) VP fit, equal-evidence, structured-only stats,
null calibration, horizon add-on.

Per-segment uncertainty (documented method)
-------------------------------------------
1. Image noise sigma_n: Immerkaer (1996) fast estimator (Laplacian-difference mask, mean abs), floored at 0.5 grey level.
2. Gradients: Sobel/8 (grey levels per px). Per-component gradient noise sigma_g = sigma_n * sqrt(12)/8.
3. For each segment (unit dir d, normal n, length L): pixels with along-coordinate t in [0.05L, 0.95L] and |s| <= 2.5 px,
   whose gradient is aligned with +-n within 22.5 deg. Weight w_i = |grad_i . n|.
4. Per 1-px t-bin j: sub-pixel edge offset s*_j = sum w s / sum w ("gradient centroid") and its variance from noise
   propagation var_j = sigma_g^2 * sum (s_i - s*_j)^2 / (sum w_i)^2 (+ 0.02^2 px^2 floor). High contrast -> small var.
5. Weighted LS line s = a + b t (weights 1/var_j). Covariance scaled by max(1, chi2/dof) (non-straight / textured edges
   inflate it). sigma_theta = sqrt(var b) [rad], sigma_d = sqrt(var a at the bin centroid) [px]. The refined segment is
   the original one moved onto this line (endpoints projected).
6. Fallback (<5 valid bins): original segment, sigma_theta = 2/L rad, sigma_d = 1 px.
z of a segment w.r.t. VP v: endpoint distance from the line through its midpoint and v, (L/2)|sin(alpha)|, divided by the
endpoint position uncertainty perpendicular to that line, sqrt((L/2)^2 sigma_theta^2 + SIG_SYS^2), SIG_SYS = 0.1 px.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import S, VPConfig, Segs, ang_err, fit_and_score, make_segs, refine_vp, weighted_median, _norm_vp

SIG_SYS = 0.1
Z_IN = 3.0
ASSIGN_DEG = 15.0


def noise_sigma(gray):
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float32), -1, M, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    s = np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2))
    return max(float(s), 0.5)


def refine_with_uncertainty(img512, seg_px, lmin_px):
    gray = cv2.cvtColor(img512, cv2.COLOR_RGB2GRAY).astype(np.float32) if img512.ndim == 3 else img512.astype(np.float32)
    sn = noise_sigma(gray)
    sg = sn * np.sqrt(12) / 8
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    seg_px = np.asarray(seg_px, np.float64).reshape(-1, 4)
    out_xy, sth, sd = [], [], []
    H, W = gray.shape
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
        res = None
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
                    ww = 1 / var
                    A = np.stack([np.ones_like(tc), tc - np.average(tc, weights=ww)], 1)
                    C = np.linalg.inv(A.T @ (A * ww[:, None]))
                    coef = C @ (A.T @ (ww * sstar))
                    r = sstar - A @ coef
                    chi = (ww * r**2).sum() / max(len(tc) - 2, 1)
                    C = C * max(1.0, chi)
                    t0 = np.average(tc, weights=ww)
                    # refined line: point p1 + t0 d + a n, direction d + b n
                    q = p1 + t0 * d + coef[0] * n
                    dn = d + coef[1] * n
                    dn /= np.linalg.norm(dn)
                    e1 = q + ((p1 - q) @ dn) * dn
                    e2 = q + ((p2 - q) @ dn) * dn
                    res = (np.r_[e1, e2], np.sqrt(C[1, 1]), np.sqrt(C[0, 0]))
        if res is None:
            res = (np.r_[p1, p2], 2.0 / L, 1.0)
        out_xy.append(res[0]); sth.append(res[1]); sd.append(res[2])
    xy = np.array(out_xy).reshape(-1, 4)
    sth, sd = np.array(sth), np.array(sd)
    L = np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]) if len(xy) else np.zeros(0)
    k = L >= lmin_px
    return xy[k], sth[k], sd[k], sn


_SYS = [SIG_SYS]  # systematic floor in 512-frame px (rescaled for other resolutions in compute_all)


def z_of(alpha_deg, L, sth):
    D = (L / 2) * np.abs(np.sin(np.radians(alpha_deg)))
    return D / np.sqrt((L / 2) ** 2 * sth**2 + _SYS[0] ** 2)


def fit_z(xy, sth, seed=0, iters=512, max_vps=3, min_frac=0.10):
    """Sequential RANSAC with inlier = z < 3; refinement weighted by inverse orientation variance."""
    rng = np.random.default_rng(seed)
    sg = make_segs(xy, 0.0)
    Lpx = sg.w.copy()
    sgw = Segs(sg.xy, sg.m, sg.d, 1.0 / np.maximum(sth, 1e-5) ** 2, sg.lines)  # weights for LS refit
    Ltot = Lpx.sum()
    vps, avail = [], np.arange(sg.n)
    for _ in range(max_vps):
        na = len(avail)
        if na < 2:
            break
        p = Lpx[avail] / Lpx[avail].sum()
        i, j = rng.choice(na, iters, p=p), rng.choice(na, iters, p=p)
        ok = i != j
        i, j = avail[i[ok]], avail[j[ok]]
        V = np.cross(sg.lines[i], sg.lines[j]); nv = np.linalg.norm(V, axis=1); V = V[nv > 1e-12] / nv[nv > 1e-12, None]
        if len(V) == 0:
            break
        Z = z_of(np.atleast_2d(ang_err(sg.m[avail], sg.d[avail], V)), Lpx[avail][None], sth[avail][None])
        v = V[int(np.argmax(((Z < Z_IN) * Lpx[avail][None]).sum(1)))]
        for _r in range(3):
            inl = avail[z_of(ang_err(sg.m[avail], sg.d[avail], v), Lpx[avail], sth[avail]) < Z_IN]
            v = refine_vp(sgw, inl, v)
        za = z_of(ang_err(sg.m[avail], sg.d[avail], v), Lpx[avail], sth[avail])
        if Lpx[avail[za < Z_IN]].sum() < min_frac * Ltot:
            break
        vps.append(v)
        avail = avail[za >= Z_IN]
    return sg, Lpx, vps


def a1_score(xy, sth, seed=0):
    if len(xy) < 2:
        return np.nan, {}
    sg, Lpx, vps = fit_z(xy, sth, seed)
    if not vps:
        return np.nan, {}
    E = np.atleast_2d(ang_err(sg.m, sg.d, np.stack(vps)))
    best, arg = E.min(0), E.argmin(0)
    asg = best < ASSIGN_DEG
    if Lpx[asg].sum() <= 0:
        return np.nan, {}
    z = z_of(best, Lpx, sth)
    return float(1 - Lpx[asg & (z > Z_IN)].sum() / Lpx[asg].sum()), dict(vps=vps, best=best, arg=arg, z=z, asg=asg)


def perfect_null(xy, sth, sd, vps, best, arg, asg, rng):
    """Rotate each assigned segment about its midpoint to point exactly at its VP, then add noise matched to its
    uncertainty (orientation N(0, sth), normal offset N(0, sd)). Unassigned segments are left unchanged."""
    out = xy.copy()
    sg = make_segs(xy, 0.0)
    for i in np.where(asg)[0]:
        v = vps[arg[i]]
        mpx = (xy[i, :2] + xy[i, 2:]) / 2
        mn = sg.m[i]
        u = v[:2] - v[2] * mn
        if np.linalg.norm(u) < 1e-9:
            continue
        u = u / np.linalg.norm(u)
        if u @ sg.d[i] < 0:
            u = -u
        a = rng.normal(0, sth[i])
        c, s_ = np.cos(a), np.sin(a)
        u = np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1]])
        nrm = np.array([-u[1], u[0]])
        mpx = mpx + rng.normal(0, sd[i]) * nrm
        h = sg.w[i] / 2
        out[i] = np.r_[mpx - h * u, mpx + h * u]
    return out


def horizon_err(vps):
    """Angle (deg) between the horizon (line through the two non-vertical VPs) and the direction perpendicular to
    center->vertical-VP (square pixels, principal point at image center). NaN if not 1 vertical + 2 horizontal VPs."""
    if len(vps) < 3:
        return np.nan
    dirs = []
    for v in vps:
        u = v[:2] if abs(v[2]) < 1e-12 else v[:2] / v[2]  # normalized coords, image center = 0
        dirs.append(u / (np.linalg.norm(u) + 1e-12))
    vert = [k for k, u in enumerate(dirs) if abs(u[1]) > np.cos(np.radians(30))]
    if len(vert) != 1:
        return np.nan
    hz = [vps[k] for k in range(len(vps)) if k != vert[0]][:2]
    h = np.cross(hz[0], hz[1])
    hn = h[:2] / (np.linalg.norm(h[:2]) + 1e-12)
    c = abs(hn @ dirs[vert[0]])
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


def compute_all(img512, seg_px, seed=0, n_null=20, lmin_frac=0.04, tau=1.5, do_null=True):
    """All Task-3 scores for one image. Higher = better for every key starting with 's_'."""
    N = img512.shape[0]  # square image of any size; refine at native res, then map to the 512 frame
    xy, sth, sd, sn = refine_with_uncertainty(img512, seg_px, lmin_frac * N)
    f = S / N
    xy, sd = xy * f, sd * f
    _SYS[0] = SIG_SYS * f
    out = dict(noise_sigma=sn, n_segs=len(xy), total_len=float(np.hypot(xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]).sum()) if len(xy) else 0.0,
               med_sigma_theta_deg=float(np.degrees(np.median(sth))) if len(xy) else np.nan)
    cfg = VPConfig(lmin_frac=0.0, tau=tau)
    # ---- angular fit (Task 2 machinery on refined segments)
    r = fit_and_score(xy, cfg, seed)
    out.update(s_RegMean=r.scores["RegMean"], s_G=r.scores["G"], R_mean=r.diag["R_mean"], Lc=r.diag["Lc"], n_vps=len(r.diag["vps"]))
    vps = [np.array(v) for v in r.diag["vps"]]
    out["horizon_err_deg"] = horizon_err(vps)
    if vps and len(xy):
        sg = make_segs(xy, 0.0)
        E = np.atleast_2d(ang_err(sg.m, sg.d, np.stack(vps)))
        best, arg = E.min(0), E.argmin(0)
        asg = best < ASSIGN_DEG
        z = z_of(best, sg.w, sth)
        # A3: structured-only distributions
        if asg.sum() >= 3:
            for nm, x in (("deg", best[asg]), ("z", z[asg])):
                out[f"s_A3_med_{nm}"] = -float(np.median(x)); out[f"s_A3_mean_{nm}"] = -float(np.mean(x)); out[f"s_A3_p90_{nm}"] = -float(np.percentile(x, 90))
        # A2: equal evidence (k longest per cluster)
        for k in (5, 10, 20):
            md, mz = [], []
            for c in range(len(vps)):
                idx = np.where(asg & (arg == c))[0]
                if len(idx) >= k:
                    top = idx[np.argsort(-sg.w[idx])[:k]]
                    md.append(best[top].mean()); mz.append(z[top].mean())
            out[f"s_A2_k{k}_deg"] = -float(np.mean(md)) if md else np.nan
            out[f"s_A2_k{k}_z"] = -float(np.mean(mz)) if mz else np.nan
        ang_ctx = (vps, best, arg, asg)
    else:
        ang_ctx = None
    # ---- A1 (z fit)
    a1, ctx = a1_score(xy, sth, seed)
    out["s_A1"] = a1
    # ---- A4 null calibration
    if do_null:
        rng = np.random.default_rng(10_000 + seed)
        if ang_ctx is not None and not np.isnan(out["R_mean"]):
            nulls = []
            for _ in range(n_null):
                nx = perfect_null(xy, sth, sd, *ang_ctx, rng)
                nulls.append(fit_and_score(nx, cfg, seed).diag["R_mean"])
            nulls = np.array([x for x in nulls if not np.isnan(x)])
            if len(nulls) >= 5:
                out["s_A4_RegMean"] = float((nulls.mean() - out["R_mean"]) / max(nulls.std(), 1e-3))
        if ctx:
            nulls = []
            for _ in range(n_null):
                nx = perfect_null(xy, sth, sd, ctx["vps"], ctx["best"], ctx["arg"], ctx["asg"], rng)
                v, _c = a1_score(nx, sth, seed)
                nulls.append(1 - v)  # violation fraction
            nulls = np.array([x for x in nulls if not np.isnan(x)])
            if len(nulls) >= 5:
                out["s_A4_A1"] = float((nulls.mean() - (1 - a1)) / max(nulls.std(), 1e-3))
    return out
