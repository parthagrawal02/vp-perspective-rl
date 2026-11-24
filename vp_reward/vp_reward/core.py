"""Perspective-consistency reward: LSD line segments + sequential RANSAC vanishing points.

Conventions
-----------
* Images are preprocessed to a 512x512 square (center crop + resize).
* Segment coordinates are in pixels of the 512 image. For fitting they are mapped to
  normalized coords x_n = (x - c) / (S/2) so the image spans [-1, 1].
* Vanishing points (VPs) are unit homogeneous 3-vectors in normalized coords (w=0 -> at infinity).
* Angular error of segment i w.r.t. VP v: angle between the segment direction d_i and the
  direction from its midpoint m_i to v, u_i = v_xy - v_w * m_i (sign-free). Range [0, 90] deg.

All candidate scores are oriented "higher = more perspective-consistent":
  G        length-weighted fraction of segments that are inliers (< tau) to some VP
  NM       1 - (length-weighted fraction with tau <= best error < 8 deg)   [near-miss penalty]
  RegMean  exp(-R_mean / 2deg), R_mean = length-weighted mean of per-(tile,VP) LOO median errors
  RegMax   exp(-R_max / 2deg),  R_max  = worst (tile,VP) LOO median error
Structure gate g = min(1, Lc / L_ref) (Lc = length of segments assigned (<15 deg) to a VP).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import cv2
import numpy as np

S = 512  # working resolution
NM_HI = 8.0  # deg, upper edge of near-miss band
ASSIGN_DEG = 15.0  # deg, assignment threshold for regional score / Lc
REG_SCALE = 2.0  # deg, temperature of the regional score


@dataclass
class VPConfig:
    lmin_frac: float = 0.04  # keep segments with length >= lmin_frac * S
    tau: float = 1.5  # deg, RANSAC inlier threshold
    max_vps: int = 3
    min_vp_frac: float = 0.10  # stop when a new VP explains < this fraction of total length
    ransac_iters: int = 512
    refine_rounds: int = 3
    grid: int = 3
    min_tile_segs: int = 3
    l_ref: float | None = None  # gate reference length (px); None -> gate = 1

    def tag(self) -> str:
        return f"L{int(round(self.lmin_frac * 100))}_t{self.tau:g}"


# ----------------------------------------------------------------------------- preprocessing
def center_square_resize(img: np.ndarray, size: int = S) -> np.ndarray:
    """Center-crop to a square, resize to size x size. img: HxWx3 uint8 RGB."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = img[y0 : y0 + s, x0 : x0 + s]
    if s != size:
        interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
        img = cv2.resize(img, (size, size), interpolation=interp)
    return np.ascontiguousarray(img)


_LSD = None


def detect_segments(img512: np.ndarray) -> np.ndarray:
    """OpenCV LSD on grayscale. Returns (N,4) float32 [x1,y1,x2,y2] (no length filtering)."""
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    gray = cv2.cvtColor(img512, cv2.COLOR_RGB2GRAY)
    lines = _LSD.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


# ----------------------------------------------------------------------------- geometry
@dataclass
class Segs:
    xy: np.ndarray  # (N,4) pixel coords
    m: np.ndarray  # (N,2) normalized midpoints
    d: np.ndarray  # (N,2) unit directions
    w: np.ndarray  # (N,) lengths in px
    lines: np.ndarray  # (N,3) homogeneous lines through segment (normalized coords), (a,b) unit

    @property
    def n(self):
        return len(self.w)


def make_segs(seg_px: np.ndarray, lmin_frac: float) -> Segs:
    seg_px = np.asarray(seg_px, np.float64).reshape(-1, 4)
    L = np.hypot(seg_px[:, 2] - seg_px[:, 0], seg_px[:, 3] - seg_px[:, 1])
    keep = L >= lmin_frac * S
    seg_px, L = seg_px[keep], L[keep]
    p1 = (seg_px[:, :2] - S / 2) / (S / 2)
    p2 = (seg_px[:, 2:] - S / 2) / (S / 2)
    m = (p1 + p2) / 2
    d = p2 - p1
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)  # unit normal
    lines = np.concatenate([nrm, -(nrm * m).sum(1, keepdims=True)], 1)
    return Segs(seg_px.astype(np.float32), m, d, L, lines)


def ang_err(segs_m, segs_d, V):
    """Angular error (deg). V: (3,) or (K,3). Returns (N,) or (K,N)."""
    V = np.atleast_2d(V)
    u = V[:, None, :2] - V[:, None, 2:3] * segs_m[None]  # (K,N,2)
    un = np.linalg.norm(u, axis=2)
    c = np.abs((u * segs_d[None]).sum(2)) / np.maximum(un, 1e-12)
    e = np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))
    e = np.where(un < 1e-9, 90.0, e)
    return e[0] if e.shape[0] == 1 else e


def _norm_vp(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def refine_vp(segs: Segs, idx: np.ndarray, v0: np.ndarray, iters: int = 4, robust_deg: float | None = None):
    """Length-weighted least squares VP refit on segments idx.

    Minimizes sum_i w_i * (l_i . v)^2 / |u_i|^2 (~ sum w_i sin^2(err_i)) by IRLS starting at v0.
    If robust_deg is given, adds Cauchy weights 1/(1+(err/robust_deg)^2) (used for LOO refits,
    whose input set contains up-to-15-deg assignments).
    """
    if len(idx) < 2:
        return v0
    l = segs.lines[idx]
    w = segs.w[idx]
    m, d = segs.m[idx], segs.d[idx]
    v = v0.copy()
    for _ in range(iters):
        u = v[:2][None] - v[2] * m
        un2 = np.maximum((u**2).sum(1), 1e-6)
        ww = w / un2
        if robust_deg is not None:
            e = ang_err(m, d, v)
            ww = ww / (1 + (e / robust_deg) ** 2)
        A = (l * ww[:, None]).T @ l
        evals, evecs = np.linalg.eigh(A)
        vn = _norm_vp(evecs[:, 0])
        if vn @ v < 0:
            vn = -vn
        v = vn
    return v


def ransac_vp(segs: Segs, avail: np.ndarray, tau: float, iters: int, rng: np.random.Generator):
    """One RANSAC VP among segments `avail` (index array). Returns (v, inlier_idx_within_avail_mask)."""
    na = len(avail)
    if na < 2:
        return None
    p = segs.w[avail] / segs.w[avail].sum()
    i = rng.choice(na, size=iters, p=p)
    j = rng.choice(na, size=iters, p=p)
    ok = i != j
    i, j = avail[i[ok]], avail[j[ok]]
    V = np.cross(segs.lines[i], segs.lines[j])
    nv = np.linalg.norm(V, axis=1)
    good = nv > 1e-12
    V = V[good] / nv[good, None]
    if len(V) == 0:
        return None
    E = ang_err(segs.m[avail], segs.d[avail], V)  # (K, na)
    E = np.atleast_2d(E)
    sc = ((E < tau) * segs.w[avail][None]).sum(1)
    b = int(np.argmax(sc))
    return V[b]


@dataclass
class VPResult:
    scores: dict
    diag: dict


def fit_and_score(seg_px: np.ndarray, cfg: VPConfig, seed: int = 0) -> VPResult:
    rng = np.random.default_rng(seed)
    segs = make_segs(seg_px, cfg.lmin_frac)
    Ltot = float(segs.w.sum())
    diag = dict(n_segs=int(segs.n), total_len=Ltot, vps=[], vp_len=[], Lc=0.0, tile_err=[], R_mean=np.nan,
                R_max=np.nan, n_tile_entries=0)
    nan_scores = dict(G=0.0, NM=1.0, RegMean=0.0, RegMax=0.0, gate=0.0)
    if segs.n < 2 or Ltot <= 0:
        return VPResult(nan_scores, diag)

    # ---- sequential RANSAC
    vps = []
    avail = np.arange(segs.n)
    for _ in range(cfg.max_vps):
        v = ransac_vp(segs, avail, cfg.tau, cfg.ransac_iters, rng)
        if v is None:
            break
        for _r in range(cfg.refine_rounds):
            inl = avail[ang_err(segs.m[avail], segs.d[avail], v) < cfg.tau]
            v = refine_vp(segs, inl, v)
        e_av = ang_err(segs.m[avail], segs.d[avail], v)
        inl = avail[e_av < cfg.tau]
        if segs.w[inl].sum() < cfg.min_vp_frac * Ltot:
            break
        vps.append(v)
        diag["vp_len"].append(float(segs.w[inl].sum()))
        avail = avail[e_av >= cfg.tau]
        if len(avail) < 2:
            break
    diag["vps"] = [v.tolist() for v in vps]
    if not vps:
        return VPResult(dict(nan_scores, NM=1.0), diag)

    V = np.stack(vps)
    E = np.atleast_2d(ang_err(segs.m, segs.d, V))  # (K,N)
    best = E.min(0)
    arg = E.argmin(0)
    G = float(segs.w[best < cfg.tau].sum() / Ltot)
    NMfrac = float(segs.w[(best >= cfg.tau) & (best < NM_HI)].sum() / Ltot)
    assigned = best < ASSIGN_DEG
    Lc = float(segs.w[assigned].sum())
    diag["Lc"] = Lc

    # ---- regional leave-one-region-out
    px_mid = (segs.m + 1) * (S / 2)
    tx = np.clip((px_mid[:, 0] / S * cfg.grid).astype(int), 0, cfg.grid - 1)
    ty = np.clip((px_mid[:, 1] / S * cfg.grid).astype(int), 0, cfg.grid - 1)
    tile = ty * cfg.grid + tx
    entries = []  # (tile, k, err_median, weight)
    seg_tile_err = np.full(segs.n, np.nan)  # per-segment LOO error (for visualisation)
    for t in range(cfg.grid * cfg.grid):
        for k in range(len(vps)):
            in_t = np.where(assigned & (arg == k) & (tile == t))[0]
            if len(in_t) < cfg.min_tile_segs:
                continue
            out_t = np.where(assigned & (arg == k) & (tile != t))[0]
            if len(out_t) < 2:
                continue
            v_loo = refine_vp(segs, out_t, V[k], iters=6, robust_deg=cfg.tau)
            e = ang_err(segs.m[in_t], segs.d[in_t], v_loo)
            seg_tile_err[in_t] = e
            med = weighted_median(e, segs.w[in_t])
            entries.append((t, k, float(med), float(segs.w[in_t].sum())))
    diag["tile_err"] = entries
    diag["n_tile_entries"] = len(entries)
    diag["seg_tile_err"] = seg_tile_err
    diag["seg_best_err"] = best
    diag["seg_xy"] = segs.xy
    if entries:
        ew = np.array([e[3] for e in entries])
        em = np.array([e[2] for e in entries])
        R_mean = float((em * ew).sum() / ew.sum())
        R_max = float(em.max())
        RegMean, RegMax = float(np.exp(-R_mean / REG_SCALE)), float(np.exp(-R_max / REG_SCALE))
    else:
        R_mean = R_max = np.nan
        RegMean = RegMax = 0.0  # no measurable structure -> worst regional score
    diag["R_mean"], diag["R_max"] = R_mean, R_max
    gate = 1.0 if cfg.l_ref is None else float(min(1.0, Lc / cfg.l_ref))
    return VPResult(dict(G=G, NM=1.0 - NMfrac, RegMean=RegMean, RegMax=RegMax, gate=gate), diag)


def weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, 0.5 * cw[-1])]
