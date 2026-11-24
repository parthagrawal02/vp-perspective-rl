"""Synthetic perspective scenes for unit tests (rendered, then run through LSD like real images)."""
import cv2
import numpy as np

S = 512


def _segments_toward(vp, n, rng, lo=30, hi=110, region=None):
    """n segments lying on rays that pass through vp (pixel coords; vp may be far away / at infinity
    given as ('inf', angle_deg))."""
    segs = []
    while len(segs) < n:
        x, y = rng.uniform(20, S - 20, 2)
        if region is not None and not region(x, y):
            continue
        if isinstance(vp, tuple) and vp[0] == "inf":
            a = np.radians(vp[1])
            d = np.array([np.cos(a), np.sin(a)])
        else:
            d = np.array(vp, float) - (x, y)
            if np.linalg.norm(d) < 60:
                continue
            d /= np.linalg.norm(d)
        L = rng.uniform(lo, hi)
        p1, p2 = np.array([x, y]) - d * L / 2, np.array([x, y]) + d * L / 2
        if p1.min() < 5 or p2.min() < 5 or p1.max() > S - 5 or p2.max() > S - 5:
            continue
        segs.append(np.r_[p1, p2])
    return np.array(segs)


def scene(kind="one", seed=0, n_per_vp=40):
    rng = np.random.default_rng(seed)
    if kind == "one":
        vps = [(256.0, 230.0), ("inf", 0.0), ("inf", 90.0)]
    elif kind == "two":
        vps = [(-420.0, 250.0), (930.0, 262.0), ("inf", 90.0)]
    else:
        raise ValueError(kind)
    groups = [_segments_toward(v, n_per_vp, rng) for v in vps]
    return groups  # list of (n,4) arrays, one per VP


def bend(groups, delta_deg, group=0, region=lambda x, y: x > 2 * S / 3):
    """Rotate the segments of `group` whose midpoint lies in `region` by delta_deg around the
    region's segment centroid (bends that part of one structure)."""
    g = groups[group].copy()
    mid = (g[:, :2] + g[:, 2:]) / 2
    sel = np.array([region(x, y) for x, y in mid])
    if sel.sum() == 0 or delta_deg == 0:
        return [x.copy() for x in groups], sel
    piv = mid[sel].mean(0)
    a = np.radians(delta_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    for j in np.where(sel)[0]:
        for k in (0, 2):
            g[j, k : k + 2] = (g[j, k : k + 2] - piv) @ R.T + piv
    out = [x.copy() for x in groups]
    out[group] = g
    return out, sel


def render(groups, seed=0, noise=2.0):
    rng = np.random.default_rng(seed + 1000)
    img = np.full((S, S), 200, np.uint8)
    for g in groups:
        for x1, y1, x2, y2 in g:
            cv2.line(img, (int(round(x1 * 16)), int(round(y1 * 16))), (int(round(x2 * 16)), int(round(y2 * 16))),
                     40, 2, cv2.LINE_AA, shift=4)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, 2)
