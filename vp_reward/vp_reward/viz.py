"""Visualize the frozen perspective reward's geometry on an image: fitted lines, their VP assignment, per-segment z, and
where the vanishing points are. Read-only w.r.t. the reward: it replays v4.compute_t8 step by step (same refit, sigma,
RANSAC seed, thresholds) and asserts its G / Lc equal compute_t8's.

    python -m vp_reward.viz IMG [IMG ...] [--detector auto|deeplsd|lsd] [--out DIR] [--synthetic one|two] [--bend DEG]

--detector auto uses DeepLSD (the reward's detector) when it and $VP_DEEPLSD_CKPT are available, else OpenCV LSD
(a stand-in: the numbers then differ from the real reward; the title says which detector was used).
Non-512 images are center-cropped and resized to 512 (core.center_square_resize).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import final as F
from . import v2
from .core import S, ang_err, center_square_resize, detect_segments, make_segs
from .v4 import compute_t8, refine_both, stretch, sth_fixed

VP_COLORS = ["#1f77b4", "#2ca02c", "#9467bd"]
FAR = 6.0  # VPs farther than FAR * S/2 from the image centre are drawn as directions on the panel border


def _load(path):
    from PIL import Image
    a = np.array(Image.open(path).convert("RGB"))
    return a if a.shape[:2] == (S, S) else center_square_resize(a)


def get_detector(kind="auto", device=None):
    """-> (name, fn(pre) -> (N,4) px lines)."""
    if kind in ("auto", "deeplsd"):
        try:
            import torch
            dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            if not os.path.exists(F.DEEPLSD_CKPT):
                raise FileNotFoundError(F.DEEPLSD_CKPT)
            net = F.load_deeplsd(dev)
            return f"DeepLSD ({dev})", lambda pre: F.detect_lines(net, pre, dev)
        except Exception as e:  # noqa: BLE001
            if kind == "deeplsd":
                raise
            print(f"[viz] DeepLSD unavailable ({type(e).__name__}: {e}); falling back to OpenCV LSD")
    return "OpenCV LSD (stand-in)", lambda pre: detect_segments(np.repeat(pre.astype(np.uint8)[..., None], 3, 2))


def analyse(img, detect):
    """Replays compute_t8 (FINAL.md section 2) and returns everything needed to draw it."""
    pre = stretch(img)
    raw = np.asarray(detect(pre), np.float64).reshape(-1, 4)
    _, _, xyu, _, _, _ = refine_both(pre, raw, F.LMIN_FRAC * S)  # 512 input -> f = 1
    out = dict(pre=pre, raw=raw, xy=xyu, vps=[], G=np.nan, Lc=0.0)
    if len(xyu) >= 2:
        L = np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1])
        sth = sth_fixed(L, F.SIGMA0)
        v2._SYS[0] = v2.SIG_SYS
        sg, Lpx, vps = v2.fit_z(xyu, sth, F.RANSAC_SEED)
        out.update(L=L, vps=vps)
        if vps:
            E = np.atleast_2d(ang_err(sg.m, sg.d, np.stack(vps)))
            best, arg = E.min(0), E.argmin(0)
            asg = best < v2.ASSIGN_DEG
            z = v2.z_of(best, Lpx, sth)
            out.update(best=best, arg=arg, asg=asg, z=z)
            if asg.any():
                w = (L[asg] / 100.0) ** 2
                vz = F_sig(z[asg])
                out["G"] = float(1 - (w * vz).sum() / w.sum())
                out["Lc"] = float(L[asg].sum())
    ref = compute_t8(pre, raw, seed=F.RANSAC_SEED, lmin_frac=F.LMIN_FRAC, s0=F.SIGMA0)
    assert np.allclose([out["G"], out["Lc"]], [ref["C_p2_sig"], ref["Lc_v2"]], equal_nan=True), (out["G"], ref["C_p2_sig"])
    return out


def F_sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip((z - 3.0) / 0.5, -50, 50)))


def vp_pixel(v):
    """Normalized homogeneous VP -> (pixel xy or None, unit direction from centre)."""
    if abs(v[2]) > 1e-9:
        p = v[:2] / v[2]
        r = np.linalg.norm(p)
        if r < FAR:
            return p * (S / 2) + S / 2, p / max(r, 1e-12)
    d = v[:2] / max(np.linalg.norm(v[:2]), 1e-12)
    return None, d


def draw(img, a, title, path, detector):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    xy, vps = a["xy"], a["vps"]
    has = "z" in a
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    grey = np.stack([a["pre"] / 255] * 3, -1) * 0.55 + 0.45  # washed-out background so lines read

    # ---- panel 1: assignment
    ax[0].imshow(grey)
    if len(xy):
        segs = xy.reshape(-1, 2, 2)
        if has:
            cols = [VP_COLORS[k] if ok else "#999999" for k, ok in zip(a["arg"], a["asg"])]
            bad = a["asg"] & (a["z"] > 3)
            ax[0].add_collection(LineCollection(segs[bad], colors="#d62728", linewidths=4.5, alpha=0.55))
            ax[0].add_collection(LineCollection(segs, colors=cols, linewidths=[1.0 if not s else 1.8 for s in a["asg"]]))
        else:
            ax[0].add_collection(LineCollection(segs, colors="#999999", linewidths=1.2))
    handles = [plt.Line2D([], [], c=VP_COLORS[k], lw=2, label=f"VP{k + 1}: {int((a['asg'] & (a['arg'] == k)).sum())} segs")
               for k in range(len(vps))] if has else []
    handles += [plt.Line2D([], [], c="#999999", lw=1, label="unassigned (>15°)"),
                plt.Line2D([], [], c="#d62728", lw=4, alpha=0.55, label="violator (z > 3)")]
    ax[0].legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
    ax[0].set_title(f"Lines by vanishing point  ({len(a['raw'])} detected, {len(xy)} kept ≥ {F.LMIN_FRAC * S:.1f}px)", fontsize=10)

    # ---- panel 2: z per segment
    ax[1].imshow(grey)
    if has and a["asg"].any():
        m = a["asg"]
        lc = LineCollection(xy.reshape(-1, 2, 2)[m], array=np.minimum(a["z"][m], 10), cmap="RdYlGn_r",
                            norm=matplotlib.colors.Normalize(0, 10), linewidths=1.2 + 3 * (a["L"][m] / a["L"][m].max()))
        ax[1].add_collection(lc)
        cb = fig.colorbar(lc, ax=ax[1], fraction=0.046, pad=0.02)
        cb.set_label("z (miss distance / own uncertainty), capped at 10; inlier < 3", fontsize=8)
        cb.ax.axhline(3, c="k", lw=1)
    ax[1].set_title("Per-segment z w.r.t. its VP (width ∝ length; weight in G ∝ L²)", fontsize=10)

    # ---- panel 3: zoomed-out geometry
    pts = [vp_pixel(v) for v in vps]
    fin = [p for p, _ in pts if p is not None]
    span = max([S * 0.75] + [np.abs(p - S / 2).max() * 1.12 for p in fin])
    lo, hi = S / 2 - span, S / 2 + span
    ax[2].imshow(grey, extent=(0, S, S, 0))
    ax[2].add_patch(plt.Rectangle((0, 0), S, S, fill=False, ec="k", lw=1))
    if has:
        for k, (p, d) in enumerate(pts):
            idx = np.where(a["asg"] & (a["arg"] == k))[0]
            c = VP_COLORS[k]
            mids = (xy[idx, :2] + xy[idx, 2:]) / 2
            if p is not None:  # ray from each segment midpoint to the VP
                rays = np.stack([mids, np.repeat(p[None], len(mids), 0)], 1)
                ax[2].add_collection(LineCollection(rays, colors=c, linewidths=0.5, alpha=0.35))
                ax[2].plot(*p, marker="*", ms=18, c=c, mec="k", zorder=5)
                ax[2].annotate(f"VP{k + 1}", p, xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold", c=c)
            else:  # (near-)infinite: parallel lines; draw the direction through the image and an arrow at the border
                ext = 2 * span
                rays = np.stack([mids - d * ext, mids + d * ext], 1)
                ax[2].add_collection(LineCollection(rays, colors=c, linewidths=0.5, alpha=0.25))
                tip = S / 2 + d * span * 0.93
                ax[2].annotate("", tip, S / 2 + d * span * 0.7, arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
                ax[2].annotate(f"VP{k + 1} → ∞", tip, xytext=(4, -12), textcoords="offset points", fontsize=9, weight="bold", c=c)
            ax[2].add_collection(LineCollection(xy[idx].reshape(-1, 2, 2), colors=c, linewidths=1.6))
    ax[2].set_xlim(lo, hi); ax[2].set_ylim(hi, lo); ax[2].set_aspect("equal")
    ax[2].set_title(f"Vanishing points (zoomed out; ≥{FAR:g}× half-width shown as ∞ direction)", fontsize=10)

    for x in ax[:2]:
        x.set_xlim(0, S); x.set_ylim(S, 0)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    G = a["G"]
    zG = ((G if not np.isnan(G) else F.G_NAN_FILL) - F.G_MEAN) / F.G_STD
    gtxt = f"G = {G:.3f}" if not np.isnan(G) else f"G undefined → filled {F.G_NAN_FILL:.3f}"
    fig.suptitle(f"{title}   |   {gtxt}   z(G) = {zG:+.2f}   Lc = {a['Lc']:.0f}px   VPs = {len(vps)}   |   lines: {detector}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vp_reward.viz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--detector", default="auto", choices=["auto", "deeplsd", "lsd"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=".", help="output directory (files: <stem>_vp.png)")
    ap.add_argument("--synthetic", choices=["one", "two"], help="also render a synthetic one-/two-point scene (vp_reward.synth)")
    ap.add_argument("--bend", type=float, default=0.0, help="with --synthetic: bend part of VP1's lines by this many degrees")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    name, det = get_detector(a.detector, a.device)
    jobs = [(p, _load(p), os.path.splitext(os.path.basename(p))[0]) for p in a.images]
    if a.synthetic:
        from .synth import bend, render, scene
        g, _ = bend(scene(a.synthetic, seed=2), a.bend)
        jobs.append((f"synthetic {a.synthetic}-point, bend {a.bend:g}°", render(g, 0), f"synth_{a.synthetic}_bend{a.bend:g}"))
    if not jobs:
        ap.error("give image paths and/or --synthetic")
    with np.errstate(all="ignore"):  # numpy+Accelerate emits spurious matmul warnings on macOS
        for title, img, stem in jobs:
            r = analyse(img, det)
            p = draw(img, r, os.path.basename(title), os.path.join(a.out, f"{stem}_vp.png"), name)
            print(f"{p}: G={r['G']:.4f} Lc={r['Lc']:.0f} VPs={len(r['vps'])}")


if __name__ == "__main__":
    main()
