"""Batch / parallel API and DiffusionNFT-style wrapper."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .core import VPConfig, center_square_resize, detect_segments, fit_and_score

CANDIDATES = ("G", "NM", "RegMean", "RegMax")


def _init_worker():
    import cv2

    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def _seg_one(img):
    return detect_segments(center_square_resize(img))


def _fit_one(args):
    seg, cfg, seed, keep_seg_diag = args
    r = fit_and_score(seg, cfg, seed)
    if not keep_seg_diag:
        for k in ("seg_tile_err", "seg_best_err", "seg_xy"):
            r.diag.pop(k, None)
    return r.scores, r.diag


def _pool(n_procs):
    return ProcessPoolExecutor(max_workers=n_procs, initializer=_init_worker)


def to_uint8_nhwc(images):
    """Accept uint8 NHWC numpy, list of HxWx3 arrays/PIL, or float torch NCHW in [0,1]."""
    if hasattr(images, "detach"):  # torch tensor
        x = images.detach().float().cpu().clamp(0, 1).numpy()
        if x.ndim == 4 and x.shape[1] in (1, 3):
            x = x.transpose(0, 2, 3, 1)
        return [(xi * 255).round().astype(np.uint8) for xi in x]
    return [np.asarray(im.convert("RGB") if hasattr(im, "convert") else im, dtype=np.uint8) for im in images]


def detect_batch(images, n_procs=8, chunksize=4):
    imgs = to_uint8_nhwc(images)
    if n_procs <= 1:
        return [_seg_one(i) for i in imgs]
    with _pool(n_procs) as ex:
        return list(ex.map(_seg_one, imgs, chunksize=chunksize))


def fit_batch(segs, cfg: VPConfig, seed=0, n_procs=8, keep_seg_diag=False, chunksize=8):
    args = [(s, cfg, seed, keep_seg_diag) for s in segs]
    if n_procs <= 1:
        out = [_fit_one(a) for a in args]
    else:
        with _pool(n_procs) as ex:
            out = list(ex.map(_fit_one, args, chunksize=chunksize))
    scores = {k: np.array([o[0][k] for o in out]) for k in out[0][0]}
    return scores, [o[1] for o in out]


def score_images(images, cfg: VPConfig | None = None, seed=0, n_procs=8, keep_seg_diag=False):
    """images -> (dict of per-image score arrays incl. 'gate', list of per-image diagnostics)."""
    cfg = cfg or VPConfig()
    segs = detect_batch(images, n_procs)
    return fit_batch(segs, cfg, seed, n_procs, keep_seg_diag)


def make_reward_fn(cfg: VPConfig, score_name: str = "RegMean", use_gate: bool = True, seed: int = 0, n_procs: int = 8):
    """DiffusionNFT reward contract: fn(images, prompts, metadata) -> (scores list, info dict)."""

    def _fn(images, prompts, metadata):
        sc, diag = score_images(images, cfg, seed, n_procs)
        s = sc[score_name] * (sc["gate"] if use_gate else 1.0)
        return s.tolist(), {"vp_" + k: v.tolist() for k, v in sc.items()} | {"vp_diag": diag}

    return _fn
