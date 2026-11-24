"""Score images with the frozen perspective reward (FINAL.md). Thin wrapper around vp_reward.final; no numbers change.

    python -m vp_reward.score IMG_OR_DIR [...] [--prompt TEXT | --prompts map.json] [--lc-base X | --lc-base-json map.json]
                              [--device cuda] [--resize] [--csv out.csv]

Always printed:   G (NaN-filled per FINAL.md §2.9), G_nan flag, zG, Lc (assigned segment length, px), n_lines (DeepLSD)
With a prompt:    PickScore (x26 logit), aesthetic, zQ               (PickScore is prompt-conditional, so no prompt -> no Q)
With Lc_base:     P (floor penalty)
With both:        R = zG - 4 P + 0.25 zQ

Images must be 512x512 RGB (the reward is defined at 512). --resize center-crops to a square and resizes to 512
(vp_reward.core.center_square_resize, the same preprocessing as every reference image in this project).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from . import final as F
from .paths import add_nft_to_path

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
COLS = ["G", "G_nan", "zG", "Lc", "n_lines", "pickscore", "aesthetic", "zQ", "Lc_base", "P", "R"]


def _load(x, resize):
    from PIL import Image
    from .core import center_square_resize
    a = np.array(Image.open(x).convert("RGB")) if isinstance(x, (str, os.PathLike)) else np.asarray(x)
    if a.shape[:2] != (512, 512):
        if not resize:
            raise ValueError(f"{x if isinstance(x, str) else 'image'}: {a.shape[1]}x{a.shape[0]}; the reward is defined at 512x512 (use --resize)")
        a = center_square_resize(a)
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
    return a


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        else:
            out.append(p)
    return out


class Scorer:
    """Loads DeepLSD once (and the PickScore / aesthetic models on first use)."""

    def __init__(self, device=None, n_workers=0):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = F.load_deeplsd(self.device)
        self.pool = None
        if n_workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("forkserver"))
        self._q = None

    def quality_models(self):
        if self._q is None:
            import torch
            add_nft_to_path()
            from flow_grpo.pickscore_scorer import PickScoreScorer
            from flow_grpo.aesthetic_scorer import AestheticScorer
            self._q = (PickScoreScorer(device=self.device, dtype=torch.float32), AestheticScorer(dtype=torch.float32, device=self.device))
        return self._q

    def __call__(self, images, prompts=None, lc_base=None, resize=False, chunk=32):
        """images: paths or uint8 HxWx3 arrays. prompts: list (or None). lc_base: float, list, or None.
        Returns dict of numpy arrays (keys COLS); entries that need a missing input are NaN."""
        ims = np.stack([_load(x, resize) for x in images])
        n = len(ims)
        G, Lc, nl = np.empty(n), np.empty(n), np.empty(n)
        for b in range(0, n, chunk):
            sl = slice(b, b + chunk)
            pres = [F.stretch(x) for x in ims[sl]]
            lines = [F.detect_lines(self.net, p, self.device) for p in pres]
            geo = list(self.pool.map(F._geo_job, zip(pres, lines))) if self.pool else [F.geometry(p, l) for p, l in zip(pres, lines)]
            G[sl] = [g for g, _ in geo]; Lc[sl] = [l for _, l in geo]; nl[sl] = [len(l) for l in lines]
        pick, aes = np.full(n, np.nan), np.full(n, np.nan)
        if prompts is not None:
            ps, ae = self.quality_models()
            for b in range(0, n, chunk):
                pick[b:b + chunk], aes[b:b + chunk] = F.quality_batch(ps, ae, ims[b:b + chunk], list(prompts[b:b + chunk]))
        Lb = np.full(n, np.nan) if lc_base is None else np.broadcast_to(np.asarray(lc_base, float), (n,)).copy()
        R, info = F.combine(G, Lc, Lb, pick, aes)
        P = np.where(np.isnan(Lb), np.nan, info["P"])       # combine() treats a missing Lc_base as "no floor"; report it as unknown
        R = np.where(np.isnan(Lb) | np.isnan(info["zQ"]), np.nan, R)
        return dict(G=info["G"], G_nan=info["G_nan"], zG=info["zG"], Lc=Lc, n_lines=nl, pickscore=pick, aesthetic=aes, zQ=info["zQ"],
                    Lc_base=Lb, P=P, R=R)


def score_images(images, prompts=None, lc_base=None, device=None, resize=False, n_workers=0):
    """One-shot convenience: returns a pandas DataFrame with one row per image (columns COLS)."""
    import pandas as pd
    out = Scorer(device, n_workers)(images, prompts, lc_base, resize)
    df = pd.DataFrame(out)[COLS]
    df["n_lines"] = df.n_lines.astype(int)
    df.insert(0, "image", [x if isinstance(x, str) else f"#{i}" for i, x in enumerate(images)])
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vp_reward.score", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files and/or directories")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--prompt", help="one prompt for every image (enables PickScore / aesthetic / zQ)")
    g.add_argument("--prompts", help="JSON {filename or basename: prompt}")
    h = ap.add_mutually_exclusive_group()
    h.add_argument("--lc-base", type=float, help="Lc_base for every image (mean Lc of base SD3.5-M samples of the prompt)")
    h.add_argument("--lc-base-json", help="JSON {prompt: Lc_base} (the RL format, e.g. dataset/vp_hypersim/lc_base.json)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resize", action="store_true", help="center-crop + resize non-512 images to 512x512")
    ap.add_argument("--workers", type=int, default=0, help="process pool for the CPU geometry fit (0 = in-process)")
    ap.add_argument("--csv", help="also write the table here")
    a = ap.parse_args(argv)
    files = expand(a.images)
    if not files:
        sys.exit("no images found")
    prompts = None
    if a.prompt is not None:
        prompts = [a.prompt] * len(files)
    elif a.prompts:
        m = json.load(open(a.prompts))
        prompts = [m.get(f, m.get(os.path.basename(f))) for f in files]
        miss = [f for f, p in zip(files, prompts) if p is None]
        if miss:
            sys.exit(f"--prompts has no entry for {len(miss)} image(s), e.g. {miss[0]}")
    lb = a.lc_base
    if a.lc_base_json:
        if prompts is None:
            sys.exit("--lc-base-json is keyed by prompt; give --prompt/--prompts")
        m = json.load(open(a.lc_base_json)); lb = [m[p] for p in prompts]
    df = score_images(files, prompts, lb, a.device, a.resize, a.workers)
    import pandas as pd
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        shown = df.dropna(axis=1, how="all") if len(df) else df
        print(shown.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False)
    return df


if __name__ == "__main__":
    main()
