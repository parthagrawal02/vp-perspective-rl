"""Task 15: evaluate the experimental v2 reward (vp_reward/reward_v2.py) against the frozen one on vp_pairs_v1 + probes.

    python experiments/task15/eval15.py fit      [--root ~/Downloads/vp_pairs_v1] [--n-per-set 80] [--n-fresh 120]
    python experiments/task15/eval15.py synth    # bend-angle sweep and split-VP probe (DeepLSD on rendered scenes)
    python experiments/task15/eval15.py densify  # add K lines toward each twin's own VPs; frozen vs v2 reward parts
    python experiments/task15/eval15.py report   # discrimination per BETA + G_b standardization constants

Images: the JPEG q95 copies in vp_pairs_v1 (the stored metadata G was computed on PNGs), so every variant here, including the
frozen one (BETA = 0), is re-scored on the same JPEG pixels: compare like with like. Outputs: $VP_DATA/outputs/task15/.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from vp_reward import final as F
from vp_reward import reward_v2 as V
from vp_reward import viz
from vp_reward.paths import DATA

OUT = os.path.join(DATA, "outputs", "task15")
BETAS = (0.0, 0.25, 0.5, 1.0)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_meta(root):
    rows = [r for r in csv.DictReader(open(os.path.join(root, "metadata.csv"))) if r["role"] in ("reference", "twin")]
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["pair_id"]].append(r)
    return pairs


def cmd_fit(a):
    pairs = load_meta(a.root)
    rnd = random.Random(0)
    by = defaultdict(list)
    for pid, rs in pairs.items():
        by[(rs[0]["set"], rs[0]["split"])].append(pid)
    chosen = []
    for s in ("hypersim_select", "nyu", "sun"):
        chosen += [(p, "SELECT") for p in rnd.sample(sorted(by[(s, "SELECT")]), a.n_per_set)]
    chosen += [(p, "fresh") for p in rnd.sample(sorted(by[("hypersim_fresh", "fresh")]), a.n_fresh)]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fits.pkl")
    fits = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    dev = device(); net = F.load_deeplsd(dev)
    todo = [(pid, sp, r) for pid, sp in chosen for r in pairs[pid] if r["file"] not in fits]
    print(f"{len(chosen)} pairs, {len(todo)} images to fit on {dev}", flush=True)
    t0 = time.time()
    for k, (pid, sp, r) in enumerate(todo):
        img = viz._load(os.path.join(a.root, r["file"]))
        pre = F.stretch(img)
        lines = F.detect_lines(net, pre, dev)
        with np.errstate(all="ignore"):
            f = V.fit(pre, lines)
        f.update(pair_id=pid, split=sp, set=r["set"], role=r["role"], seed=-1 if r["role"] == "reference" else int(float(r["seed"])),
                 file=r["file"], G_stored=float(r["G"]) if r["G"] else np.nan, n_raw=len(lines))
        fits[r["file"]] = f
        if (k + 1) % 100 == 0 or k + 1 == len(todo):
            pickle.dump(fits, open(path, "wb"))
            print(f"  {k + 1}/{len(todo)}  {(time.time() - t0) / (k + 1):.2f}s/img", flush=True)
    # sanity: BETA = 0 must equal the frozen geometry on the same pixels
    r = next(iter(pairs[chosen[0][0]]))
    img = viz._load(os.path.join(a.root, r["file"])); pre = F.stretch(img)
    with np.errstate(all="ignore"):
        g0 = F.geometry(pre, F.detect_lines(net, pre, dev)); g1 = V.score(fits[r["file"]], 0.0)
    assert np.allclose(g0, g1, equal_nan=True), (g0, g1)
    print("BETA=0 reproduces final.geometry:", g0)


def table(fits, beta):
    rows = []
    for f in fits.values():
        G, Lc = V.score(f, beta)
        rows.append(dict(id=f["pair_id"], split=f["split"], ds=f["set"], set="real" if f["role"] == "reference" else "base",
                         seed=f["seed"], rseed=0, Lc=Lc, G=G))
    return pd.DataFrame(rows)


def cmd_report(a):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eval import metrics as M
    fits = pickle.load(open(os.path.join(OUT, "fits.pkl"), "rb"))
    out, consts = [], {}
    for b in BETAS:
        df = table(fits, b)
        sel_tw = df[(df.split == "SELECT") & (df.set == "base")].G
        consts[b] = (float(sel_tw.mean()), float(sel_tw.std()), float(sel_tw.quantile(0.01)))
        for sp in ("SELECT", "fresh"):
            for ds in sorted(df[df.split == sp].ds.unique()) + ["all"]:
                d = df[(df.split == sp) & ((df.ds == ds) if ds != "all" else True)]
                pw = M.paired_win_rate(d, "G"); wr, tie = M.untied_win_rate(d, "G"); q = M.lc_quintile_win_rates(d, "G")
                ln = d[d.set == "base"].groupby("id").G.std().median() / max(d[d.set == "base"].G.std(), 1e-9)
                out.append(dict(beta=b, split=sp, ds=ds, n_pairs=pw["n_pairs"], wr_untied=wr, p=pw["p"], q_above=q["q_above"],
                                within_caption_std_over_total=ln, nan_rate=float(d.G.isna().mean())))
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(OUT, "discrimination15.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(R[R.ds == "all"].to_string(index=False)); print(); print(R[R.ds != "all"].pivot_table(index=["split", "ds"], columns="beta", values="wr_untied"))
    print("\nG_b standardization on SELECT twins (JPEG; mean, std, 1st pct):")
    for b, c in consts.items():
        print(f"  {b}: {c}")
    pickle.dump(consts, open(os.path.join(OUT, "g_stats.pkl"), "wb"))


def _render_scene(groups, seed=0):
    from vp_reward.synth import render
    return render(groups, seed)


def cmd_synth(a):
    from vp_reward.synth import _segments_toward, bend, scene
    dev = device(); net = F.load_deeplsd(dev)

    def fitimg(img):
        pre = F.stretch(img)
        with np.errstate(all="ignore"):
            return V.fit(pre, F.detect_lines(net, pre, dev))

    rows = []
    for d in (0, 1, 2, 4, 8, 12, 16, 20, 30, 45):
        for s in (2, 3, 4):
            f = fitimg(_render_scene(bend(scene("one", seed=s), d)[0]))
            for b in BETAS:
                rows.append(dict(probe="bend", x=d, scene=s, beta=b, G=V.score(f, b)[0], Lc=V.score(f, b)[1]))
    rng = np.random.default_rng(5)
    vert = _segments_toward(("inf", 90.0), 40, rng)
    left = _segments_toward((256.0, 230.0), 30, rng, region=lambda x, y: x < 256)
    for shift in (0, 40, 120, 250):
        right = _segments_toward((256.0 + shift, 230.0 - shift / 2), 30, np.random.default_rng(7), region=lambda x, y: x >= 256)
        f = fitimg(_render_scene([vert, left, right]))
        for b in BETAS:
            rows.append(dict(probe="splitvp", x=shift, scene=0, beta=b, G=V.score(f, b)[0], Lc=V.score(f, b)[1]))
    D = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); D.to_csv(os.path.join(OUT, "synth15.csv"), index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("bend sweep (mean G over 3 scenes):"); print(D[D.probe == "bend"].pivot_table(index="x", columns="beta", values="G"))
        print("\nsplit VP (one structure, right half aimed at a displaced VP):"); print(D[D.probe == "splitvp"].pivot_table(index="x", columns="beta", values="G"))


def densify(img, vps_px_dirs, K, rng):
    """Draw K dark anti-aliased segments that point exactly at the image's own fitted VPs (finite or at infinity)."""
    import cv2
    out = img.copy()
    S = img.shape[0]
    for _ in range(K):
        px, d = vps_px_dirs[rng.integers(len(vps_px_dirs))]
        x, y = rng.uniform(30, S - 30, 2)
        if px is not None:
            u = np.array(px) - (x, y); n = np.linalg.norm(u)
            if n < 40:
                continue
            u /= n
        else:
            u = np.array(d)
        L = rng.uniform(40, 110)
        p1, p2 = np.array([x, y]) - u * L / 2, np.array([x, y]) + u * L / 2
        col = int(rng.uniform(20, 70))
        cv2.line(out, tuple(int(round(v * 16)) for v in p1), tuple(int(round(v * 16)) for v in p2), (col, col, col), 2, cv2.LINE_AA, shift=4)
    return out


def cmd_densify(a):
    fits = pickle.load(open(os.path.join(OUT, "fits.pkl"), "rb"))
    by = defaultdict(dict)
    for f in fits.values():
        by[f["pair_id"]][f["seed"]] = f
    pids = sorted(p for p, d in by.items() if all(s in d for s in range(4)))
    pids = random.Random(1).sample(pids, min(a.n, len(pids)))
    dev = device(); net = F.load_deeplsd(dev); rng = np.random.default_rng(0)
    beta = V.BETA
    if beta not in V.G_STATS:
        V.G_STATS.update(pickle.load(open(os.path.join(OUT, "g_stats.pkl"), "rb")))
    rows = []
    for pid in pids:
        f = by[pid][0]
        others = [V.score(by[pid][s], 0.0)[1] for s in (1, 2, 3)]
        Lb = float(np.mean(others))
        img = viz._load(os.path.join(a.root, f["file"]))
        with np.errstate(all="ignore"):
            an = viz.analyse(img, lambda pre: F.detect_lines(net, pre, dev))
        vpd = [viz.vp_pixel(v) for v in an["vps"]]
        if not vpd:
            continue
        for K in (0, 20, 60, 150):
            im2 = img if K == 0 else densify(img, [(None if p is None else p.tolist(), dd.tolist()) for p, dd in vpd], K, rng)
            pre = F.stretch(im2)
            with np.errstate(all="ignore"):
                ff = V.fit(pre, F.detect_lines(net, pre, dev))
            G0, Lc = V.score(ff, 0.0); Gb, _ = V.score(ff, beta)
            R1, i1 = F.combine([G0], [Lc], [Lb], [F.PICK_MEAN], [F.AES_MEAN])
            R2, i2 = V.combine([Gb], [Lc], [Lb], [F.PICK_MEAN], [F.AES_MEAN], beta=beta)
            rows.append(dict(pair_id=pid, K=K, Lc=Lc, Lc_base=Lb, G_frozen=G0, R_frozen=R1[0], G_v2=Gb, P=i2["P"][0], P_hi=i2["P_hi"][0], R_v2=R2[0]))
        print(pid, flush=True)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "densify15.csv"), index=False)
    base = D[D.K == 0].set_index("pair_id")
    D["dR_frozen"] = D.R_frozen - D.pair_id.map(base.R_frozen); D["dR_v2"] = D.R_v2 - D.pair_id.map(base.R_v2)
    D["dG_frozen"] = D.G_frozen - D.pair_id.map(base.G_frozen)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        g = D.groupby("K")
        print(pd.DataFrame(dict(Lc_ratio=g.apply(lambda x: (x.Lc / x.Lc_base).median()), dG_frozen=g.dG_frozen.mean(),
                                dR_frozen=g.dR_frozen.mean(), frac_R_frozen_up=g.apply(lambda x: (x.dR_frozen > 0).mean()),
                                dR_v2=g.dR_v2.mean(), frac_R_v2_up=g.apply(lambda x: (x.dR_v2 > 0).mean()), P_hi=g.P_hi.mean())))
    print("(quality term held at its mean: zQ = 0 for both)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "synth", "densify", "report"])
    ap.add_argument("--root", default=os.path.expanduser("~/Downloads/vp_pairs_v1"))
    ap.add_argument("--n-per-set", type=int, default=80)
    ap.add_argument("--n-fresh", type=int, default=120)
    ap.add_argument("--n", type=int, default=60, help="densify: number of pairs")
    a = ap.parse_args()
    dict(fit=cmd_fit, synth=cmd_synth, densify=cmd_densify, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()
