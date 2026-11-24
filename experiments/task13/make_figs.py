"""Task 13: report figures from existing images/lines only (no generation, no detection).
  python make_figs.py stats      -> outputs/report_figs/e2_long_short.csv + data13.pkl (per-image fresh-split analyses)
  python make_figs.py A1 A2 B C D E1 E2 F G   (any subset)
Lines: cached stretched-DeepLSD outputs (Task 11 fresh split: dlsd11_*.pkl; Task 10 SELECT NYU/SUN: dlsd10_*.pkl).
Scores: vp_reward.final (G/Lc via figlib.analyze == final.geometry, asserted; R via the Task 11 table = final.combine)."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, code_dir("task13")); sys.path.insert(0, code_dir("task11"))
from figlib import *
from PIL import Image, ImageDraw

P = DATA; O11 = f"{P}/outputs/task11"; FR = f"{P}/datasets/hypersim_fresh"
CAP = json.load(open(f"{FR}/captions.json"))["captions"]
TW = "SD3.5-M twin (CFG 4.5, 40 steps, 512 px)"
_DL = {}


def DL():
    if not _DL:
        for f in sorted(glob.glob(f"{O11}/dlsd11_*.pkl")): _DL.update(pickle.load(open(f, "rb")))
    return _DL


def T11():
    fn = f"{OUT}/t11_table.pkl"
    if not os.path.exists(fn):
        import eval11; os.makedirs(OUT, exist_ok=True); eval11.build().to_pickle(fn)
    T = pd.read_pickle(fn); return T[T.rseed == 0].copy()


def img(k):
    import pipe11; return pipe11.get_img(k)


def ana(k):
    return analyze(img(k), DL()[k])


def realk(i): return ("m", "real", i, -1, 0.0)
def basek(i, s): return ("m", "base", i, s, 0.0)


def Gcap(t, px=24):  # text strip under an image
    return text_block(t, t_w[0], px, wrap=False)


t_w = [480]


def under(im, txt, px=24, bold=False):
    return vstack([im, text_block(txt, im.width, px, bold, wrap=False, pad=4)])


def head(txt, w, px=26):
    return text_block(txt, w, px, True)


# ------------------------------------------------------------------ A1
def figA1():
    import pipe6
    rng = np.random.default_rng(1301); TS = 450
    ids = sorted(int(k) for k in CAP); hyp = sorted(rng.choice(ids, 4, replace=False).tolist())
    D10 = {}
    for f in sorted(glob.glob(f"{P}/outputs/task10/dlsd10_*.pkl")): D10.update(pickle.load(open(f, "rb")))
    other = []
    for ds in ("nyu", "sun_kv2"):
        cand = sorted({k[2][1] for k in D10 if k[0] == "m" and k[1] == ds and k[2][0] == "real"
                       and all(("m", ds, ("base", k[2][1], s, 0.0)) in D10 for s in (0, 1, 2))})
        other.append((ds, int(rng.choice(cand)), len(cand)))
    rows, notes = [], []
    cols = ["Reference photo", "Twin, seed 0", "Twin, seed 1", "Twin, seed 2"]
    rows.append(hstack([head(c, TS) for c in cols], 12))
    for src, i in [("hf", i) for i in hyp] + [(ds, i) for ds, i, _ in other]:
        if src == "hf":
            ks = [realk(i)] + [basek(i, s) for s in (0, 1, 2)]; ims = [img(k) for k in ks]; As = [ana(k) for k in ks]
            cap = CAP[str(i)]; tag = f"Hypersim (fresh split) #{i}"
        else:
            ks = [("m", src, ("real", i, -1, 0.0))] + [("m", src, ("base", i, s, 0.0)) for s in (0, 1, 2)]
            ims = [pipe6.get_img(k) for k in ks]; As = [analyze(x, D10[k]) for x, k in zip(ims, ks)]
            cap = json.load(open(f"{P}/datasets/{src}/captions.json"))["captions"][str(i)]
            tag = {"nyu": "NYU Depth v2", "sun_kv2": "SUN RGB-D (Kinect v2)"}[src] + f" (SELECT split) #{i}"
        notes.append((tag, [a["G"] for a in As]))
        tiles = [under(tile(x, TS), gstr(a["G"]), 26, True) for x, a in zip(ims, As)]
        rows.append(vstack([head(tag, 4 * TS + 36, 24), hstack(tiles, 12), text_block('Caption: "' + cap + '"', 4 * TS + 36, 22)]))
    fig = vstack(rows, 26)
    gl = "\n".join(f"  {t}: ref G {g[0]:.3f}; twins {', '.join(f'{x:.3f}' for x in g[1:])}" for t, g in notes)
    cap = f"""A1. Caption-matched pairs, typical examples (no overlays).
Each row: a real reference photo, then three SD3.5-Medium images generated from that photo's VLM caption (CFG 4.5, 40 steps, 512 px, seeds 0-2). G (frozen perspective-consistency score, vp_reward/final.py; higher = line segments more consistent with fitted vanishing points) is printed under each image; the caption text is under each row.
Selection (not by hand): numpy default_rng(1301); 4 Hypersim ids drawn uniformly from the 596-image fresh split, then 1 NYU and 1 SUN id drawn uniformly from the SELECT-split images that have stretched-DeepLSD lines cached for the reference and twin seeds 0-2 (NYU {other[0][2]} candidates, SUN {other[1][2]}). The fresh split was never used for tuning; the NYU/SUN SELECT images were used for reward selection.
G values:
{gl}"""
    save(fig, "A1_pairs_typical", cap)


# ------------------------------------------------------------------ helpers for overlay pairs
def ov_tile(k, a, TS, txt=None):
    return corner(tile(overlay(img(k), a), TS), txt or gstr(a["G"]))


def fig_pairs(name, pairs, titles, cap, TS=560, per_row=2):
    """pairs: list of (keyL, keyR, labelL, labelR, header)."""
    blocks = []
    for kL, kR, lL, lR, hdr in pairs:
        aL, aR = ana(kL), ana(kR)
        b = hstack([under(ov_tile(kL, aL, TS), lL, 24), under(ov_tile(kR, aR, TS), lR, 24)], 10)
        blocks.append(vstack([text_block(hdr, b.width, 22), b]))
    rows = [hstack(blocks[r:r + per_row], 40) for r in range(0, len(blocks), per_row)]
    W = max(r.width for r in rows)
    save(vstack([head(titles, W, 28)] + rows + [legend_strip(W)], 24), name, cap)


# ------------------------------------------------------------------ A2
def figA2():
    T = T11()
    r = T[(T.kind == "m") & (T.set == "real")].set_index("id").G
    b = T[(T.kind == "m") & (T.set == "base")][["id", "seed", "G"]].copy()
    b["gap"] = b.id.map(r) - b.G; b = b.dropna(subset=["gap"]).sort_values("gap", ascending=False)
    top = b.drop_duplicates("id").head(4)
    pairs = [(realk(i), basek(i, s), "Reference photo", f"Twin, seed {s}", f'#{i}  gap {g:+.3f}:  "{CAP[str(i)][:150]}{"..." if len(CAP[str(i)]) > 150 else ""}"')
             for i, s, g in zip(top.id, top.seed, top.gap)]
    allg = b.gap
    cap = f"""A2. Caption-matched pairs with the LARGEST reference-minus-twin G gap (rule-picked, not typical).
Rule: over all {len(allg)} (reference, twin) pairs of the 596-image Hypersim fresh split (twins: CFG 4.5, 40 steps, seeds 0-3, NaN G excluded), sort by G(reference) - G(twin) and take the top 4 distinct captions. These are the extreme tail: the median gap over all pairs is {allg.median():+.3f} (mean {allg.mean():+.3f}; reference wins {(allg > 0).mean():.1%}).
Overlays: DeepLSD segments kept by the reward (length >= 4% of the image), coloured by their z-score to the assigned vanishing point (green z<3 consistent -> red violation; G down-weights segments by sigmoid((z-3)/0.5), weight length^2); thin grey = not assigned to any VP (ignored by G); yellow cross = fitted VP inside the frame, yellow arrow = VP outside the frame (direction). G in the top-left corner.
Pairs (id, twin seed, G ref, G twin, gap): """ + "; ".join(f"#{i} s{s} {r[i]:.3f} vs {gt:.3f} ({g:+.3f})" for i, s, gt, g in zip(top.id, top.seed, top.G, top.gap))
    fig_pairs("A2_pairs_largest_gap", pairs, "Largest G gaps, reference vs twin (rule: top 4 G(ref) - G(twin), fresh split)", cap)


# ------------------------------------------------------------------ B
def figB():
    rng = np.random.default_rng(1302); i = int(rng.choice(sorted(int(k) for k in CAP)))
    TS = 1060; out = []; info = []
    for k, lab in ((realk(i), "Reference photo (Hypersim)"), (basek(i, 0), "SD3.5-M twin, seed 0")):
        a = ana(k); out.append(under(ov_tile(k, a, TS), lab, 28, True))
        n_as = int(a["asg"].sum()); viol = a["asg"] & (a["z"] > 3)
        info.append(f"{lab}: G {a['G']:.3f}; {a['n_det']} DeepLSD segments, {len(a['seg'])} kept (length >= 20.5 px), {n_as} assigned to {len(a['vps'])} VPs; "
                    f"assigned length Lc {a['Lc']:.0f} px, of which {a['L'][viol].sum() / max(a['Lc'], 1e-9):.1%} has z > 3")
    row = hstack(out, 20)
    fig = vstack([head(f'What the reward sees. Hypersim fresh #{i}: "{CAP[str(i)]}"', row.width, 26), row, legend_strip(row.width, 24)], 12)
    cap = f"""B. What the reward sees: one Hypersim reference photo and its SD3.5-M twin (seed 0, CFG 4.5, 40 steps), with the full geometry overlay.
Selection: numpy default_rng(1302), one id drawn uniformly from the 596-image fresh split (#{i}); twin seed fixed at 0.
Pipeline (frozen, vp_reward/final.py): grey + 1-99th percentile contrast stretch -> DeepLSD (wireframe weights) -> keep segments >= 4% of 512 px -> per-segment refit -> sequential RANSAC for up to 3 VPs -> each segment assigned to its best VP if within 15 deg -> z = angular error / sigma(length) -> G = 1 - sum w*sigmoid((z-3)/0.5) / sum w, w = length^2.
Colours: assigned segments green (z<3, consistent) -> red (violation); thin grey = unassigned (ignored by G); yellow cross = VP in frame, yellow arrow = VP outside the frame (direction toward it).
""" + "\n".join(info)
    save(fig, "B_what_reward_sees", cap)


# ------------------------------------------------------------------ C
def figC():
    import pipe11
    from imageops import choose_window, WIN
    T = T11(); rng = np.random.default_rng(1303); i = int(rng.choice(sorted(int(k) for k in CAP)))
    k0, k2 = realk(i), ("m", "real_rot", i, -1, 2.0)
    a0, a2 = ana(k0), ana(k2)
    (x0, y0), _ = choose_window(pipe11.lsd_segs(img(k0)))
    TS = 1000; sc = TS / 512; ts = []
    for k, a, lab in ((k0, a0, "Original"), (k2, a2, "Region rotated 2 deg")):
        t = ov_tile(k, a, TS); d = ImageDraw.Draw(t)
        d.rectangle([x0 * sc, y0 * sc, (x0 + WIN) * sc, (y0 + WIN) * sc], outline=(0, 120, 255), width=5)
        ts.append(under(t, f"{lab}: G = {a['G']:.3f}", 28, True))
    row = hstack(ts, 20)
    R = T[(T.kind == "m") & (T.set.isin(["real", "real_rot"]))]
    g0 = R[R.set == "real"].set_index("id").G; g2 = R[(R.set == "real_rot") & (R.theta == 2.0)].set_index("id").G
    dg = (g2 - g0).dropna()
    fig = vstack([head(f"Rotation test, Hypersim fresh #{i} (blue box = rotated region, {WIN}x{WIN} px, feathered edges)", row.width, 26), row, legend_strip(row.width, 24)], 12)
    cap = f"""C. Rotation test: one real Hypersim photo vs the same photo with one region rotated by 2 deg (rule-free pick).
Selection: numpy default_rng(1303), one id drawn uniformly from the 596-image fresh split (#{i}). The region (blue box, {WIN}x{WIN} px of 512, raised-cosine feathered edge) is chosen by the fixed Task-3 rule: the window holding the most LSD segment length; its content is rotated about its centre and blended back.
G original {a0['G']:.3f} -> rotated {a2['G']:.3f} (change {a2['G'] - a0['G']:+.3f}).
Context, all {len(dg)} fresh-split images at 2 deg: mean change in G {dg.mean():+.4f}, median {dg.median():+.4f}, G drops for {(dg < 0).mean():.1%}.
Overlay colours as in Fig. B (green z<3 -> red violation; grey unassigned; yellow cross/arrow = VP)."""
    save(fig, "C_rotation_test", cap)


# ------------------------------------------------------------------ D
def figD():
    import pipe11
    T = T11(); rng = np.random.default_rng(1304); i = int(rng.choice(pipe11.ex_ids()))
    ko, kb = ("x", i, "orig", None), ("x", i, "blur", 2.0)
    TS = 1000; ts = []; info = []
    X = T[(T.kind == "x") & (T.id == i)].set_index("pert")
    for k, lab, pk in ((ko, "Twin (original)", "orig"), (kb, "Gaussian blur, sigma 2 px", "blur 2")):
        a = ana(k); r = X.loc[pk]
        assert abs(r.G - a["G"]) < 1e-12 or (np.isnan(r.G) and np.isnan(a["G"]))
        txt = f"{lab}\n{a['n_det']} segments detected, {len(a['seg'])} kept; Lc {a['Lc']:.0f} px\nG {a['G']:.3f}   R {r.R:+.3f}  (zG {r.zG:+.2f}, floor P {r.P:.2f}, zQ {r.zQ:+.2f})"
        ts.append(vstack([tile(overlay(img(k), a), TS), text_block(txt, TS, 26, True, wrap=False)]))
        info.append((lab, a["n_det"], len(a["seg"]), a["Lc"], a["G"], r.R, r.zG, r.P, r.zQ, r.Lc_base))
    row = hstack(ts, 20)
    E = T[(T.kind == "x")].pivot_table(index="id", columns="pert", values=["G", "R", "Lc"])
    dG = (E["G"]["blur 2"] - E["G"]["orig"]).dropna(); dR = (E["R"]["blur 2"] - E["R"]["orig"]).dropna(); rL = (E["Lc"]["blur 2"] / E["Lc"]["orig"]).dropna()
    fig = vstack([head(f'Why degradation fools a line-fraction score. Twin of Hypersim fresh #{i} (seed 0)', row.width, 26), row, legend_strip(row.width, 24)], 12)
    cap = f"""D. Why degradation fools a line-fraction score: one SD3.5-M twin vs the same image blurred (Gaussian sigma 2 px), detected lines overlaid.
Selection: numpy default_rng(1304), one id drawn uniformly from the 500 fresh-split exploit-test ids (#{i}); seed-0 twin (CFG 4.5, 40 steps).
Blur removes weak / textured segments; the survivors are long, clean edges that fit the VPs well, so a score that only measures the consistent FRACTION of line length (G) can go up. The frozen reward counters this with the structure floor P = max(0, 1 - Lc / (0.8 Lc_base)) (Lc_base = mean Lc of the caption's other base twins, here {info[0][9]:.0f} px) plus the quality term zQ.
""" + "\n".join(f"{l}: {nd} segments detected, {nk} kept (>= 20.5 px), Lc {lc:.0f} px, G {g:.3f}, R {R:+.3f} (zG {zg:+.2f}, P {p:.2f}, zQ {zq:+.2f})" for l, nd, nk, lc, g, R, zg, p, zq, _ in info) + f"""
Context, all {len(dG)} exploit ids, blur 2 minus original: G {dG.mean():+.4f} (mean), Lc ratio {rL.median():.2f} (median), R {dR.mean():+.3f} (mean; R includes the floor and quality terms)."""
    save(fig, "D_blur_fools_fraction", cap)


# ------------------------------------------------------------------ E1
VAR = [("base CFG 4.5", lambda i: basek(i, 0)), ("+ perspective suffix", lambda i: ("b", "base_suffix_cfg4.5", i, 0)),
       ("base, no CFG (1.0)", lambda i: ("b", "base_cfg1.0", i, 0)), ("Flow-GRPO (PickScore)", lambda i: ("b", "flowgrpo_cfg4.5", i, 0)),
       ("DiffusionNFT (multi-reward)", lambda i: ("b", "nft_cfg1.0", i, 0))]


def figE1():
    rng = np.random.default_rng(1305); ids = sorted(rng.choice(sorted(int(k) for k in CAP), 4, replace=False).tolist())
    TS = 450; rows = [hstack([head(n, TS, 24) for n, _ in VAR], 10)]; gl = []
    for i in ids:
        As = [(f(i), ana(f(i))) for _, f in VAR]
        r = hstack([under(tile(img(k), TS), gstr(a["G"]), 26, True) for k, a in As], 10)
        rows.append(vstack([r, text_block(f'#{i}: "{CAP[str(i)]}"', r.width, 22)]))
        gl.append(f"  #{i}: " + ", ".join(f"{n} {a['G']:.3f}" for (n, _), (_, a) in zip(VAR, As)))
    cap = f"""E1. Baselines, typical examples. Same caption and seed (0) across columns.
Columns: SD3.5-M base (CFG 4.5, 40 steps) | base with the perspective prompt suffix appended (CFG 4.5) | base without CFG (CFG 1.0) | Flow-GRPO LoRA (jieliu/SD3.5M-FlowGRPO-PickScore, CFG 4.5) | DiffusionNFT LoRA (worstcoder/SD3.5M-DiffusionNFT-MultiReward, CFG 1.0). All 40 steps, 512 px. G under each image.
Selection: numpy default_rng(1305), 4 ids drawn uniformly from the 596-image Hypersim fresh split.
G values:
""" + "\n".join(gl)
    save(vstack(rows, 22), "E1_baselines_typical", cap)


# ------------------------------------------------------------------ stats (E2 long/short) - per-image analyses over the fresh split
def _job(kl):
    import cv2; cv2.setNumThreads(1)
    k, lines = kl; a = analyze(img(k), lines); return k, dict(G=a["G"], Lc=a["Lc"], L=a["L"][a["asg"]], z=a["z"][a["asg"]], n_kept=len(a["seg"]))


def stats():
    from concurrent.futures import ProcessPoolExecutor
    ids = sorted(int(k) for k in CAP)
    K = [basek(i, s) for i in ids for s in (0, 1)] + [("b", "flowgrpo_cfg4.5", i, s) for i in ids for s in (0, 1)]
    DL()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 48))) as ex: res = dict(ex.map(_job, [(k, DL()[k]) for k in K], chunksize=8))
    pickle.dump(res, open(f"{OUT}/e2_segments.pkl", "wb"))
    T = T11()
    for k, r in res.items():  # consistency with the Task 11 table (same frozen code)
        t = T[T.key.map(str) == str(k)].G.values[0]
        assert (np.isnan(t) and np.isnan(r["G"])) or abs(t - r["G"]) < 1e-12, (k, t, r["G"])
    allL = np.concatenate([r["L"] for r in res.values()]); med = float(np.median(allL))
    rows = []
    for i in ids:
        for s in (0, 1):
            b, f = res[basek(i, s)], res[("b", "flowgrpo_cfg4.5", i, s)]
            d = dict(id=i, seed=s, G_base=b["G"], G_fg=f["G"])
            for tag, r in (("base", b), ("fg", f)):
                v = r["z"] > 3; lo = r["L"] <= med; w = (r["L"] / 100) ** 2; sg = SIG(r["z"])
                d.update({f"viol_long_{tag}": r["L"][v & ~lo].sum(), f"viol_short_{tag}": r["L"][v & lo].sum(),
                          f"asg_long_{tag}": r["L"][~lo].sum(), f"asg_short_{tag}": r["L"][lo].sum(),
                          f"pen_long_{tag}": (w * sg)[~lo].sum(), f"w_long_{tag}": w[~lo].sum(), f"w_short_{tag}": w[lo].sum(), f"pen_short_{tag}": (w * sg)[lo].sum(),
                          f"nviol_long_{tag}": int((v & ~lo).sum()), f"nviol_short_{tag}": int((v & lo).sum())})
            rows.append(d)
    D = pd.DataFrame(rows); D.to_csv(f"{OUT}/e2_pairs.csv", index=False)
    S_ = {}
    for q in ("viol", "pen", "asg"):
        xl = (D[f"{q}_long_fg"] - D[f"{q}_long_base"]).sum(); xs = (D[f"{q}_short_fg"] - D[f"{q}_short_base"]).sum()
        S_[q] = dict(extra_long=xl, extra_short=xs, share_long=xl / (xl + xs) if (xl + xs) != 0 else np.nan,
                     base_long=D[f"{q}_long_base"].sum(), base_short=D[f"{q}_short_base"].sum(),
                     fg_long=D[f"{q}_long_fg"].sum(), fg_short=D[f"{q}_short_fg"].sum())
    # per-pair bootstrap CI of share_long (violating length)
    rng = np.random.default_rng(0); bs = []
    el = (D.viol_long_fg - D.viol_long_base).values; es = (D.viol_short_fg - D.viol_short_base).values
    for _ in range(2000):
        j = rng.integers(0, len(D), len(D)); bs.append(el[j].sum() / (el[j].sum() + es[j].sum()))
    S_["viol"]["share_long_ci95"] = tuple(np.percentile(bs, [2.5, 97.5]))
    for tag in ("base", "fg"):
        for c in ("long", "short"):
            S_[f"violfrac_{c}_{tag}"] = D[f"viol_{c}_{tag}"].sum() / D[f"asg_{c}_{tag}"].sum()  # violating share of assigned length
            S_[f"penfrac_{c}_{tag}"] = D[f"pen_{c}_{tag}"].sum() / D[f"w_{c}_{tag}"].sum()  # G-weighted penalty rate
        S_[f"wshare_long_{tag}"] = D[f"w_long_{tag}"].sum() / (D[f"w_long_{tag}"].sum() + D[f"w_short_{tag}"].sum())
        S_[f"meanG_{tag}"] = float(D[f"G_{tag}"].mean())
    S_["median_len_px"] = med; S_["n_pairs"] = len(D); S_["dG_mean"] = float((D.G_fg - D.G_base).mean())
    S_["n_viol_segs"] = {c: int(D[c].sum()) for c in D if c.startswith("nviol")}
    json.dump(S_, open(f"{OUT}/e2_long_short.json", "w"), indent=1, default=float); print(json.dumps(S_, indent=1, default=float))


# ------------------------------------------------------------------ E2
def figE2():
    D = pd.read_csv(f"{OUT}/e2_pairs.csv"); S_ = json.load(open(f"{OUT}/e2_long_short.json"))
    D["drop"] = D.G_base - D.G_fg; top = D.dropna(subset=["drop"]).sort_values("drop", ascending=False).drop_duplicates("id").head(4)
    pairs = [(basek(i, s), ("b", "flowgrpo_cfg4.5", i, s), f"Base CFG 4.5, seed {s}", f"Flow-GRPO, seed {s}",
              f'#{i}  drop {dd:.3f}:  "{CAP[str(i)][:150]}{"..." if len(CAP[str(i)]) > 150 else ""}"') for i, s, dd in zip(top.id, top.seed, top["drop"])]
    v, p = S_["viol"], S_["pen"]; ci = v["share_long_ci95"]
    cap = f"""E2. Flow-GRPO's largest G drops vs base SD3.5-M (rule-picked, not typical).
Rule: over all {S_['n_pairs']} (caption, seed) pairs of the 596-image Hypersim fresh split, seeds 0 and 1, base (CFG 4.5) vs Flow-GRPO LoRA (CFG 4.5), same caption and seed; sort by G(base) - G(Flow-GRPO), take the top 4 distinct captions. Overlays as in Fig. B.
Pairs (id, seed, G base, G Flow-GRPO): """ + "; ".join(f"#{i} s{s} {gb:.3f} vs {gf:.3f}" for i, s, gb, gf in zip(top.id, top.seed, top.G_base, top.G_fg)) + f"""
Mean G change over all pairs (Flow-GRPO - base): {S_['dG_mean']:+.4f}.
Long vs short (is the drop just decorative detail?): median length of all assigned segments (both models pooled) = {S_['median_len_px']:.1f} px. Violating = assigned with z > 3. Summed over all pairs, Flow-GRPO has {v['extra_long'] + v['extra_short']:+.0f} px more violating length than base; {v['share_long']:.1%} of that extra comes from segments LONGER than the median (bootstrap 95% CI over pairs {ci[0]:.1%}-{ci[1]:.1%}), {1 - v['share_long']:.1%} from shorter ones. Total violating length long/short: base {v['base_long']:.0f}/{v['base_short']:.0f} px, Flow-GRPO {v['fg_long']:.0f}/{v['fg_short']:.0f} px.
But that absolute split misses what moves G, because G is a RATIO and Flow-GRPO images have less long structure: assigned length in long segments base {S_['asg']['base_long']:.0f} -> Flow-GRPO {S_['asg']['fg_long']:.0f} px, short {S_['asg']['base_short']:.0f} -> {S_['asg']['fg_short']:.0f} px. Violating share of assigned length: long segments {S_['violfrac_long_base']:.1%} (base) -> {S_['violfrac_long_fg']:.1%} (Flow-GRPO); short segments {S_['violfrac_short_base']:.1%} -> {S_['violfrac_short_fg']:.1%}. With G's own weighting (length^2, soft z penalty) long segments carry {S_['wshare_long_base']:.1%} (base) / {S_['wshare_long_fg']:.1%} (Flow-GRPO) of the weight, and their penalty rate rises {S_['penfrac_long_base']:.3f} -> {S_['penfrac_long_fg']:.3f} (short: {S_['penfrac_short_base']:.3f} -> {S_['penfrac_short_fg']:.3f}). So the G drop comes from long segments (a larger violating fraction among fewer long lines), not from extra short decorative detail."""
    fig_pairs("E2_flowgrpo_largest_drops", pairs, "Flow-GRPO's largest G drops vs base (rule: top 4 G(base) - G(Flow-GRPO), same caption + seed)", cap)


# ------------------------------------------------------------------ F
def figF():
    T = T11(); rng = np.random.default_rng(1306); ids = sorted(rng.choice(sorted(int(k) for k in CAP), 3, replace=False).tolist())
    B = T[(T.kind == "m") & (T.set == "base")]
    TS = 540; blocks = []; gl = []
    for i in ids:
        b = B[B.id == i].sort_values("R"); hi, lo = b.iloc[-1], b.iloc[0]
        ts = [under(tile(img(basek(i, int(r.seed))), TS), f"{lab}, seed {int(r.seed)}:  G {r.Gf:.3f}  R {r.R:+.2f}", 24, True) for lab, r in (("Best R", hi), ("Worst R", lo))]
        rr = hstack(ts, 10); blocks.append(vstack([rr, text_block(f'#{i}: "{CAP[str(i)]}"', rr.width, 22)]))
        gl.append(f"  #{i}: R of the 4 twins (seed: R / G) " + ", ".join(f"s{int(r.seed)}: {r.R:+.2f} / {r.Gf:.3f}" for _, r in b.sort_values("seed").iterrows()))
    fig = vstack(blocks, 30)
    cap = f"""F. Variation within one prompt: for 3 captions, the best and worst of the 4 base SD3.5-M twins (CFG 4.5, 40 steps, seeds 0-3) ranked by the frozen reward R, side by side with G and R.
Selection: numpy default_rng(1306), 3 ids drawn uniformly from the 596-image Hypersim fresh split. R = z(G) - 4 P + 0.25 zQ (vp_reward/final.py) with Lc_base = mean Lc of the caption's other 3 base twins (Task 11 convention).
Note: best-of-4 picked by R is optimistic - the same R that ranks the images also scores them, so the best image's R is biased upward (selection on noise); it shows the spread available to RL within one prompt, not an achievable improvement.
""" + "\n".join(gl)
    save(fig, "F_within_prompt_variation", cap)


# ------------------------------------------------------------------ G
def figG():
    RD = f"{P}/outputs/task12/nft_logs/nft_sd3_vp_2026.11.17_19.13.59"
    st = pd.read_csv(f"{RD}/vp_epoch_stats.csv").set_index("epoch")
    ims = []
    for e in (0, 2):
        g = Image.open(f"{RD}/grid_epoch_{e:04d}.jpg").convert("RGB"); W = 1170; g = g.resize((W, round(g.height * W / g.width)), Image.LANCZOS)
        r = st.loc[e]
        ims.append(vstack([head(f"Epoch {e}", W, 28), g, text_block(f"epoch mean (1152 samples): Lc {r.Lc_mean:.0f} px, G {r.G_raw_mean:.3f}, R {r.R_mean:+.2f}, floor active {r.floor_activity:.1%}", W, 24, wrap=True)]))
    fig = hstack(ims, 30)
    cap = f"""G. DiffusionNFT smoke test with the frozen perspective reward (Task 12; 3 epochs, 4 GPUs, run nft_sd3_vp_2026.11.17_19.13.59): the existing first (epoch 0) and last (epoch 2) training-sample grids (the first 16 rank-0 rollout samples of the epoch; 10-step dpm2 sampler, CFG 1.0, 512 px).
Epoch means over all 1152 rollout samples: epoch 0 Lc {st.loc[0].Lc_mean:.0f} px, G {st.loc[0].G_raw_mean:.3f}, R {st.loc[0].R_mean:+.3f}; epoch 2 Lc {st.loc[2].Lc_mean:.0f} px, G {st.loc[2].G_raw_mean:.3f}, R {st.loc[2].R_mean:+.3f}.
This is a plumbing check only (3 optimizer steps, different prompts each epoch); it is not evidence of learning and should not be read as a trend."""
    save(fig, "G_smoke_test_grids", cap)


if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method("forkserver", force=True)
    for a in sys.argv[1:]:
        {"stats": stats, "A1": figA1, "A2": figA2, "B": figB, "C": figC, "D": figD, "E1": figE1, "E2": figE2, "F": figF, "G": figG}[a]()
