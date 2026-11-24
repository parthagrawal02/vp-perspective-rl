"""Task 13 figure helpers. Every score / overlay comes from the frozen reward code (vp_reward/final.py):
analyze() re-runs exactly final.geometry's fit (stretch -> cached DeepLSD lines -> compute_t8 internals) to expose
per-segment z / assignment / VPs, and asserts its G and Lc equal final.geometry() bit-for-bit."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, textwrap, numpy as np
for p in ["vp_reward", "task2", "task3", "task4", "task6"]:
    if code_dir(p) not in sys.path: sys.path.insert(0, code_dir(p))
from PIL import Image, ImageDraw, ImageFont
from vp_reward import final as F, v2
from vp_reward.v4 import stretch, refine_both, sth_fixed, V8
from vp_reward.core import S

import matplotlib; FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
FONTB = FONT.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
OUT = DATA + "/outputs/report_figs"
Z_IN = 3.0
MINPX = 22  # smallest font used anywhere (px, before any final resize)
SIG = V8["sig"]


def font(px, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, px)


def analyze(img, lines):
    """img uint8 RGB 512, lines = cached DeepLSD output on stretch(img). Returns dict with G (raw, may be NaN), Lc,
    seg (N,4 px, refined), L, z, asg, vps (homog. normalized), n_det."""
    pre = stretch(img)
    _, _, xyu, _, _, _ = refine_both(pre, lines, F.LMIN_FRAC * S)
    out = dict(seg=xyu, L=np.hypot(xyu[:, 2] - xyu[:, 0], xyu[:, 3] - xyu[:, 1]) if len(xyu) else np.zeros(0),
               z=np.full(len(xyu), np.nan), asg=np.zeros(len(xyu), bool), vps=[], G=np.nan, Lc=0.0, n_det=len(lines))
    if len(xyu) >= 2:
        L = out["L"]
        v2._SYS[0] = v2.SIG_SYS
        _, ctx = v2.a1_score(xyu, sth_fixed(L, F.SIGMA0), F.RANSAC_SEED)
        if ctx and ctx["asg"].any():
            z, a = ctx["z"][ctx["asg"]], L[ctx["asg"]]
            w = (a / 100.0) ** 2
            out.update(z=ctx["z"], asg=ctx["asg"], vps=ctx["vps"], Lc=float(a.sum()), G=float(1 - (w * SIG(z)).sum() / w.sum()))
    g, lc = F.geometry(pre, lines)
    assert (np.isnan(g) and np.isnan(out["G"])) or g == out["G"], (g, out["G"])
    assert lc == out["Lc"], (lc, out["Lc"])
    return out


def zcolor(z):
    """green (z=0) -> yellow-green (z<3, consistent) | orange (z=3) -> red (z>=6, violation)."""
    t = float(np.clip(z, 0, 6))
    if t < Z_IN:
        u = t / Z_IN; return (int(20 + 160 * u), int(200 + 20 * u), int(40))
    u = (t - Z_IN) / 3.0; return (255, int(150 * (1 - u)), 0)


def overlay(img, a, scale=2, show_unassigned=True, vp=True):
    """Draw at `scale`x resolution of a 512 image. Returns PIL image (512*scale)."""
    W = S * scale
    im = Image.fromarray(img).resize((W, W), Image.BICUBIC)
    im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.25)  # light wash so lines read
    d = ImageDraw.Draw(im)
    seg, z, asg = a["seg"], a["z"], a["asg"]
    if show_unassigned:
        for s in seg[~asg]:
            d.line([tuple(s[:2] * scale), tuple(s[2:] * scale)], fill=(140, 140, 140), width=max(1, scale))
    order = np.argsort(np.where(asg, z, -1))  # violations drawn on top
    for i in order:
        if asg[i]:
            d.line([tuple(seg[i, :2] * scale), tuple(seg[i, 2:] * scale)], fill=zcolor(z[i]), width=3 * scale)
    if vp:
        for v in a["vps"]:
            draw_vp(d, v, W)
    return im


def gstr(g):
    return f"G = {g:.3f}" if np.isfinite(g) else "G = NaN"


def corner(im, txt, px=26):
    im = im.copy(); label(ImageDraw.Draw(im), (10, 8), txt, px); return im


def label(d, xy, txt, px, fg=(0, 0, 0), bg=(255, 255, 255)):
    f = font(px, True)
    x0, y0, x1, y1 = d.textbbox(xy, txt, font=f)
    d.rectangle([x0 - 6, y0 - 4, x1 + 6, y1 + 4], fill=bg)
    d.text(xy, txt, font=f, fill=fg)


def draw_vp(d, v, W):
    """Yellow cross at a VP inside the frame; else a yellow arrow at the frame edge pointing toward it
    (VP at infinity -> arrows at both ends of the direction)."""
    c = W / 2
    yel, blk = (255, 220, 0), (0, 0, 0)
    if abs(v[2]) > 1e-9:
        p = v[:2] / v[2] * c + c
        if 0 <= p[0] < W and 0 <= p[1] < W:
            r = W * 0.025
            for wd, col in ((int(W * 0.012), blk), (int(W * 0.006), yel)):
                d.line([(p[0] - r, p[1] - r), (p[0] + r, p[1] + r)], fill=col, width=wd)
                d.line([(p[0] - r, p[1] + r), (p[0] + r, p[1] - r)], fill=col, width=wd)
            return
        dirs = [np.sign(v[2]) * v[:2]]
    else:
        dirs = [v[:2], -v[:2]]
    for u in dirs:
        u = u / (np.linalg.norm(u) + 1e-12)
        t = min((c * 0.97) / abs(u[0]) if abs(u[0]) > 1e-9 else 1e9, (c * 0.97) / abs(u[1]) if abs(u[1]) > 1e-9 else 1e9)
        tip = np.array([c, c]) + t * u
        n = np.array([-u[1], u[0]]); Lh = W * 0.06; wh = W * 0.03
        base = tip - Lh * u
        tri = [tuple(tip), tuple(base + wh * n), tuple(base - wh * n)]
        d.polygon(tri, fill=yel, outline=blk, width=max(2, int(W * 0.003)))


def legend_strip(width, px=22):
    """Horizontal legend: z colour bar, grey unassigned, VP cross/arrow."""
    h = int(px * 3.2)
    im = Image.new("RGB", (width, h), "white"); d = ImageDraw.Draw(im); f = font(px)
    x = 10; y = int(px * 0.4)
    d.text((x, y), "segment z:", font=f, fill="black"); x += d.textlength("segment z:", font=f) + 12
    bw = 300
    for k in range(bw):
        d.line([(x + k, y), (x + k, y + px)], fill=zcolor(6 * k / (bw - 1)))
    for zz in (0, 3, 6):
        xx = x + bw * zz / 6
        d.line([(xx, y + px), (xx, y + px + 6)], fill="black", width=2)
        t = {0: "0", 3: "3", 6: "6+"}[zz]; d.text((xx - d.textlength(t, font=f) / 2, y + px + 6), t, font=f, fill="black")
    x += bw + 30
    d.line([(x, y + px // 2), (x + 50, y + px // 2)], fill=(140, 140, 140), width=2); x += 60
    d.text((x, y), "unassigned", font=f, fill="black"); x += d.textlength("unassigned", font=f) + 30
    for wd, col in ((8, (0, 0, 0)), (4, (255, 220, 0))):
        d.line([(x, y), (x + px, y + px)], fill=col, width=wd); d.line([(x, y + px), (x + px, y)], fill=col, width=wd)
    x += px + 10; t = "vanishing point (arrow = outside frame)"; d.text((x, y), t, font=f, fill="black")
    return im


def text_block(txt, width, px=22, bold=False, wrap=True, pad=6, color="black"):
    f = font(px, bold); d0 = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    lines = []
    for para in txt.split("\n"):
        if not wrap: lines.append(para); continue
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w) if cur else w
            if d0.textlength(t, font=f) <= width - 4 or not cur: cur = t
            else: lines.append(cur); cur = w
        lines.append(cur)
    lh = int(px * 1.25)
    im = Image.new("RGB", (width, lh * len(lines) + 2 * pad), "white"); d = ImageDraw.Draw(im)
    for k, l in enumerate(lines):
        assert d.textlength(l, font=f) <= width + 1 or not wrap or " " not in l, l
        d.text((2, pad + k * lh), l, font=f, fill=color)
    return im


def tile(img, size):
    return (img if isinstance(img, Image.Image) else Image.fromarray(img)).resize((size, size), Image.LANCZOS)


def vstack(ims, gap=0):
    W = max(i.width for i in ims); H = sum(i.height for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (W, H), "white"); y = 0
    for i in ims: out.paste(i, (0, y)); y += i.height + gap
    return out


def hstack(ims, gap=0):
    H = max(i.height for i in ims); W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    out = Image.new("RGB", (W, H), "white"); x = 0
    for i in ims: out.paste(i, (x, 0)); x += i.width + gap
    return out


def pad(im, p=16):
    out = Image.new("RGB", (im.width + 2 * p, im.height + 2 * p), "white"); out.paste(im, (p, p)); return out


def save(im, name, caption):
    os.makedirs(OUT, exist_ok=True)
    im = pad(im); k0 = min(1.0, 2400 / im.width)
    if k0 < 1: im = im.resize((2400, round(im.height * k0)), Image.LANCZOS)
    q = 90; fn = f"{OUT}/{name}.jpg"; sc = 1.0
    im.save(fn, quality=q)
    while os.path.getsize(fn) > 1.5e6:  # keep quality 90, shrink until < 1.5 MB
        sc *= 0.92; im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS).save(fn, quality=q)
    open(f"{OUT}/{name}.txt", "w").write(caption.strip() + "\n")
    mp = MINPX * k0 * sc
    print(name, Image.open(fn).size, f"{os.path.getsize(fn)/1e6:.2f} MB", f"min text {mp:.1f}px", "" if mp >= 20 else "WARN<20px", flush=True)
    return fn
