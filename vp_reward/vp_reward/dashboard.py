"""Local dashboard for a vp_pairs dataset: browse reference/twin pairs, the frozen reward's parts per image, and the
fitted lines + vanishing points. Serves only on 127.0.0.1 (the dataset is private; nothing is uploaded).

    python -m vp_reward.dashboard ~/Downloads/vp_pairs_v1 [--port 8765] [--quality q.csv] [--detector auto|deeplsd|lsd]

Reward parts (FINAL.md):  R = z(G) - 4 P + 0.25 zQ
  G, Lc    stored values from metadata.csv (DeepLSD, the real reward).
  Lc_base  validation rule of FINAL.md section 3: twins -> mean Lc of the pair's other 3 twins (leave-one-out),
           reference -> mean Lc of all 4 twins.
  zQ       needs PickScore / aesthetic (GPU scorers, not in metadata.csv). Pass --quality CSV with columns
           file,pickscore,aesthetic (file as in metadata.csv; pickscore = the x26 logit) to fill it in; else zQ is
           unknown and R is shown as z(G) - 4P ("R without quality").
Overlay: lines are re-detected locally (DeepLSD when available, else OpenCV LSD as a stand-in) and replayed through the
frozen geometry (viz.analyse), so the overlay's G can differ from the stored G when the stand-in detector is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import final as F
from . import viz

HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_pairs(root, quality=None):
    rows = list(csv.DictReader(open(os.path.join(root, "metadata.csv"), newline="")))
    q = {}
    if quality:
        for r in csv.DictReader(open(quality, newline="")):
            q[r["file"]] = (_f(r.get("pickscore")), _f(r.get("aesthetic")))
    pairs = {}
    for r in rows:
        if r["role"] not in ("reference", "twin"):
            continue
        p = pairs.setdefault(r["pair_id"], dict(pair_id=r["pair_id"], set=r["set"], split=r["split"], subset=r.get("subset") or "",
                                                scene=r["scene"], caption=r["caption"], ref=None, twins=[]))
        im = dict(file=r["file"], role=r["role"], seed=None if r["role"] == "reference" else int(float(r["seed"])),
                  G=_f(r["G"]), Lc=_f(r["Lc"]) or 0.0, n_kept=int(float(r["n_segments_kept"] or 0)),
                  n_det=int(float(r["n_segments_detected"] or 0)))
        im["pickscore"], im["aesthetic"] = q.get(r["file"], (None, None))
        if r["role"] == "reference":
            p["ref"] = im
        else:
            p["twins"].append(im)
    for p in pairs.values():
        p["twins"].sort(key=lambda t: t["seed"])
        tl = [t["Lc"] for t in p["twins"]]
        for im in ([p["ref"]] if p["ref"] else []) + p["twins"]:
            if im["role"] == "twin":
                others = [t["Lc"] for t in p["twins"] if t is not im]
                im["Lc_base"] = float(np.mean(others)) if others else None
            else:
                im["Lc_base"] = float(np.mean(tl)) if tl else None
            G = np.nan if im["G"] is None else im["G"]
            Lb = np.nan if im["Lc_base"] is None else im["Lc_base"]
            hasq = im["pickscore"] is not None and im["aesthetic"] is not None
            R, info = F.combine([G], [im["Lc"]], [Lb], [im["pickscore"] if hasq else F.PICK_MEAN], [im["aesthetic"] if hasq else F.AES_MEAN])
            im.update(G_nan=bool(info["G_nan"][0]), G_used=float(info["G"][0]), zG=float(info["zG"][0]), P=float(info["P"][0]),
                      zQ=float(info["zQ"][0]) if hasq else None)
            im["R_noQ"] = im["zG"] - F.LAMBDA_F * im["P"]
            im["R"] = float(R[0]) if hasq else None
    return [p for p in pairs.values() if p["ref"]]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class App:
    def __init__(self, root, quality, detector):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.pairs = load_pairs(self.root, quality)
        self.pairs_json = json.dumps(dict(pairs=self.pairs, quality=bool(quality), const=dict(
            G_MEAN=F.G_MEAN, G_STD=F.G_STD, ALPHA=F.ALPHA, LAMBDA_F=F.LAMBDA_F, LAMBDA_Q=F.LAMBDA_Q, G_NAN_FILL=F.G_NAN_FILL,
            PICK_MEAN=F.PICK_MEAN, PICK_STD=F.PICK_STD, AES_MEAN=F.AES_MEAN, AES_STD=F.AES_STD))).encode()
        self.det_name, self.det = viz.get_detector(detector)
        self.lock = threading.Lock()
        self.cache = {}

    def path(self, rel):
        p = os.path.realpath(os.path.join(self.root, rel))
        if not p.startswith(self.root + os.sep) or not os.path.isfile(p):
            raise FileNotFoundError(rel)
        return p

    def analyse(self, rel):
        if rel in self.cache:
            return self.cache[rel]
        img = viz._load(self.path(rel))
        with self.lock, np.errstate(all="ignore"):
            a = viz.analyse(img, self.det)
        vps = []
        for k, v in enumerate(a["vps"]):
            px, d = viz.vp_pixel(v)
            vps.append(dict(px=None if px is None else px.tolist(), dir=d.tolist()))
        out = dict(detector=self.det_name, raw=a["raw"], xy=a["xy"], vps=vps, G=a["G"], Lc=a["Lc"])
        if "z" in a:
            w = (a["L"] / 100.0) ** 2
            out.update(L=a["L"], arg=a["arg"], asg=a["asg"], z=a["z"], err=a["best"],
                       viol=np.where(a["asg"], w * viz.F_sig(a["z"]), 0.0), wsum=float(w[a["asg"]].sum()))
        out = _clean(out)
        if len(self.cache) > 512:
            self.cache.pop(next(iter(self.cache)))
        self.cache[rel] = out
        return out


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if ctype.startswith("image/"):
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    return self._send(200, open(HTML, "rb").read(), "text/html; charset=utf-8")
                if u.path == "/api/pairs":
                    return self._send(200, app.pairs_json, "application/json")
                if u.path == "/api/analyse":
                    rel = parse_qs(u.query)["file"][0]
                    return self._send(200, json.dumps(app.analyse(rel)).encode(), "application/json")
                if u.path.startswith("/img/"):
                    return self._send(200, open(app.path(unquote(u.path[5:])), "rb").read(), "image/jpeg")
                self._send(404, b"not found", "text/plain")
            except (FileNotFoundError, KeyError) as e:
                self._send(404, str(e).encode(), "text/plain")
            except Exception as e:  # noqa: BLE001
                self._send(500, f"{type(e).__name__}: {e}".encode(), "text/plain")

        def log_message(self, *a):
            pass

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vp_reward.dashboard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="vp_pairs dataset folder (contains metadata.csv and images/)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--quality", help="CSV file,pickscore,aesthetic to fill in zQ and the full R")
    ap.add_argument("--detector", default="auto", choices=["auto", "deeplsd", "lsd"])
    a = ap.parse_args(argv)
    app = App(a.root, a.quality, a.detector)
    print(f"{len(app.pairs)} pairs from {app.root}; overlay lines: {app.det_name}")
    print(f"open http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app)).serve_forever()


if __name__ == "__main__":
    main()
