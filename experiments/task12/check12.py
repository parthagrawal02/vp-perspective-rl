"""Task 12 offline checks.
  python check12.py wiring <run_dir>   recompute R with vp_reward.final on the dumped epoch-0 samples, compare with logged values
  python check12.py grids <run_dir>    4x4 grids of the first and last epoch dumps
  python check12.py lcstats            Lc_base distribution vs Lc of the CFG 4.5 twins on the fresh split"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, json, glob, multiprocessing, numpy as np
for p in ["DiffusionNFT", "vp_reward", "DeepLSD"]: sys.path.insert(0, code_dir(p))
LC = f"{code_dir('DiffusionNFT')}/dataset/vp_hypersim/lc_base.json"


def wiring(run):
    from PIL import Image
    from vp_reward.final import perspective_reward
    recs = json.load(open(f"{run}/samples/epoch_0000/logged.json"))
    ims = np.stack([np.array(Image.open(r["file"]).convert("RGB")) for r in recs]); P = [r["prompt"] for r in recs]
    fn = perspective_reward("cuda", lc_base=json.load(open(LC)), n_workers=8)
    Rv, info = fn(ims, P, None)
    pairs = {"R (avg)": ("avg", Rv), "R (vp_perspective)": ("vp_perspective", Rv), "zG": ("vp_zG", info["zG"]), "G_raw": ("vp_G_raw", info["G_raw"]),
             "P": ("vp_P", info["P"]), "Lc": ("vp_Lc", info["Lc"]), "Lc_base": ("vp_Lc_base", info["Lc_base"]),
             "PickScore": ("vp_pickscore", info["pickscore"]), "aesthetic": ("vp_aesthetic", info["aesthetic"]), "zQ": ("vp_zQ", info["zQ"])}
    rows = []
    for name, (k, v) in pairs.items():
        lg = np.array([r[k] for r in recs], float); d = np.abs(lg - np.asarray(v, float))
        rows.append((name, len(recs), float(np.nanmax(d)), float(np.nanmax(np.abs(lg)))))
        print(f"{name:20s} n={len(recs)} max|logged - offline| = {np.nanmax(d):.3g}   (max |value| {np.nanmax(np.abs(lg)):.3g})")
    json.dump(rows, open(f"{run}/wiring_check.json", "w"), indent=1)


def grids(run):
    from PIL import Image
    eps = sorted(glob.glob(f"{run}/samples/epoch_*"))
    for e in (eps[0], eps[-1]):
        fs = sorted(glob.glob(f"{e}/[0-9][0-9].png"))[:16]; ims = [Image.open(f).convert("RGB").resize((256, 256)) for f in fs]
        g = Image.new("RGB", (1024, 1024), "white")
        for j, im in enumerate(ims): g.paste(im, ((j % 4) * 256, (j // 4) * 256))
        out = f"{run}/grid_{os.path.basename(e)}.jpg"; g.save(out, quality=90); print(out)


def lcstats():
    import pandas as pd
    lc = pd.Series(json.load(open(LC))); pm = json.load(open(f"{code_dir('DiffusionNFT')}/dataset/vp_hypersim/prompt_map.json"))
    sp = lc.index.map(lambda p: pm[p]["split"])
    T = pd.read_pickle(DATA + "/outputs/task11/table11_data.pkl")
    tw = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)].groupby("id").Lc.mean()  # CFG 4.5, 40 steps, 4 twins
    D = pd.read_csv(DATA + "/outputs/task12/lcbase_all.csv")
    q = lambda x: {k: round(float(v), 0) for k, v in x.quantile([.05, .25, .5, .75, .95]).items()}
    print("Lc_base (10-step dpm2, CFG 1.0, mean of 4) train:", len(lc[sp == "train"]), "mean", round(lc[sp == "train"].mean()), q(lc[sp == "train"]))
    print("Lc_base test:", len(lc[sp == "test"]), "mean", round(lc[sp == "test"].mean()), q(lc[sp == "test"]))
    test = lc[sp == "test"]; ids = test.index.map(lambda p: pm[p]["id"]); a = pd.Series(test.values, index=ids); b = tw.reindex(a.index)
    print("fresh-split CFG 4.5 / 40-step twins, mean of 4:", "mean", round(b.mean()), q(b))
    r = (a / b).replace([np.inf, -np.inf], np.nan).dropna()
    print("per prompt ratio rollout / CFG4.5:", "median", round(float(r.median()), 3), "IQR", [round(float(x), 3) for x in r.quantile([.25, .75])],
          "Spearman", round(float(a.corr(b, method="spearman")), 3))
    print("per-sample: G NaN rate", round(float(D.G.isna().mean()), 4), "| Lc==0 share", round(float((D.Lc == 0).mean()), 4),
          "| within-prompt CV of Lc median", round(float((D.groupby("prompt").Lc.std(ddof=0) / D.groupby("prompt").Lc.mean()).median()), 3))
    real = T[(T.kind == "m") & (T.set == "real") & (T.rseed == 0)].set_index("id").Lc
    print("floor activity with the new Lc_base (alpha 0.8): real Hypersim (test)", round(float((real.reindex(a.index) < 0.8 * a).mean()), 3),
          "| CFG4.5 twins s0-1", round(float(pd.concat([(T[(T.kind == 'm') & (T.set == 'base') & (T.rseed == 0) & (T.seed == s)].set_index('id').Lc.reindex(a.index) < 0.8 * a) for s in (0, 1)]).mean()), 3))
    # rollout samples own floor activity (leave-one-out within the 4)
    g = D.groupby("prompt").Lc; loo = (g.transform("sum") - D.Lc) / 3
    print("rollout samples, P>0 with leave-one-out Lc_base:", round(float((D.Lc < 0.8 * loo).mean()), 3), "| with full mean:", round(float((D.Lc < 0.8 * D.prompt.map(lc)).mean()), 3))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    {"wiring": lambda: wiring(sys.argv[2]), "grids": lambda: grids(sys.argv[2]), "lcstats": lcstats}[sys.argv[1]]()
