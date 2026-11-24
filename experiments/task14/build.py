"""Task 14: package caption-matched pairs (existing images only) as vp_pairs_v1 (+ baselines extras).
  python build.py dlsd <shard> <nshard>   (nft-vp env, GPU) DeepLSD on stretched images that have no cached lines
  python build.py score                   (CPU)  frozen G / Lc / n_kept for every image -> scores14.pkl
  python build.py export                  write folder, metadata, captions, checksums, README
  python build.py check                   checks listed in Task 14
  python build.py zip
Lines: Task 10 cache (SELECT side, all 3 sets), Task 11 cache (fresh split + baselines), else new DeepLSD (frozen config)."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, glob, json, pickle, hashlib, time, numpy as np
P = DATA
for p in ["vp_reward", "DeepLSD", "task13"]: sys.path.insert(0, code_dir(p))
O = f"{P}/outputs/task14"; PKG = f"{O}/vp_pairs_v1"; EXT = f"{O}/vp_pairs_v1_baselines"
SETS = {  # set: (dataset dir, twin dir, Task-10 ds tag)
    "hypersim_select": ("hypersim", f"{P}/outputs/task3/gen_hypersim_512/base_cfg4.5", "hypersim"),
    "hypersim_fresh": ("hypersim_fresh", f"{P}/outputs/task11/gen/base_cfg4.5", None),
    "nyu": ("nyu", f"{P}/outputs/task2/gen/base_cfg4.5", "nyu"),
    "sun": ("sun_kv2", f"{P}/outputs/task3/gen_sun_kv2_512/base_cfg4.5", "sun_kv2")}
BVARS = [("base_suffix_cfg4.5", "SD3.5-M + perspective suffix", 4.5), ("base_cfg1.0", "SD3.5-M (no CFG)", 1.0),
         ("flowgrpo_cfg4.5", "SD3.5-M + Flow-GRPO LoRA (jieliu/SD3.5M-FlowGRPO-PickScore)", 4.5),
         ("nft_cfg1.0", "SD3.5-M + DiffusionNFT LoRA (worstcoder/SD3.5M-DiffusionNFT-MultiReward)", 1.0)]


def meta(ds):
    return {m["id"]: m for m in json.load(open(f"{P}/datasets/{ds}/meta.json"))["items"]}


def caps(ds):
    return {int(k): v for k, v in json.load(open(f"{P}/datasets/{ds}/captions.json"))["captions"].items()}


def records():
    """One dict per image (core + extras)."""
    R = []
    for st, (ds, tw, _) in SETS.items():
        M, C = meta(ds), caps(ds)
        for i in sorted(C):
            pid = f"{st}_{i:04d}"
            R.append(dict(set=st, id=i, role="reference", seed=-1, variant="reference", src=f"{P}/datasets/{ds}/rgb512/{i:04d}.png", pair_id=pid))
            R += [dict(set=st, id=i, role="twin", seed=s, variant="base_cfg4.5", src=f"{tw}/{i:04d}_s{s}.png", pair_id=pid) for s in range(4)]
            if st == "hypersim_fresh":
                R += [dict(set=st, id=i, role="baseline", seed=s, variant=v, src=f"{P}/outputs/task11/gen/{v}/{i:04d}_s{s}.png", pair_id=pid)
                      for v, _, _ in BVARS for s in (0, 1)]
    return R


def rkey(r): return (r["set"], r["id"], r["variant"], r["seed"])


def cached_lines():
    D10, D11 = {}, {}
    for f in sorted(glob.glob(f"{P}/outputs/task10/dlsd10_*.pkl")): D10.update(pickle.load(open(f, "rb")))
    for f in sorted(glob.glob(f"{P}/outputs/task11/dlsd11_*.pkl")): D11.update(pickle.load(open(f, "rb")))
    out = {}
    for r in records():
        st, i, v, s = rkey(r)
        if st == "hypersim_fresh":
            k = ("m", "real", i, -1, 0.0) if v == "reference" else ("m", "base", i, s, 0.0) if v == "base_cfg4.5" else ("b", v, i, s)
            if k in D11: out[rkey(r)] = D11[k]
        else:
            k = ("m", SETS[st][2], ("real", i, -1, 0.0) if v == "reference" else ("base", i, s, 0.0))
            if k in D10: out[rkey(r)] = D10[k]
    return out


def rgb(p):
    from PIL import Image
    return np.array(Image.open(p).convert("RGB"))


def stage_dlsd(shard, n):
    from vp_reward.final import load_deeplsd, detect_lines
    from vp_reward.v4 import stretch
    from concurrent.futures import ThreadPoolExecutor
    C = cached_lines(); recs = records()
    todo = [r for r in recs if rkey(r) not in C]
    rng = np.random.default_rng(14); recheck = [recs[j] for j in rng.choice(len(recs), 400, replace=False) if rkey(recs[j]) in C][:40]
    J = (todo + recheck)[shard::n]; net = load_deeplsd("cuda"); out = {}; t0 = time.time()
    print("todo", len(todo), "recheck", len(recheck), "this shard", len(J), flush=True)
    with ThreadPoolExecutor(8) as ex:
        for m, (r, pre) in enumerate(zip(J, ex.map(lambda r: stretch(rgb(r["src"])), J))):
            out[rkey(r)] = detect_lines(net, pre)
            if m % 1000 == 0: print(shard, m, len(J), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"{O}/dlsd14_{shard}.pkl", "wb")); print("DLSD14_DONE", shard, len(out), f"{time.time()-t0:.0f}s", flush=True)


def _score(a):
    import cv2; cv2.setNumThreads(1)
    from figlib import analyze
    k, src, lines = a; x = analyze(rgb(src), lines)
    return k, dict(G=x["G"], Lc=x["Lc"], n_kept=len(x["seg"]), n_detected=len(lines))


def stage_score():
    from concurrent.futures import ProcessPoolExecutor
    C = cached_lines(); N = {}
    for f in sorted(glob.glob(f"{O}/dlsd14_*.pkl")): N.update(pickle.load(open(f, "rb")))
    recs = records(); jobs = []; src_of = {}
    for r in recs:
        k = rkey(r); src_of[k] = "task10/11 cache" if k in C else "task14 DeepLSD"
        jobs.append((k, r["src"], C[k] if k in C else N[k]))
    t0 = time.time()
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 64))) as ex: S = dict(ex.map(_score, jobs, chunksize=16))
    # recheck: freshly detected lines on cached images -> G
    rc = [(("recheck",) + k, r, N[k]) for r in recs for k in [rkey(r)] if k in C and k in N]
    with ProcessPoolExecutor(32) as ex: RC = dict(ex.map(_score, [(k, r["src"], l) for k, r, l in rc]))
    pickle.dump(dict(S=S, src=src_of, RC=RC), open(f"{O}/scores14.pkl", "wb"))
    print("SCORE_DONE", len(S), "recheck", len(RC), f"{time.time()-t0:.0f}s", flush=True)


def ntok_fn():
    from transformers import CLIPTokenizer
    toks = [CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder=s) for s in ("tokenizer", "tokenizer_2")]
    return lambda t: max(len(tk(t, add_special_tokens=False).input_ids) for tk in toks)  # same counter as task3/caption.py


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _jpg(a):
    from PIL import Image
    src, dst = a; os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src).convert("RGB"); assert im.size == (512, 512), (src, im.size)
    im.save(dst, quality=95); return dst


def stage_export():
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    Z = pickle.load(open(f"{O}/scores14.pkl", "rb")); S = Z["S"]
    ntok = ntok_fn(); rows = []; capl = []
    code_hash = {f: sha(f"{code_dir('vp_reward')}/vp_reward/{f}") for f in ("final.py", "v4.py", "v2.py", "core.py")}
    for st, (ds, _, _) in SETS.items():
        M, C = meta(ds), caps(ds)
        for i in sorted(C):
            m = M[i]; pid = f"{st}_{i:04d}"
            capl.append(dict(pair_id=pid, set=st, caption=C[i], caption_tokens=ntok(C[i])))
    CT = {c["pair_id"]: c["caption_tokens"] for c in capl}
    for r in records():
        st, i = r["set"], r["id"]; ds = SETS[st][0]; m = meta(ds)[i]; C = caps(ds)
        split = {"hypersim_fresh": "fresh"}.get(st, m.get("split"))
        orig = {"nyu": f"nyu_depth_v2_labeled.mat index {i} (0-based)"}.get(st) or m["src"].split("/datasets/")[1].split("/", 1)[1].replace("raw/", "")
        if r["role"] == "reference":
            fn = "reference.jpg"; model = "none (real data)" if st in ("nyu", "sun") else "none (Hypersim render)"; cfg = steps = np.nan
        elif r["role"] == "twin":
            fn = f"twin_s{r['seed']}.jpg"; model = "stabilityai/stable-diffusion-3.5-medium"; cfg, steps = 4.5, 40
        else:
            v = r["variant"]; fn = f"{v}_s{r['seed']}.jpg"; model = dict((a, b) for a, b, _ in BVARS)[v]; cfg = dict((a, c) for a, _, c in BVARS)[v]; steps = 40
        sc = S[rkey(r)]
        rows.append(dict(pair_id=r["pair_id"], set=st, split=split, scene=m["scene"], source_id=i, source_path=orig,
                         role=r["role"], reference_kind=("render" if st.startswith("hypersim") else "photo"), variant=r["variant"],
                         model=model, cfg=cfg, steps=steps, seed=(r["seed"] if r["role"] != "reference" else np.nan),
                         prompt_suffix=(", architectural photography, straight lines, correct perspective" if r["variant"] == "base_suffix_cfg4.5" else ""),
                         file=f"images/{st}/{r['pair_id']}/{fn}", caption=C[i], caption_tokens=CT[r["pair_id"]],
                         G=sc["G"], Lc=sc["Lc"], n_segments_kept=sc["n_kept"], n_segments_detected=sc["n_detected"],
                         lines_source=Z["src"][rkey(r)], _src=r["src"]))
    D = pd.DataFrame(rows)
    D["subset"] = [meta("hypersim_fresh")[i].get("subset", "") if s == "hypersim_fresh" else "" for s, i in zip(D.set, D.source_id)]
    for root, sub in ((PKG, D[D.role != "baseline"]), (EXT, D[D.role == "baseline"])):
        os.makedirs(root, exist_ok=True)
        with ProcessPoolExecutor(48) as ex: list(ex.map(_jpg, [(s, f"{root}/{f}") for s, f in zip(sub._src, sub.file)], chunksize=32))
        sub.drop(columns=["_src", "lines_source"]).to_csv(f"{root}/metadata.csv", index=False)
    with open(f"{PKG}/captions.jsonl", "w") as f:
        for c in capl: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    json.dump(dict(code_sha256=code_hash, lines_source=D.groupby(["set", "lines_source"]).size().to_dict().__repr__()), open(f"{O}/provenance.json", "w"), indent=1)
    D.to_pickle(f"{O}/metadata_full.pkl"); print("EXPORT_DONE", len(D), flush=True)


def checksums(root):
    from concurrent.futures import ThreadPoolExecutor
    fs = sorted(os.path.relpath(os.path.join(d, f), root) for d, _, F in os.walk(root) for f in F if f != "checksums.sha256")
    with ThreadPoolExecutor(32) as ex: H = list(ex.map(lambda f: sha(f"{root}/{f}"), fs))
    open(f"{root}/checksums.sha256", "w").write("".join(f"{h}  {f}\n" for h, f in zip(H, fs)))


def stage_jpegtest(n=200):
    import pandas as pd
    from vp_reward.final import load_deeplsd, detect_lines, geometry
    from vp_reward.v4 import stretch
    D = pd.read_csv(f"{PKG}/metadata.csv").sample(n, random_state=141); net = load_deeplsd("cuda"); d = []
    for r in D.itertuples():
        pre = stretch(rgb(f"{PKG}/{r.file}")); g, _ = geometry(pre, detect_lines(net, pre))
        if np.isfinite(g) and np.isfinite(r.G): d.append(g - r.G)
    d = np.array(d); a = np.abs(d)
    J = dict(n=len(d), mean=float(d.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(a.max()))
    json.dump(J, open(f"{O}/jpeg_rescore.json", "w"), indent=1); print("JPEGTEST", J)


if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method("forkserver", force=True)
    a = sys.argv; os.makedirs(O, exist_ok=True)
    if a[1] == "dlsd": stage_dlsd(int(a[2]), int(a[3]))
    elif a[1] == "score": stage_score()
    elif a[1] == "export": stage_export()
    elif a[1] == "jpegtest": stage_jpegtest()
    elif a[1] == "checksums": checksums(PKG); checksums(EXT); print("CHECKSUMS_DONE")
