"""Task 14 checks + README.  python check.py  (after build.py export + checksums)"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, json, glob, pickle, numpy as np, pandas as pd
sys.path.insert(0, code_dir("task14"))
from build import PKG, EXT, O, P, SETS
out = {}

# 1. every row -> existing file; every file -> a row
for name, root in (("core", PKG), ("baselines", EXT)):
    D = pd.read_csv(f"{root}/metadata.csv")
    files = {os.path.relpath(os.path.join(d, f), root) for d, _, F in os.walk(root) for f in F if f.endswith(".jpg")}
    rows = set(D.file)
    out[f"{name}_rows"] = len(D); out[f"{name}_files"] = len(files)
    out[f"{name}_rows_without_file"] = len(rows - files); out[f"{name}_files_without_row"] = len(files - rows)
    out[f"{name}_dup_file_rows"] = int(D.file.duplicated().sum())
    cs = [l.split("  ", 1)[1].strip() for l in open(f"{root}/checksums.sha256")]
    out[f"{name}_checksum_entries"] = len(cs); out[f"{name}_checksum_missing"] = len(set(cs) ^ {os.path.relpath(os.path.join(d, f), root) for d, _, F in os.walk(root) for f in F if f != "checksums.sha256"})
    out[f"{name}_G_nan"] = int(D.G.isna().sum())

D = pd.read_csv(f"{PKG}/metadata.csv"); B = pd.read_csv(f"{EXT}/metadata.csv")
# 2. counts vs Task 2/3/11
exp = {"hypersim_select": (998, 3992), "hypersim_fresh": (596, 2384), "nyu": (1449, 5796), "sun": (985, 3940)}
cnt = D.groupby(["set", "role"]).size().unstack()
out["counts"] = {s: dict(reference=int(cnt.loc[s, "reference"]), twin=int(cnt.loc[s, "twin"]), expected=exp[s],
                         ok=bool((cnt.loc[s, "reference"], cnt.loc[s, "twin"]) == exp[s])) for s in exp}
out["counts_split"] = D[D.role == "reference"].groupby(["set", "split"]).size().to_dict().__repr__()
out["baseline_counts"] = B.groupby("variant").size().to_dict()
out["baselines_ok"] = bool((B.groupby("variant").size() == 1192).all() and len(B) == 4768)
# 3. pair ids
cap = [json.loads(l) for l in open(f"{PKG}/captions.jsonl")]
out["dup_pair_ids_captions"] = len(cap) - len({c["pair_id"] for c in cap})
out["pairs_with_not_1ref_4twins"] = int((D.groupby("pair_id").role.apply(lambda r: (r == "reference").sum() != 1 or (r == "twin").sum() != 4)).sum())
out["baseline_pair_ids_not_in_core"] = len(set(B.pair_id) - set(D.pair_id))
out["caption_tokens_max"] = max(c["caption_tokens"] for c in cap); out["caption_tokens_median"] = float(np.median([c["caption_tokens"] for c in cap]))

# 4. 20 random rows with a stored frozen-G score from an earlier task (Task 10 SELECT side, Task 11 fresh split)
g10 = pd.read_pickle(f"{P}/outputs/task10/g10.pkl"); g10 = g10[g10.rseed == 0]; S10 = dict(zip(g10.key.map(str), zip(g10.C_p2_sig, g10.Lc_v2)))
g11 = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{P}/outputs/task11/g11_*.pkl")]); g11 = g11[g11.rseed == 0]; S11 = dict(zip(g11.key.map(str), zip(g11.G, g11.Lc)))
A = pd.concat([D, B])


def stored(r):
    i = int(r.source_id); s = None if pd.isna(r.seed) else int(r.seed)
    if r.set == "hypersim_fresh":
        k = ("m", "real", i, -1, 0.0) if r.role == "reference" else ("m", "base", i, s, 0.0) if r.role == "twin" else ("b", r.variant, i, s)
        return S11.get(str(k))
    ds = SETS[r.set][2]
    k = ("m", ds, ("real", i, -1, 0.0) if r.role == "reference" else ("base", i, s, 0.0))
    return S10.get(str(k))


A["stored"] = [stored(r) for r in A.itertuples()]
has = A[A.stored.notna()]
out["rows_with_stored_score"] = len(has); out["rows_without_stored_score"] = int(A.stored.isna().sum())
smp = has.sample(20, random_state=14)
d = [abs(r.G - r.stored[0]) if not (np.isnan(r.G) and np.isnan(r.stored[0])) else 0.0 for r in smp.itertuples()]
out["check20"] = dict(max_abs_dG=float(max(d)), max_abs_dLc=float(max(abs(r.Lc - r.stored[1]) for r in smp.itertuples())),
                      rows=[f"{r.file}: {r.G:.6f} vs {r.stored[0]:.6f}" for r in smp.itertuples()])
dall = np.array([abs(r.G - r.stored[0]) for r in has.itertuples() if not (np.isnan(r.G) and np.isnan(r.stored[0]))])
out["all_stored_max_abs_dG"] = float(np.nanmax(dall)); out["all_stored_n_mismatch"] = int((dall > 1e-12).sum())
# 5. DeepLSD re-detection on cached images (determinism of the new detection run)
Z = pickle.load(open(f"{O}/scores14.pkl", "rb"))
rc = [(k[1:], v["G"], Z["S"][k[1:]]["G"]) for k, v in Z["RC"].items()]
dd = np.array([abs(a - b) for _, a, b in rc if not (np.isnan(a) and np.isnan(b))])
out["redetect"] = dict(n=len(rc), max_abs_dG=float(dd.max()), n_exact=int((dd == 0).sum()), median_abs_dG=float(np.median(dd)))
# 6. JPEG re-score sensitivity is documented only (no detection on JPEGs here)
json.dump(out, open(f"{O}/checks14.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "check20"}, indent=1, default=str)); print(json.dumps(out["check20"], indent=1))
