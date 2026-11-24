"""Fresh Hypersim validation split (Task 11): frames never used before; subsets 'newscene' (13 scenes never used) and
'reportscene' (new frames of REPORT scenes, >= 15 frame ids from any downloaded frame). The frozen selection is
datasets/hypersim_fresh/selection.json: it depends on what was already on disk, so do not re-run the selection.

Origin: experiments/task11/fresh_dl.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA
import sys, os, csv, json, glob, zipfile, collections, requests, numpy as np
from concurrent.futures import ThreadPoolExecutor
H = DATA + "/datasets/hypersim"; OUT = f"{H}/raw"; F = DATA + "/datasets/hypersim_fresh"
os.makedirs(F, exist_ok=True)
sys.path.insert(0, f"{H}/ml-hypersim/contrib/99991"); import download as dl
pub = collections.defaultdict(list)
for r in csv.DictReader(open(f"{H}/ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv")):
    if r["included_in_public_release"] == "True": pub[r["scene_name"]].append((r["camera_name"], int(r["frame_id"])))
M = json.load(open(f"{H}/meta.json"))["items"]; used = {m["src"].split("/raw/")[1] for m in M}; split = {m["scene"]: m["split"] for m in M}
have = sorted(p.split("raw/", 1)[1] for p in glob.glob(f"{OUT}/*/images/*/*.tonemap.jpg"))
rel = lambda s, c, f: f"{s}/images/scene_{c}_final_preview/frame.{f:04d}.tonemap.jpg"
parse = lambda p: (p.split("/")[0], p.split("/")[2][len("scene_"):-len("_final_preview")], int(p.split("frame.")[1][:4]))
rng = np.random.default_rng(11); sel = []
newsc = sorted(s for s in pub if s not in split)
for s in newsc:
    cams = sorted({c for c, _ in pub[s]}); fr = {c: sorted(f for cc, f in pub[s] if cc == c) for c in cams}
    pick = [(parse(p)[1], parse(p)[2]) for p in have if p.startswith(s + "/") and p not in used]
    per = max(1, int(np.ceil(20 / len(cams))))
    for c in cams:
        for q in np.linspace(0, 1, per + 2)[1:-1]:
            f = fr[c][int(q * (len(fr[c]) - 1))]
            if all(not (cc == c and abs(ff - f) < 15) for cc, ff in pick): pick.append((c, f))
    sel += [dict(subset="newscene", scene=s, cam=c, frame=f) for c, f in pick[:20]]
for s in sorted(k for k, v in split.items() if v == "REPORT"):
    dlh = [parse(p) for p in have if p.startswith(s + "/")]
    pick = [(c, f) for (_, c, f) in dlh if rel(s, c, f) not in used]
    cand = [(c, f) for c, f in pub[s] if all(not (cc == c and abs(ff - f) < 15) for _, cc, ff in dlh)]
    for _ in range(2):  # up to 2 new frames per REPORT scene, >=15 apart from each other and from downloaded frames
        cand = [(c, f) for c, f in cand if all(not (cc == c and abs(ff - f) < 15) for cc, ff in pick)]
        if cand: pick.append(cand[rng.integers(len(cand))])
    sel += [dict(subset="reportscene", scene=s, cam=c, frame=f) for c, f in pick]
B = [x for x in sel if x["subset"] == "reportscene"]; A = [x for x in sel if x["subset"] == "newscene"]
keep = sorted(rng.choice(len(B), min(len(B), 600 - len(A)), replace=False)); sel = A + [B[j] for j in keep]  # total ~600
for x in sel: x["rel"] = rel(x["scene"], x["cam"], x["frame"])
assert not ({x["rel"] for x in sel} & used) and not any(split.get(x["scene"]) == "SELECT" for x in sel)
json.dump(sel, open(f"{F}/selection.json", "w"), indent=0)
print({k: (len(v), len({x['scene'] for x in v})) for k, v in collections.Counter().items()} or "",
      {sub: (sum(x["subset"] == sub for x in sel), len({x["scene"] for x in sel if x["subset"] == sub})) for sub in ("newscene", "reportscene")}, flush=True)
want = collections.defaultdict(set)
for x in sel:
    if not os.path.exists(f"{OUT}/{x['rel']}"): want[x["scene"]].add(x["rel"])
print("to download", sum(map(len, want.values())), "from", len(want), "scenes", flush=True)
def job(url):
    s = url.rsplit("/", 1)[1][:-4]
    if s not in want: return s, 0
    try:
        z = zipfile.ZipFile(dl.WebFile(url, requests.session())); names = set(z.namelist()); n = 0
        for w in want[s]:
            if w in names: z.extract(w, OUT); n += 1
        return s, n
    except Exception as e:
        return s, f"ERR {e}"
with ThreadPoolExecutor(12) as ex:
    for s, n in ex.map(job, dl.URLS):
        if n: print(s, n, flush=True)
miss = [x["rel"] for x in sel if not os.path.exists(f"{OUT}/{x['rel']}")]
print("MISSING", len(miss), miss[:10]); print("FRESHDL_DONE", flush=True)
