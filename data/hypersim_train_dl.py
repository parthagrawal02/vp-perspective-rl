"""RL training prompts (Task 12): ~2500 never-used Hypersim frames from SELECT scenes only, >= 15 frame ids apart.
Frozen output: datasets/hypersim_train/selection.json (depends on download state; do not re-run the selection).

Origin: experiments/task12/train_dl.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA
import sys, os, csv, json, glob, zipfile, zlib, collections, requests, numpy as np
from concurrent.futures import ThreadPoolExecutor
H = DATA + "/datasets/hypersim"; OUT = f"{H}/raw"; F = DATA + "/datasets/hypersim_train"
os.makedirs(F, exist_ok=True)
sys.path.insert(0, f"{H}/ml-hypersim/contrib/99991"); import download as dl
if os.path.exists(f"{F}/selection.json"):
    sel = json.load(open(f"{F}/selection.json")); print("using frozen selection", len(sel))
else:
    pub = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{H}/ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv")):
        if r["included_in_public_release"] == "True": pub[r["scene_name"]].append((r["camera_name"], int(r["frame_id"])))
    M = json.load(open(f"{H}/meta.json"))["items"]; split = {m["scene"]: m["split"] for m in M}
    fresh = {x["rel"] for x in json.load(open(DATA + "/datasets/hypersim_fresh/selection.json"))}
    have = json.load(open(f"{F}/pre_task12_ondisk.json"))  # frozen list of frames on disk before Task 12 (used or downloaded earlier)
    parse = lambda p: (p.split("/")[0], p.split("/")[2][len("scene_"):-len("_final_preview")], int(p.split("frame.")[1][:4]))
    rel = lambda s, c, f: f"{s}/images/scene_{c}_final_preview/frame.{f:04d}.tonemap.jpg"
    per_scene = {}
    for s in sorted(k for k, v in split.items() if v == "SELECT"):
        block = [(c, f) for (ss, c, f) in map(parse, have) if ss == s]  # everything already on disk counts as used/blocked
        cams = sorted({c for c, _ in pub[s]}); fr = {c: sorted(f for cc, f in pub[s] if cc == c) for c in cams}
        # per camera: greedy over all public frames (in order), keep frames >= 15 ids from blocked and already kept frames
        percam = {}
        for c in cams:
            kept = []
            for f in fr[c]:
                if all(not (cc == c and abs(ff - f) < 15) for cc, ff in block) and all(abs(ff - f) >= 15 for ff in kept): kept.append(f)
            rng0 = np.random.default_rng(zlib.crc32(f"{s}|{c}".encode())); percam[c] = list(rng0.permutation(kept))
        order, k = [], 0  # round-robin over cameras -> spread across cameras when truncated
        while any(percam.values()):
            c = cams[k % len(cams)]; k += 1
            if percam[c]: order.append((c, int(percam[c].pop(0))))
        per_scene[s] = order
    print("capacity", sum(map(len, per_scene.values())), "scenes", len(per_scene))
    cap = 0  # water-filling: equal per-scene quota, raised until 2500 frames
    while sum(min(cap + 1, len(v)) for v in per_scene.values()) <= 2500 and cap < 1000: cap += 1
    sel = []
    for s, v in per_scene.items(): sel += [dict(scene=s, cam=c, frame=f, rel=rel(s, c, f)) for c, f in v[:cap]]
    rest = [(s, v[cap]) for s, v in per_scene.items() if len(v) > cap]
    rng = np.random.default_rng(12)
    for j in rng.permutation(len(rest))[: 2500 - len(sel)]:
        s, (c, f) = rest[j]; sel.append(dict(scene=s, cam=c, frame=f, rel=rel(s, c, f)))
    print("per-scene quota", cap)
    used = {m["src"].split("/raw/")[1] for m in M}
    assert not ({x["rel"] for x in sel} & (used | fresh)) and all(split[x["scene"]] == "SELECT" for x in sel)
    json.dump(sel, open(f"{F}/selection.json", "w"), indent=0)
print("selected", len(sel), "scenes", len({x["scene"] for x in sel}), "cams", len({(x["scene"], x["cam"]) for x in sel}), flush=True)
want = collections.defaultdict(set)
for x in sel:
    p = f"{OUT}/{x['rel']}"
    if not os.path.exists(p) or os.path.getsize(p) == 0: want[x["scene"]].add(x["rel"])
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
miss = [x["rel"] for x in sel if not os.path.exists(f"{OUT}/{x['rel']}") or os.path.getsize(f"{OUT}/{x['rel']}") == 0]
print("MISSING", len(miss), miss[:10]); print("TRAINDL_DONE", flush=True)
