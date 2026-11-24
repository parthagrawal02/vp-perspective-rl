# Selective Hypersim download: up to 3 public tonemapped frames per scene (spread across cameras/frames), via HTTP range requests.
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, zipfile, csv, collections, requests
from concurrent.futures import ThreadPoolExecutor
H = DATA + "/datasets/hypersim"
sys.path.insert(0, f"{H}/ml-hypersim/contrib/99991")
import download as dl
OUT = f"{H}/raw"; os.makedirs(OUT, exist_ok=True)
pub = collections.defaultdict(list)
for r in csv.DictReader(open(f"{H}/ml-hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv")):
    if r["included_in_public_release"] == "True": pub[r["scene_name"]].append((r["camera_name"], int(r["frame_id"])))
def pick(lst):
    cams = sorted(set(c for c, _ in lst)); out = []
    for k in range(3):
        c = cams[k % len(cams)]; fr = sorted(f for cc, f in lst if cc == c)
        f = fr[[0, len(fr) // 2, -1][k // len(cams) if len(cams) < 3 else 1] if len(fr) else 0]
        if (c, f) not in out: out.append((c, f))
    return out
def job(url):
    scene = url.rsplit("/", 1)[1][:-4]
    if scene not in pub: return scene, 0
    want = {f"{scene}/images/scene_{c}_final_preview/frame.{f:04d}.tonemap.jpg" for c, f in pick(pub[scene])}
    try:
        z = zipfile.ZipFile(dl.WebFile(url, requests.session())); names = set(z.namelist()); n = 0
        for w in want:
            if w in names and not os.path.exists(f"{OUT}/{w}"): z.extract(w, OUT); n += 1
        return scene, n
    except Exception as e:
        return scene, f"ERR {e}"
with ThreadPoolExecutor(12) as ex:
    for s, n in ex.map(job, dl.URLS): print(s, n, flush=True)
print("HSDONE")
