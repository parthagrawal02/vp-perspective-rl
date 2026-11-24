"""Build the DiffusionNFT prompt set dataset/vp_hypersim/{train,test}.txt + prompt_map.json inside $VP_NFT_DIR.
train = hypersim_train captions (SELECT scenes), test = hypersim_fresh captions; exact duplicates across the two are dropped from train.

Origin: experiments/task12/make_dataset.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA, NFT_DIR
import json, collections
D = DATA + "/datasets"; OUT = NFT_DIR + "/dataset/vp_hypersim"
import os; os.makedirs(OUT, exist_ok=True)
def load(ds):
    C = json.load(open(f"{D}/{ds}/captions.json"))["captions"]; M = {m["id"]: m for m in json.load(open(f"{D}/{ds}/meta.json"))["items"]}
    return [(" ".join(c.split()), int(i), M[int(i)]["scene"]) for i, c in sorted(C.items(), key=lambda kv: int(kv[0]))]
tr, te = load("hypersim_train"), load("hypersim_fresh")
cnt = collections.Counter(p for p, _, _ in tr + te); dup = {p for p, n in cnt.items() if n > 1}
print("duplicates (train+test):", len(dup))
tr = [x for x in tr if x[0] not in dup]; te_set = {p for p, _, _ in te}; assert not dup & te_set, "duplicate involves a test prompt"
assert all("\n" not in p and p == p.strip() for p, _, _ in tr + te)
open(f"{OUT}/train.txt", "w").write("\n".join(p for p, _, _ in tr) + "\n"); open(f"{OUT}/test.txt", "w").write("\n".join(p for p, _, _ in te) + "\n")
pm = {p: dict(split="train", dataset="hypersim_train", id=i, scene=s) for p, i, s in tr}
pm.update({p: dict(split="test", dataset="hypersim_fresh", id=i, scene=s) for p, i, s in te})
json.dump(pm, open(f"{OUT}/prompt_map.json", "w"), indent=0)
print("train", len(tr), "scenes", len({s for _, _, s in tr}), "| test", len(te), "scenes", len({s for _, _, s in te}), "| unique", len(pm) == len(tr) + len(te))
