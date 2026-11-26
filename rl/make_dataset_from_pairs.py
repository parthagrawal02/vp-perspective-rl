"""FALLBACK prompt set for DiffusionNFT when the Task 12 training captions ($VP_DATA/datasets/hypersim_train) are not available
(e.g. on a fresh cluster account). Builds $VP_NFT_DIR/dataset/vp_hypersim/{train,test}.txt + prompt_map.json from vp_pairs_v1:

    python rl/make_dataset_from_pairs.py /path/to/vp_pairs_v1 [--train-sets hypersim_select,nyu,sun]

train = captions of the SELECT-split pairs of the chosen sets (default all three: 497 + 705 + 498 = 1,700 prompts)
test  = captions of hypersim_fresh (596), the same test prompts as rl/make_dataset.py.
Exact duplicates across train/test are dropped from train. Differences from rl/make_dataset.py (record them with any result):
the train prompts describe SELECT-split images, i.e. the images used to design and tune the reward (the RL policy never sees
the images, only the captions), and they include NYU / SUN RGB-D photos, not only Hypersim renders.
"""
import argparse
import collections
import csv
import json
import os

from vp_reward.paths import NFT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--train-sets", default="hypersim_select,nyu,sun")
a = ap.parse_args()
sets = a.train_sets.split(",")
pairs = {}
for r in csv.DictReader(open(os.path.join(a.root, "metadata.csv"), newline="")):
    if r["role"] == "reference":
        pairs[r["pair_id"]] = dict(set=r["set"], split=r["split"], scene=r["scene"], caption=" ".join(r["caption"].split()))
tr = [(p["caption"], pid, p) for pid, p in sorted(pairs.items()) if p["split"] == "SELECT" and p["set"] in sets]
te = [(p["caption"], pid, p) for pid, p in sorted(pairs.items()) if p["set"] == "hypersim_fresh"]
cnt = collections.Counter(c for c, _, _ in tr + te); dup = {c for c, n in cnt.items() if n > 1}
tr = [x for x in tr if x[0] not in dup]
te = list({x[0]: x for x in te}.values())  # one line per distinct test caption
assert all("\n" not in c and c for c, _, _ in tr + te)
OUT = os.path.join(NFT_DIR, "dataset", "vp_hypersim"); os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/train.txt", "w").write("\n".join(c for c, _, _ in tr) + "\n")
open(f"{OUT}/test.txt", "w").write("\n".join(c for c, _, _ in te) + "\n")
pm = {c: dict(split="train", dataset=p["set"], pair_id=pid, scene=p["scene"]) for c, pid, p in tr}
pm.update({c: dict(split="test", dataset=p["set"], pair_id=pid, scene=p["scene"]) for c, pid, p in te})
json.dump(pm, open(f"{OUT}/prompt_map.json", "w"), indent=0)
print("duplicates dropped:", len(dup), "| train", len(tr), collections.Counter(p["set"] for _, _, p in tr), "| test", len(te), "->", OUT)
