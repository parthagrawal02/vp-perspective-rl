"""Prompt sets for DiffusionNFT from vp_pairs_v1, with a train / validation / final-test split by scene-level split:

    python rl/make_dataset_from_pairs.py /path/to/vp_pairs_v1 [--filter ref_better|none] [--train-sets hypersim_select,nyu,sun]

-> $VP_NFT_DIR/dataset/vp_hypersim/
   train.txt       SELECT-split captions of --train-sets             (RL training prompts)
   test.txt        hypersim_fresh captions                           (DiffusionNFT's periodic eval = VALIDATION: monitoring,
                                                                       checkpoint choice; DiffusionNFT reads this file name)
   final_test.txt  REPORT-split captions of the same sets as train   (held out from reward design and RL: evaluate ONCE, on
                                                                       the chosen checkpoint, for the reported numbers)
   prompt_map.json caption -> role, set, pair_id, scene, ref_G, twin_G
--filter ref_better (default) keeps a pair only if its reference image's stored G is higher than the mean G of its 4 base
twins: prompts where SD3.5-M draws worse perspective than the real scene, i.e. where there is room to learn. Applied to all three
roles so they share one distribution. Pairs with undefined G are dropped. Note the selection uses only 4 twins per caption, so
it partly selects prompts whose twins were unlucky (regression to the mean): expect the base model's average on these prompts to
be somewhat higher than the twins suggest.
Differences from rl/make_dataset.py (Task 12; record with any result): train prompts describe SELECT-split images (used to
design the reward; the policy never sees images, only captions) and include NYU / SUN RGB-D photos unless --train-sets excludes them.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np

from vp_reward.paths import NFT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--filter", default="ref_better", choices=["ref_better", "none"])
ap.add_argument("--train-sets", default="hypersim_select,nyu,sun")
a = ap.parse_args()
sets = a.train_sets.split(",")

pairs, twins = {}, collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(a.root, "metadata.csv"), newline="")):
    G = float(r["G"]) if r["G"] else np.nan
    if r["role"] == "reference":
        pairs[r["pair_id"]] = dict(set=r["set"], split=r["split"], scene=r["scene"], caption=" ".join(r["caption"].split()), ref_G=G)
    elif r["role"] == "twin":
        twins[r["pair_id"]].append(G)
for pid, p in pairs.items():
    p["twin_G"] = float(np.mean(twins[pid])) if twins[pid] else np.nan


def keep(p):
    if a.filter == "none":
        return True
    return np.isfinite(p["ref_G"]) and np.isfinite(p["twin_G"]) and p["ref_G"] > p["twin_G"]


roles = {
    "train": lambda p: p["split"] == "SELECT" and p["set"] in sets,
    "test": lambda p: p["set"] == "hypersim_fresh",
    "final_test": lambda p: p["split"] == "REPORT" and p["set"] in sets,
}
out = {role: [(p["caption"], pid, p) for pid, p in sorted(pairs.items()) if f(p) and keep(p)] for role, f in roles.items()}
# a caption may appear in only one role; duplicates across roles are dropped from train
cnt = collections.Counter(c for v in out.values() for c, _, _ in v)
dup = {c for c, n in cnt.items() if n > 1}
out["train"] = [x for x in out["train"] if x[0] not in dup]
for role in ("test", "final_test"):
    out[role] = list({x[0]: x for x in out[role]}.values())
assert not ({c for c, _, _ in out["test"]} & {c for c, _, _ in out["final_test"]}), "caption shared by validation and final test"

OUT = os.path.join(NFT_DIR, "dataset", "vp_hypersim"); os.makedirs(OUT, exist_ok=True)
pm = {}
for role, rows in out.items():
    assert all("\n" not in c and c for c, _, _ in rows)
    open(f"{OUT}/{role}.txt", "w").write("\n".join(c for c, _, _ in rows) + "\n")
    for c, pid, p in rows:
        pm[c] = dict(role=role, dataset=p["set"], pair_id=pid, scene=p["scene"], ref_G=p["ref_G"], twin_G=p["twin_G"])
json.dump(pm, open(f"{OUT}/prompt_map.json", "w"), indent=0)
print(f"filter={a.filter} train_sets={sets} duplicates dropped from train: {len(dup)}")
for role, rows in out.items():
    print(f"  {role:<11} {len(rows):>5}  {dict(collections.Counter(p['set'] for _, _, p in rows))}")
print("->", OUT)
