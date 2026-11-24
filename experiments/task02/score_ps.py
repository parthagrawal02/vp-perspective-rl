# PickScore (x26 = paper units) and Aesthetic for real + generated sets, using DiffusionNFT's scorers. Prompt = plain caption.
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import json, sys, torch, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, code_dir("DiffusionNFT"))
from flow_grpo.pickscore_scorer import PickScoreScorer
from flow_grpo.aesthetic_scorer import AestheticScorer
OUT = DATA + "/outputs/task2"
C = json.load(open(DATA + "/datasets/nyu/captions.json"))["captions"]
meta = json.load(open(DATA + "/datasets/nyu/meta.json"))["items"]
ALL = [m["id"] for m in meta]; REP = [m["id"] for m in meta if m["split"] == "REPORT"]
items = [("real", i, -1, DATA + f"/datasets/nyu/rgb512/{i:04d}.png") for i in ALL]
items += [("base", i, s, f"{OUT}/gen/base_cfg4.5/{i:04d}_s{s}.png") for i in ALL for s in range(4)]
for v, d in [("base_suffix", "base_suffix_cfg4.5"), ("flowgrpo", "flowgrpo_cfg4.5"), ("nft", "nft_cfg1.0")]:
    items += [(v, i, s, f"{OUT}/gen/{d}/{i:04d}_s{s}.png") for i in REP for s in range(2)]
ps = PickScoreScorer(device="cuda"); ae = AestheticScorer(dtype=torch.float32, device="cuda")
rows = []
for b in range(0, len(items), 64):
    B = items[b:b + 64]
    ims = [Image.open(p).convert("RGB") for *_, p in B]
    prompts = [C[str(i)] for _, i, _, _ in B]
    p = ps(prompts, ims).cpu().numpy() * 26
    arr = torch.tensor(np.stack([np.array(im) for im in ims])).permute(0, 3, 1, 2)
    a = ae(arr).cpu().numpy()
    rows += [dict(set=s, id=i, seed=sd, pickscore=float(x), aesthetic=float(y)) for (s, i, sd, _), x, y in zip(B, p, a)]
pd.DataFrame(rows).to_pickle(f"{OUT}/pick_aes.pkl"); print("PSDONE", len(rows))
