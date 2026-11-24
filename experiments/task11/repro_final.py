"""Reproduction check of vp_reward/final.py against Task 10 values on 20 saved SELECT base twins (+1 NaN-G twin for the fill rule).
  python repro_final.py geo   (deeplsd env, GPU)   python repro_final.py q (nft env, GPU)   python repro_final.py check"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, json, pickle, numpy as np
sys.path.insert(0, code_dir("DiffusionNFT"))
from PIL import Image
O = DATA + "/outputs/task11"; import os; os.makedirs(O, exist_ok=True)
GEN = {"nyu": DATA + "/outputs/task2/gen/base_cfg4.5", "sun_kv2": DATA + "/outputs/task3/gen_sun_kv2_512/base_cfg4.5",
       "hypersim": DATA + "/outputs/task3/gen_hypersim_512/base_cfg4.5"}


def sample():
    import pandas as pd
    T = pd.read_pickle(DATA + "/outputs/task10/table10_data.pkl")
    B = T[(T.kind == "m") & (T.set == "base") & (T.rseed == 0)]
    S = pd.concat([B[B.G_p2.notna()].sample(20, random_state=11), B[B.G_p2.isna()].head(1)])
    caps = {ds: json.load(open(DATA + f"/datasets/{ds}/captions.json"))["captions"] for ds in GEN}
    S = S.assign(prompt=[caps[d][str(i)] for d, i in zip(S.ds, S.id)], path=[f"{GEN[d]}/{i:04d}_s{s}.png" for d, i, s in zip(S.ds, S.id, S.seed)])
    S = S.assign(R_t10=S.zG_p2 - 4 * np.maximum(0, 1 - S.Lc / (0.8 * S.Lc_base)) + 0.25 * S.zQ_sum)
    return S[["ds", "id", "seed", "path", "prompt", "G_p2", "Lc", "Lc_base", "Q_pick", "Q_aes", "R_t10"]].reset_index(drop=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "sample": pickle.dump(sample().to_dict("list"), open(f"{O}/repro_sample.pkl", "wb")); print("ok")
    else:
        S = pickle.load(open(f"{O}/repro_sample.pkl", "rb")); ims = np.stack([np.array(Image.open(p).convert("RGB")) for p in S["path"]])
        if st == "geo":
            from vp_reward import final as F
            G, Lc = F.geometry_batch(F.load_deeplsd("cuda"), ims); pickle.dump((G, Lc), open(f"{O}/repro_geo.pkl", "wb")); print("geo ok")
        elif st == "q":
            import torch
            from vp_reward import final as F
            from flow_grpo.pickscore_scorer import PickScoreScorer
            from flow_grpo.aesthetic_scorer import AestheticScorer
            p, a = F.quality_batch(PickScoreScorer(device="cuda", dtype=torch.float32), AestheticScorer(dtype=torch.float32, device="cuda"), ims, list(S["prompt"]))
            pickle.dump((p, a), open(f"{O}/repro_q.pkl", "wb")); print("q ok")
        elif st == "check":
            import pandas as pd
            from vp_reward import final as F
            S = pd.DataFrame(S)
            G, Lc = pickle.load(open(f"{O}/repro_geo.pkl", "rb")); p, a = pickle.load(open(f"{O}/repro_q.pkl", "rb"))
            R, info = F.combine(G, Lc, S.Lc_base.values, p, a)
            D = S.assign(G_new=G, Lc_new=Lc, pick_new=p, aes_new=a, R_new=R, nan_flag=info["G_nan"])
            for c, n in (("G_p2", "G_new"), ("Lc", "Lc_new"), ("Q_pick", "pick_new"), ("Q_aes", "aes_new"), ("R_t10", "R_new")):
                d = np.abs(D[c] - D[n])[:20]; print(f"{c:7s} max|diff| over 20 = {np.nanmax(d):.3g}")
            last = D.iloc[20]; print("NaN-G twin:", last.ds, last.id, last.seed, "T10 G", last.G_p2, "-> new G_raw", G[20], "flag", bool(info["G_nan"][20]), "filled G", info["G"][20], "R", R[20])
            D.drop(columns=["path"]).to_csv(f"{O}/repro_check.csv", index=False)
