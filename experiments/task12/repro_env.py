"""Task 12: single-env check (nft-vp): vp_reward.final.perspective_reward on Task 11's 20 repro images (+1 NaN twin) vs Task 10 values.
Geometry workers use the forkserver start method (as flow_grpo.rewards.vp_perspective): a fork-started pool deadlocks after CUDA/OpenMP init."""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, pickle, multiprocessing, numpy as np
for p in ["DiffusionNFT", "vp_reward", "DeepLSD"]: sys.path.insert(0, code_dir(p))


def main():
    import torch
    from PIL import Image
    from vp_reward.final import perspective_reward
    S = pickle.load(open(DATA + "/outputs/task11/repro_sample.pkl", "rb"))
    ims = np.stack([np.array(Image.open(p).convert("RGB")) for p in S["path"]])
    fn = perspective_reward("cuda", lc_base=None, n_workers=8)
    md = [{"lc_base": float(x)} for x in S["Lc_base"]]
    R1, info = fn(ims, S["prompt"], md)                                                       # uint8 NHWC input
    R2, _ = fn(torch.tensor(ims).permute(0, 3, 1, 2).float().cuda() / 255, S["prompt"], md)   # DiffusionNFT float NCHW input
    ref = {k: np.array(S[k], float) for k in ("G_p2", "Lc", "Q_pick", "Q_aes", "R_t10")}
    out = {"G": np.nanmax(np.abs(info["G_raw"][:20] - ref["G_p2"][:20])), "Lc": np.max(np.abs(info["Lc"][:20] - ref["Lc"][:20])),
           "PickScore": np.max(np.abs(info["pickscore"][:20] - ref["Q_pick"][:20])), "aesthetic": np.max(np.abs(info["aesthetic"][:20] - ref["Q_aes"][:20])),
           "R": np.max(np.abs(R1[:20] - ref["R_t10"][:20])), "R float-tensor input vs uint8": np.max(np.abs(R2 - R1))}
    for k, v in out.items(): print(f"max|diff| {k:30s} {v:.3g}")
    print("NaN twin: G_nan", bool(info["G_nan"][20]), "G filled", info["G"][20], "R", R1[20], flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
