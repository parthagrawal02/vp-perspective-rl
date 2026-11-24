"""Task 12: floor reference Lc_base per prompt, from base SD3.5-M sampled exactly like DiffusionNFT training rollouts.
Code path = scripts/train_nft_sd3.py sampling: pipeline_with_logprob(num_steps=config.sample.num_steps, guidance_scale=
config.sample.guidance_scale, noise_level, deterministic, solver from config.nft.sd3_vp()), fp16 autocast, transformer weights fp32,
text encoders fp16, VAE fp32, text embeddings via encode_prompt(max_sequence_length=128), negative = "" embeds. No LoRA.
4 seeds per prompt (CPU generators, seeds 0-3). Lc via vp_reward.final (stretch -> DeepLSD -> V2 fit).
  python lcbase.py <shard> <nshard>   (nft-vp env, GPU)  -> outputs/task12/lcbase_<shard>.pkl, images in outputs/task12/lcbase_imgs/
  python lcbase.py merge              -> DiffusionNFT/dataset/vp_hypersim/lc_base.json + outputs/task12/lcbase_all.csv"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, os, json, pickle, time, numpy as np
for p in ["DiffusionNFT", "vp_reward", "DeepLSD"]: sys.path.insert(0, code_dir(p))
O = DATA + "/outputs/task12"; IMG = f"{O}/lcbase_imgs"; DSET = f"{code_dir('DiffusionNFT')}/dataset/vp_hypersim"
SEEDS = (0, 1, 2, 3)


def prompts():
    P = []
    for split in ("train", "test"):
        P += [(split, l.rstrip("\n")) for l in open(f"{DSET}/{split}.txt") if l.strip()]
    return P


def main(shard, nshard, bs=16):
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from diffusers import StableDiffusion3Pipeline
    from torch.cuda.amp import autocast as torch_autocast
    from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
    from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
    import imp
    cfgmod = imp.load_source("nftcfg", f"{code_dir('DiffusionNFT')}/config/nft.py")
    cwd = os.getcwd(); os.chdir(f"{code_dir('DiffusionNFT')}"); cfg = cfgmod.sd3_vp(); os.chdir(cwd)
    S = cfg.sample
    print("sampler:", dict(num_steps=S.num_steps, guidance_scale=S.guidance_scale, noise_level=S.noise_level, deterministic=S.deterministic,
                           solver=S.solver, resolution=cfg.resolution, mixed_precision=cfg.mixed_precision), flush=True)
    from vp_reward import final as F
    dev = "cuda"
    pipe = StableDiffusion3Pipeline.from_pretrained(cfg.pretrained.model)
    pipe.vae.to(dev, dtype=torch.float32)
    for te in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3): te.to(dev, dtype=torch.float16)
    pipe.transformer.to(dev); pipe.transformer.eval(); pipe.set_progress_bar_config(disable=True)
    tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]; toks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    with torch.no_grad():
        ne, npe = encode_prompt(tes, toks, [""], 128)
    net = F.load_deeplsd(dev); pool = ProcessPoolExecutor(16)
    jobs = [(sp, p, s) for sp, p in prompts() for s in SEEDS][shard::nshard]
    os.makedirs(IMG, exist_ok=True); rows = []; t0 = time.time()
    for b in range(0, len(jobs), bs):
        J = jobs[b:b + bs]; P = [p for _, p, _ in J]
        with torch.no_grad():
            pe, ppe = encode_prompt(tes, toks, P, 128); pe, ppe = pe.to(dev), ppe.to(dev)
        gens = [torch.Generator("cpu").manual_seed(s) for _, _, s in J]
        with torch_autocast(enabled=True, dtype=torch.float16), torch.no_grad():
            ims, _, _ = pipeline_with_logprob(pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe, negative_prompt_embeds=ne.to(dev).repeat(len(P), 1, 1),
                                              negative_pooled_prompt_embeds=npe.to(dev).repeat(len(P), 1), num_inference_steps=S.num_steps,
                                              guidance_scale=S.guidance_scale, output_type="pt", height=cfg.resolution, width=cfg.resolution,
                                              noise_level=S.noise_level, deterministic=S.deterministic, solver=S.solver, model_type="sd3", generator=gens)
        u8 = F.to_uint8_nhwc(ims)
        G, Lc = F.geometry_batch(net, u8, dev, pool)
        for (sp, p, s), x, g, l in zip(J, u8, G, Lc):
            k = len(rows) * nshard + shard
            from PIL import Image
            fn = f"{IMG}/{shard}_{len(rows):05d}.png"; Image.fromarray(x).save(fn)
            rows.append(dict(split=sp, prompt=p, seed=s, G=float(g), Lc=float(l), img=fn))
        if b % (bs * 10) == 0: print(shard, b, len(jobs), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(rows, open(f"{O}/lcbase_{shard}.pkl", "wb")); print("LCBASE_DONE", shard, len(rows), f"{time.time()-t0:.0f}s", flush=True)


def merge():
    import glob, pandas as pd
    D = pd.DataFrame([r for f in sorted(glob.glob(f"{O}/lcbase_[0-9]*.pkl")) for r in pickle.load(open(f, "rb"))])
    D.to_csv(f"{O}/lcbase_all.csv", index=False)
    n = D.groupby("prompt").seed.nunique(); assert (n == len(SEEDS)).all(), n.value_counts()
    m = D.groupby("prompt").Lc.mean()
    exp = {p for _, p in prompts()}; assert set(m.index) == exp, (len(m), len(exp))
    json.dump({p: float(v) for p, v in m.items()}, open(f"{DSET}/lc_base.json", "w"), indent=0)
    print("LC_BASE prompts", len(m), "samples", len(D))


if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method("forkserver", force=True)  # fork-started pool deadlocks after CUDA init
    if sys.argv[1] == "merge": merge()
    else: main(int(sys.argv[1]), int(sys.argv[2]))
