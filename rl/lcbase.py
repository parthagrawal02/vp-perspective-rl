"""PROVISIONAL (floor reference being fixed). Floor reference Lc_base per prompt: mean Lc (vp_reward.final) of 4 base SD3.5-M
samples per prompt. Two samplers ($VP_LCBASE_SAMPLER or --sampler):
    train      (default, Task 12) exactly like DiffusionNFT training rollouts (config sd3_vp: 10 dpm2 steps, CFG 1.0)
               -> dataset/vp_hypersim/lc_base.json
    validated  exactly like the validation twins the floor was calibrated on (data/gen_twins.py: 40 steps, CFG 4.5,
               fp16 transformer/text encoders, fp32 VAE, CPU generator per seed)    -> dataset/vp_hypersim/lc_base_validated.json
    python rl/lcbase.py <shard> <nshard> [--sampler S]   (one GPU per shard)    python rl/lcbase.py merge [--sampler S]

Origin: experiments/task12/lcbase.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA, NFT_DIR
import sys, os, json, pickle, time, numpy as np
sys.path.insert(0, NFT_DIR)
SAMPLER = os.environ.get("VP_LCBASE_SAMPLER", "train")
if "--sampler" in sys.argv:
    SAMPLER = sys.argv[sys.argv.index("--sampler") + 1]; del sys.argv[sys.argv.index("--sampler"):sys.argv.index("--sampler") + 2]
assert SAMPLER in ("train", "validated"), SAMPLER
O = DATA + ("/outputs/task12" if SAMPLER == "train" else "/outputs/lcbase_validated"); IMG = f"{O}/lcbase_imgs"
DSET = f"{NFT_DIR}/dataset/vp_hypersim"; LC_JSON = f"{DSET}/lc_base.json" if SAMPLER == "train" else f"{DSET}/lc_base_validated.json"
SEEDS = (0, 1, 2, 3)


def prompts():
    P = []
    for split in ("train", "test"):
        P += [(split, l.rstrip("\n")) for l in open(f"{DSET}/{split}.txt") if l.strip()]
    return P


def main(shard, nshard, bs=16):
    if SAMPLER == "validated":
        return main_validated(shard, nshard)
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from diffusers import StableDiffusion3Pipeline
    from torch.cuda.amp import autocast as torch_autocast
    from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
    from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
    import imp
    cfgmod = imp.load_source("nftcfg", f"{NFT_DIR}/config/nft.py")
    cwd = os.getcwd(); os.chdir(NFT_DIR); cfg = cfgmod.sd3_vp(); os.chdir(cwd)
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


def main_validated(shard, nshard, bs=8):
    """Same sampler as data/gen_twins.py (the validation twins)."""
    import torch
    from concurrent.futures import ProcessPoolExecutor
    from diffusers import StableDiffusion3Pipeline
    from PIL import Image
    from vp_reward import final as F
    dev = "cuda"
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    for m in (pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3): m.to(dev, torch.float16)
    pipe.vae.to(dev, torch.float32); pipe.set_progress_bar_config(disable=True)
    net = F.load_deeplsd(dev); pool = ProcessPoolExecutor(12)
    jobs = [(sp, p, s) for sp, p in prompts() for s in SEEDS][shard::nshard]
    os.makedirs(IMG, exist_ok=True)
    ck = f"{O}/lcbase_{shard}.pkl"
    rows = pickle.load(open(ck, "rb")) if os.path.exists(ck) else []  # resumable
    done = {(r["prompt"], r["seed"]) for r in rows}; jobs = [j for j in jobs if (j[1], j[2]) not in done]
    print("sampler: validated (40 steps, CFG 4.5)", len(jobs), "to do", flush=True); t0 = time.time()
    for b in range(0, len(jobs), bs):
        J = jobs[b:b + bs]
        gens = [torch.Generator("cpu").manual_seed(s) for _, _, s in J]
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            ims = pipe([p for _, p, _ in J], num_inference_steps=40, guidance_scale=4.5, height=512, width=512, generator=gens).images
        u8 = np.stack([np.asarray(im.convert("RGB")) for im in ims])
        G, Lc = F.geometry_batch(net, u8, dev, pool)
        for (sp, p, s), x, g, l in zip(J, u8, G, Lc):
            fn = f"{IMG}/{shard}_{len(rows):05d}.png"; Image.fromarray(x).save(fn)
            rows.append(dict(split=sp, prompt=p, seed=s, G=float(g), Lc=float(l), img=fn))
        if (b // bs) % 25 == 0:
            pickle.dump(rows, open(ck, "wb")); print(shard, b, len(jobs), f"{time.time()-t0:.0f}s", flush=True)
    pickle.dump(rows, open(ck, "wb")); print("LCBASE_DONE", shard, len(rows), f"{time.time()-t0:.0f}s", flush=True)


def merge():
    import glob, pandas as pd
    D = pd.DataFrame([r for f in sorted(glob.glob(f"{O}/lcbase_[0-9]*.pkl")) for r in pickle.load(open(f, "rb"))])
    D.to_csv(f"{O}/lcbase_all.csv", index=False)
    n = D.groupby("prompt").seed.nunique(); assert (n == len(SEEDS)).all(), n.value_counts()
    m = D.groupby("prompt").Lc.mean()
    exp = {p for _, p in prompts()}; assert set(m.index) == exp, (len(m), len(exp))
    json.dump({p: float(v) for p, v in m.items()}, open(LC_JSON, "w"), indent=0)
    print("LC_BASE", SAMPLER, "prompts", len(m), "samples", len(D), "->", LC_JSON)


if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method("forkserver", force=True)  # fork-started pool deadlocks after CUDA init
    if sys.argv[1] == "merge": merge()
    else: main(int(sys.argv[1]), int(sys.argv[2]))
