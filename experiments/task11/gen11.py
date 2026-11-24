"""Task 11 generation on the fresh Hypersim split (512, 40 steps; same pipeline settings as task2/task3 gen.py).
python gen11.py --shard i --nshard N   (resumable)"""
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import argparse, json, os, torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
p = argparse.ArgumentParser(); p.add_argument("--shard", type=int); p.add_argument("--nshard", type=int); p.add_argument("--bs", type=int, default=8)
a = p.parse_args()
ROOT = DATA + "/outputs/task11/gen"
caps = json.load(open(DATA + "/datasets/hypersim_fresh/captions.json")); SUF = caps["suffix"]; C = caps["captions"]
IDS = sorted(int(k) for k in C)
VARIANTS = [  # name, lora, cfg, suffix, seeds
    ("base_cfg4.5", None, 4.5, False, [0, 1, 2, 3]),
    ("base_cfg1.0", None, 1.0, False, [0, 1]),
    ("base_suffix_cfg4.5", None, 4.5, True, [0, 1]),
    ("flowgrpo_cfg4.5", "jieliu/SD3.5M-FlowGRPO-PickScore", 4.5, False, [0, 1]),
    ("nft_cfg1.0", "worstcoder/SD3.5M-DiffusionNFT-MultiReward", 1.0, False, [0, 1]),
]
for name, lora, cfg, suf, seeds in VARIANTS:
    os.makedirs(f"{ROOT}/{name}", exist_ok=True)
    jobs = [(i, s) for i in IDS for s in seeds][a.shard::a.nshard]
    jobs = [(i, s) for i, s in jobs if not os.path.exists(f"{ROOT}/{name}/{i:04d}_s{s}.png")]
    if not jobs: continue
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    if lora: pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora).merge_and_unload()
    for m in (pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3): m.to("cuda", torch.float16)
    pipe.vae.to("cuda", torch.float32); pipe.set_progress_bar_config(disable=True)
    for b in range(0, len(jobs), a.bs):
        J = jobs[b:b + a.bs]
        prompts = [C[str(i)] + (SUF if suf else "") for i, _ in J]
        gens = [torch.Generator("cpu").manual_seed(s) for _, s in J]
        with torch.autocast("cuda", torch.float16), torch.no_grad():
            ims = pipe(prompts, num_inference_steps=40, guidance_scale=cfg, height=512, width=512, generator=gens).images
        for (i, s), im in zip(J, ims): im.save(f"{ROOT}/{name}/{i:04d}_s{s}.png")
        if b % (a.bs * 20) == 0: print(name, a.shard, b, "/", len(jobs), flush=True)
    del pipe; torch.cuda.empty_cache()
print("SHARD_DONE", a.shard, flush=True)
