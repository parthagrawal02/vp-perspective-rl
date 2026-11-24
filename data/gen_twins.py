"""Caption-matched base twins: SD3.5-Medium, 512x512, 40 steps, CFG 4.5, seeds 0-3 (CPU torch.Generator per sample), fp16
transformer / text encoders, fp32 VAE, no LoRA, no prompt suffix. One process per GPU:
    CUDA_VISIBLE_DEVICES=g python data/gen_twins.py --ds <ds> --shard k --nshard N [--out DIR]   (resumable)
Default output $VP_DATA/outputs/twins/<ds>_<res>/base_cfg4.5/<id>_s<seed>.png. Where the project's twins actually live:
nyu -> outputs/task2/gen/base_cfg4.5, sun_kv2 / hypersim -> outputs/task3/gen_<ds>_512/base_cfg4.5,
hypersim_fresh -> outputs/task11/gen/base_cfg4.5 (+ the baselines of experiments/task11/gen11.py).

Origin: experiments/task03/gen.py (same logic; only paths / CLI arguments changed)."""
from vp_reward.paths import DATA
import argparse, json, os, torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
p = argparse.ArgumentParser(); p.add_argument("--shard", type=int); p.add_argument("--nshard", type=int); p.add_argument("--bs", type=int, default=8); p.add_argument("--ds"); p.add_argument("--res", type=int, default=512); p.add_argument("--seeds", default="0,1,2,3"); p.add_argument("--max_ids", type=int, default=0); p.add_argument("--out", default=None)
a = p.parse_args()
ROOT = a.out or DATA + f"/outputs/twins/{a.ds}_{a.res}"
caps = json.load(open(DATA + f"/datasets/{a.ds}/captions.json")); SUF = caps["suffix"]; C = caps["captions"]
meta = json.load(open(DATA + f"/datasets/{a.ds}/meta.json"))["items"]
ALL = [m["id"] for m in meta]; REP = [m["id"] for m in meta if m["split"] == "REPORT"]
SEEDS = [int(x) for x in a.seeds.split(",")]
IDS = ALL if not a.max_ids else [i for k, i in enumerate(sorted(ALL, key=lambda i: (i * 2654435761) % 2**32)) if k < a.max_ids]
VARIANTS = [("base_cfg4.5", None, 4.5, False, IDS, SEEDS)]
for name, lora, cfg, suf, ids, seeds in VARIANTS:
    os.makedirs(f"{ROOT}/{name}", exist_ok=True)
    jobs = [(i, s) for i in ids for s in seeds][a.shard::a.nshard]
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
            ims = pipe(prompts, num_inference_steps=40, guidance_scale=cfg, height=a.res, width=a.res, generator=gens).images
        for (i, s), im in zip(J, ims): im.save(f"{ROOT}/{name}/{i:04d}_s{s}.png")
        if b % (a.bs * 20) == 0: print(name, a.shard, b, "/", len(jobs), flush=True)
    del pipe; torch.cuda.empty_cache()
print("SHARD_DONE", a.shard, flush=True)
