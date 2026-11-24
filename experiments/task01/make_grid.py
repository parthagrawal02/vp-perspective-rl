# 4 prompts x 3 models grid, identical prompts + seeds per column; mirrors evaluation.py loading (fp16, LoRA merged).
from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import torch, os
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from PIL import Image, ImageDraw
OUT = DATA + "/outputs/grid"; os.makedirs(OUT, exist_ok=True)
prompts = ["A red colored car.",
           "A pyramid made of falafel with a partial solar eclipse in the background.",
           "New York Skyline with 'Google Research Pizza Cafe' written with fireworks on the sky.",
           "Two cats and two dogs sitting on the grass."]
models = [("SD3.5-M (CFG 4.5)", None, 4.5),
          ("Flow-GRPO PickScore (CFG 4.5)", "jieliu/SD3.5M-FlowGRPO-PickScore", 4.5),
          ("DiffusionNFT Multi (CFG 1.0)", "worstcoder/SD3.5M-DiffusionNFT-MultiReward", 1.0)]
cols = []
for name, lora, cfg in models:
    p = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    if lora:
        p.transformer = PeftModel.from_pretrained(p.transformer, lora).merge_and_unload()
    p.to("cuda", torch.float16); p.vae.to(torch.float32)
    col = []
    for i, pr in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(1000 + i)
        with torch.autocast("cuda", torch.float16):
            im = p(pr, num_inference_steps=40, guidance_scale=cfg, height=512, width=512, generator=g).images[0]
        im.save(f"{OUT}/{name.split()[0]}_{i}.png"); col.append(im)
    cols.append(col); del p; torch.cuda.empty_cache()
S, H = 512, 40
grid = Image.new("RGB", (3 * S, H + 4 * S), "white"); d = ImageDraw.Draw(grid)
for c, (name, _, _) in enumerate(models):
    d.text((c * S + 10, 12), name, fill="black")
    for r in range(4): grid.paste(cols[c][r], (c * S, H + r * S))
grid.save(f"{OUT}/grid_4prompts_x_3models.jpg", quality=92)
open(f"{OUT}/prompts.txt", "w").write("\n".join(f"row{i} seed={1000+i}: {p}" for i, p in enumerate(prompts)))
print("saved", f"{OUT}/grid_4prompts_x_3models.jpg")
