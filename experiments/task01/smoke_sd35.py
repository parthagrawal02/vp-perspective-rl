from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import time, torch
from diffusers import StableDiffusion3Pipeline
p = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16).to("cuda")
out = DATA + "/outputs/smoke"; import os; os.makedirs(out, exist_ok=True)
prompt = "A photo of a long hallway with tall windows, one-point perspective"
g = lambda: torch.Generator("cuda").manual_seed(0)
p(prompt, num_inference_steps=40, guidance_scale=4.5, height=512, width=512, generator=g())  # warmup
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
ts = []
for i in range(3):
    t = time.time(); im = p(prompt, num_inference_steps=40, guidance_scale=4.5, height=512, width=512, generator=g()).images[0]
    torch.cuda.synchronize(); ts.append(time.time() - t)
im.save(f"{out}/sd35m_512_40step_cfg4.5.png")
print("sec/img (batch 1, fp16):", [round(x, 2) for x in ts], "peak alloc GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "peak reserved GB:", round(torch.cuda.max_memory_reserved() / 1e9, 2))
