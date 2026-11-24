from vp_reward.paths import DATA, src as code_dir  # [repo] paths come from config/env.sh (were hardcoded)
import sys, time, traceback, torch
sys.path.insert(0, code_dir("DiffusionNFT"))
from flow_grpo.rewards import multi_score
dev = "cuda"
torch.manual_seed(0)
imgs = torch.rand(2, 3, 512, 512)  # random-noise images in [0,1], as pipeline output_type="pt"
grad = torch.linspace(0, 1, 512).view(1, 1, 512).expand(3, 512, 512)
imgs[1] = grad  # second image: smooth gradient
prompts = ["a photo of a cat", "a city street with tall buildings"]
for name in ["pickscore", "clipscore", "hpsv2", "aesthetic", "imagereward"]:
    t = time.time()
    try:
        fn = multi_score(dev, {name: 1.0})
        sc, _ = fn(imgs, prompts, [{}, {}], only_strict=False)
        v = sc[name]
        v = v.tolist() if torch.is_tensor(v) else list(v)
        print(f"PASS {name:12s} scores={[round(float(x),4) for x in v]} ({time.time()-t:.1f}s)", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    torch.cuda.empty_cache()
print("peak mem GB", torch.cuda.max_memory_allocated()/1e9)
