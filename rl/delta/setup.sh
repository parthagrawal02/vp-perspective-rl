#!/bin/bash
# One-time setup on an NCSA Delta LOGIN node (needs internet; light work + one small C++ build):
#     cd ~/vp-perspective-rl && bash rl/delta/setup.sh
# SD3.5-Medium is gated: accept its license on huggingface.co. The first run stops at the prefetch and tells you to run
# `$VP_ENV/bin/huggingface-cli login`; then re-run this script (every step is idempotent).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
module load pytorch-conda                                   # provides conda (NCSA guide)
[ -f "$REPO/config/env.sh" ] || cp "$REPO/rl/delta/env.delta.sh" "$REPO/config/env.sh"
source "$REPO/config/env.sh"
echo "VP_DATA=$VP_DATA  VP_ENV=$VP_ENV"
mkdir -p "$VP_DATA" "$(dirname "$VP_ENV")"

# Env + DeepLSD + patched DiffusionNFT + vp_reward + DeepLSD / aesthetic weights (A40 = sm_86, covered by the cu128 wheels)
MAMBA=conda bash "$REPO/envs/setup_nft_vp.sh"

# Pre-fetch the Hugging Face models into $HF_HOME on /work, so compute jobs never wait on downloads
"$VP_ENV/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
# SD3.5-M: only what diffusers' StableDiffusion3Pipeline.from_pretrained loads (~16 GB). The full repo is ~49 GB (fp16
# duplicates, single-file sd3.5_medium.safetensors and text_encoders/ for ComfyUI), none of it used here.
SD35 = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*", "vae/*", "transformer/*",
        "text_encoder/config.json", "text_encoder/model.safetensors", "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
        "text_encoder_3/config.json", "text_encoder_3/model.safetensors.index.json", "text_encoder_3/model-0000*"]
for repo, pat in [("stabilityai/stable-diffusion-3.5-medium", SD35),                 # gated: needs `huggingface-cli login`
                  ("yuvalkirstain/PickScore_v1", ["*.json", "*.safetensors", "*.txt"]),
                  ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ["*.json", "*.txt"]),   # processor / tokenizer only
                  ("openai/clip-vit-large-patch14", ["*.json", "*.safetensors", "*.txt"])]:
    print("fetch", repo, flush=True)
    try:
        snapshot_download(repo, allow_patterns=pat)
    except Exception as e:  # gated SD3.5 without login
        raise SystemExit(f"\n{repo}: {type(e).__name__}. Accept its license on huggingface.co, run "
                         f"`$VP_ENV/bin/huggingface-cli login`, then re-run `bash rl/delta/setup.sh` (safe to re-run).")
print("HF_PREFETCH_OK")
EOF
echo "Setup done. Next: sbatch rl/delta/check.slurm"
