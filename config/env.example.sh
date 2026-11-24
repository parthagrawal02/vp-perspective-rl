# Project configuration. Copy to config/env.sh (git-ignored), edit VP_DATA, then `source config/env.sh` in every shell.
# Every path the code uses is derived from these variables (see vp_reward/vp_reward/paths.py); nothing is hardcoded.
export VP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Root for everything large: datasets/, outputs/, models/, checkpoints/, logs/, cache/, envs/. Never inside the repo.
export VP_DATA="${VP_DATA:-$HOME/vp-data}"
export VP_ENV="${VP_ENV:-$VP_DATA/envs/nft-vp}"                       # training/reward env (envs/setup_nft_vp.sh)
export VP_VLM_ENV="${VP_VLM_ENV:-$VP_DATA/envs/vlm}"                  # captioning env (envs/setup_vlm.sh)
export VP_THIRD_PARTY="${VP_THIRD_PARTY:-$VP_REPO/third_party}"       # upstream clones (git-ignored)
export VP_NFT_DIR="${VP_NFT_DIR:-$VP_THIRD_PARTY/DiffusionNFT}"       # NVlabs/DiffusionNFT @ cbb14f8 + rl/diffusionnft_vp.patch
export VP_DEEPLSD_DIR="${VP_DEEPLSD_DIR:-$VP_THIRD_PARTY/DeepLSD}"    # cvg/DeepLSD @ f7d9d62
export VP_DEEPLSD_CKPT="${VP_DEEPLSD_CKPT:-$VP_DATA/models/deeplsd/deeplsd_wireframe.tar}"

# Caches on the data disk (HF models: SD3.5-M, PickScore, CLIP, Qwen captioner).
export HF_HOME="$VP_DATA/cache/huggingface" HF_HUB_CACHE="$VP_DATA/cache/huggingface/hub" HF_DATASETS_CACHE="$VP_DATA/cache/huggingface/datasets"
# If you ran `huggingface-cli login` before setting HF_HOME, point at that token file instead of logging in again:
# export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export TORCH_HOME="$VP_DATA/cache/torch" PIP_CACHE_DIR="$VP_DATA/cache/pip" UV_CACHE_DIR="$VP_DATA/cache/uv"
export TRITON_CACHE_DIR="$VP_DATA/cache/triton" XDG_CACHE_HOME="$VP_DATA/cache/xdg" WANDB_DIR="$VP_DATA/wandb"
export CONDA_PKGS_DIRS="$VP_DATA/cache/conda_pkgs" TMPDIR="$VP_DATA/tmp"
export FLASHINFER_WORKSPACE_BASE="$VP_DATA/cache/flashinfer" VLLM_CACHE_ROOT="$VP_DATA/cache/vllm"
mkdir -p "$TMPDIR"
# Process pools inside the reward: avoid BLAS oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
# DiffusionNFT's flow_grpo/ has no __init__.py, so `pip install -e` exposes nothing: put the clone on PYTHONPATH.
export PYTHONPATH="$VP_NFT_DIR${PYTHONPATH:+:$PYTHONPATH}"
[ -x "$VP_ENV/bin/python" ] && export PATH="$VP_ENV/bin:$PATH"
true
