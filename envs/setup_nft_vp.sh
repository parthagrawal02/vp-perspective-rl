#!/bin/bash
# Build the training / reward env "nft-vp" from scratch (python 3.10, torch 2.7.1+cu128, our pins, DeepLSD, patched DiffusionNFT,
# vp_reward). Usage (from the repo root):   cp config/env.example.sh config/env.sh; $EDITOR config/env.sh
#                                           source config/env.sh; bash envs/setup_nft_vp.sh
# Needs: micromamba (or conda/mamba: set MAMBA=conda), git, wget, an NVIDIA driver for CUDA 12.8+. No system CUDA toolkit needed.
set -euo pipefail
: "${VP_DATA:?source config/env.sh first}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
E="${VP_ENV:?}"; MAMBA="${MAMBA:-micromamba}"
NFT_COMMIT=cbb14f8; DEEPLSD_COMMIT=f7d9d6258c0cd25d4f6eea882853565403d289be
mkdir -p "$VP_DATA"/{models/deeplsd,models/reward_ckpts,envs,tmp} "$VP_THIRD_PARTY"

echo "== 1. python 3.10.16 env at $E"
[ -x "$E/bin/python" ] || "$MAMBA" create -y -p "$E" -c conda-forge python=3.10.16
PY="$E/bin/python"

echo "== 2. torch 2.7.1+cu128 (Blackwell sm_120 needs a cu128+ build; upstream pins 2.6.0+cu126, which has no sm_120 kernels)"
"$PY" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "== 3. pinned packages (exact freeze of the env used for all results; --no-deps so pip never re-resolves torch)"
"$PY" -m pip install --no-deps -r "$REPO/envs/requirements-nft-vp.lock.txt"

echo "== 4. DeepLSD @ ${DEEPLSD_COMMIT:0:7} + pytlsd built from its submodule (commit 21381ca; reports version 0.0.3)"
[ -d "$VP_DEEPLSD_DIR/.git" ] || git clone https://github.com/cvg/DeepLSD.git "$VP_DEEPLSD_DIR"
git -C "$VP_DEEPLSD_DIR" checkout -q "$DEEPLSD_COMMIT"
git -C "$VP_DEEPLSD_DIR" submodule update --init --recursive third_party/pytlsd
# This pytlsd commit is not on PyPI. Its build pulls the newest cmake (4.x), which rejects the bundled pybind11's
# cmake_minimum_required; CMAKE_POLICY_VERSION_MINIMUM=3.5 is the workaround. Build from a copy so no build/ lands in the clone.
rm -rf "$VP_DATA/tmp/pytlsd_src"; cp -r "$VP_DEEPLSD_DIR/third_party/pytlsd" "$VP_DATA/tmp/pytlsd_src"; rm -rf "$VP_DATA/tmp/pytlsd_src/build"
CMAKE_POLICY_VERSION_MINIMUM=3.5 "$PY" -m pip install --no-deps "$VP_DATA/tmp/pytlsd_src"
"$PY" -m pip install --no-deps -e "$VP_DEEPLSD_DIR"

echo "== 5. DiffusionNFT @ $NFT_COMMIT + our patch (rl/diffusionnft_vp.patch)"
bash "$REPO/rl/apply_patch.sh"
"$PY" -m pip install --no-deps -e "$VP_NFT_DIR"

echo "== 6. vp_reward (editable)"
"$PY" -m pip install --no-deps -e "$REPO/vp_reward"

echo "== 7. checkpoints (skipped if present)"
[ -s "$VP_DEEPLSD_CKPT" ] || wget -q https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar -O "$VP_DEEPLSD_CKPT"
A="$VP_DATA/models/reward_ckpts/sac+logos+ava1-l14-linearMSE.pth"
[ -s "$A" ] || wget -q "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth" -O "$A"
[ -e "$VP_NFT_DIR/reward_ckpts" ] || ln -s "$VP_DATA/models/reward_ckpts" "$VP_NFT_DIR/reward_ckpts"
# PickScore (yuvalkirstain/PickScore_v1, laion/CLIP-ViT-H-14-laion2B-s32B-b79K) and openai/clip-vit-large-patch14 download into
# $HF_HOME on first use. SD3.5-Medium (training / generation only) is gated: accept the license on huggingface.co, then log in.

echo "== 8. verify"
"$PY" -m pip check || true   # expected notes only: diffusion-nft's setup.py pins torch 2.6.0, flash-attn, xformers, deepspeed (not installed)
"$PY" -c "import torch, deeplsd, pytlsd, kornia, omegaconf, vp_reward, cv2; from deeplsd.models.deeplsd_inference import DeepLSD; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch.cuda.get_arch_list()[-2:]); print('ENV_OK')"
