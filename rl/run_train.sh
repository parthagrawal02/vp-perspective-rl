#!/bin/bash
# PROVISIONAL. DiffusionNFT training with the frozen perspective reward (config/nft.py:sd3_vp from rl/diffusionnft_vp.patch).
# This is the Task 12 smoke-test launcher (experiments/task12/run_smoke.sh) with paths from config/env.sh.
# usage: GPUS=1,3,5,7 EPOCHS=3 RUN=smoke bash rl/run_train.sh       (check `nvidia-smi` first; use free GPUs only)
# Micro-batch 3 x 96 gradient accumulation = the same 1152-sample optimizer step as the default 6 x 48, which OOMs on 48 GB GPUs
# with the reward models (DeepLSD, PickScore, aesthetic CLIP) resident.
# Needs: rl/make_dataset.py and rl/lcbase.py outputs in $VP_NFT_DIR/dataset/vp_hypersim/ (train.txt, test.txt, lc_base.json).
set -euo pipefail
source "$(dirname "$0")/../config/env.sh"
GPUS=${GPUS:?set GPUS=comma-separated free GPU ids}; NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export WANDB_MODE=${WANDB_MODE:-offline} NFT_N_GPUS=$NG VP_NWORKERS=${VP_NWORKERS:-12} CUDA_VISIBLE_DEVICES=$GPUS
export VP_LC_BASE=${VP_LC_BASE:-$VP_NFT_DIR/dataset/vp_hypersim/lc_base.json}
cd "$VP_NFT_DIR"
echo "START $(date) GPUS=$CUDA_VISIBLE_DEVICES"
"$VP_ENV/bin/torchrun" --nproc_per_node=$NG --master_port=${PORT:-29517} scripts/train_nft_sd3.py \
  --config config/nft.py:sd3_vp \
  --config.num_epochs=${EPOCHS:-3} \
  --config.train.batch_size=3 --config.train.gradient_accumulation_steps=96 \
  --config.logdir="$VP_DATA/outputs/rl/${RUN:-run}/nft_logs" \
  --config.save_dir="$VP_DATA/checkpoints/rl/${RUN:-run}" ${EXTRA:-}
echo "EXIT $? $(date)"
