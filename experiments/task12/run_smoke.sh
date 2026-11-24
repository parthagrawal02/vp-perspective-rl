#!/bin/bash
# Task 12 smoke test: DiffusionNFT sd3_vp on 4 free GPUs, ~3 epochs.
# Micro-batch 3 x 96 accumulation = same 1152-sample optimizer step as the default 6 x 48 (which OOMs on 48 GB with the reward models resident). usage: GPUS=1,3,5,7 EPOCHS=3 run_smoke.sh
source "$(dirname "$0")/../../config/env.sh"  # [repo] was the machine-specific env.sh
export PYTHONPATH=$VP_NFT_DIR:$PYTHONPATH  # [repo] vp_reward + deeplsd are pip-installed
export WANDB_MODE=offline NFT_N_GPUS=4 VP_NWORKERS=12 CUDA_VISIBLE_DEVICES=${GPUS:-1,3,5,7}
cd $VP_NFT_DIR
echo "START $(date) GPUS=$CUDA_VISIBLE_DEVICES"
$VP_ENV/bin/torchrun --nproc_per_node=4 --master_port=29517 scripts/train_nft_sd3.py \
  --config config/nft.py:sd3_vp \
  --config.num_epochs=${EPOCHS:-3} \
  --config.train.batch_size=3 --config.train.gradient_accumulation_steps=96 \
  --config.logdir=$VP_DATA/outputs/task12/nft_logs \
  --config.save_dir=$VP_DATA/checkpoints/task12_smoke ${EXTRA}
echo "EXIT $? $(date)"
