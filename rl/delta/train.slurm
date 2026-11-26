#!/bin/bash
# DiffusionNFT + perspective reward on one 4x A40 node (48 GB each: rl/run_train.sh already uses the micro-batch 3 x 96 that
# fits 48 GB). Smoke test first, then a real run:
#     EPOCHS=3 RUN=smoke VP_REWARD=v1 sbatch --time=01:00:00 rl/delta/train.slurm
#     EPOCHS=150 RUN=v2_run1 VP_REWARD=v2 VP_LC_BASE=... sbatch rl/delta/train.slurm
# ~7.5 min/epoch + periodic eval on 4 GPUs (Task 12); A40 batch costs 0.5x per GPU-hour -> a 24 h job ~ 48 units.
#SBATCH --account=bjop-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=248G
#SBATCH --time=24:00:00
#SBATCH --job-name=vp-nft
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module unload pytorch-conda 2>/dev/null || true   # its PyTorch 2.8 libraries would shadow the env's torch 2.7.1
source config/env.sh
export VP_REWARD=${VP_REWARD:-v1}
export VP_LC_BASE=${VP_LC_BASE:-$VP_NFT_DIR/dataset/vp_hypersim/lc_base_validated.json}
[ -s "$VP_LC_BASE" ] || { echo "missing Lc_base map $VP_LC_BASE (run rl/delta/lcbase.slurm)"; exit 1; }
echo "VP_REWARD=$VP_REWARD VP_LC_BASE=$VP_LC_BASE EPOCHS=${EPOCHS:-3} RUN=${RUN:-run}"
GPUS=0,1,2,3 PORT=$((29500 + SLURM_JOB_ID % 1000)) bash rl/run_train.sh
