#!/bin/bash
# Floor reference Lc_base for every training/test prompt, 4 shards on one 4x A40 node, then merge.
#     SAMPLER=validated sbatch rl/delta/lcbase.slurm      (40 steps, CFG 4.5: the setting the floor was validated with)
#     SAMPLER=train     sbatch --time=01:00:00 rl/delta/lcbase.slurm   (Task 12's training-rollout sampler, ~20 min)
# Resumable (validated sampler): resubmit if it hits the time limit. Needs dataset/vp_hypersim/{train,test}.txt.
#SBATCH --account=bjop-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G   # 4 x 62G = 248G is over what a node lets a job use; 240G is accepted
#SBATCH --time=04:00:00   # shorter limits backfill more easily
#SBATCH --job-name=vp-lcbase
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module unload pytorch-conda 2>/dev/null || true   # its PyTorch 2.8 libraries would shadow the env's torch 2.7.1
source config/env.sh
SAMPLER=${SAMPLER:-validated}
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k "$VP_ENV/bin/python" rl/lcbase.py $k 4 --sampler "$SAMPLER" > "slurm-lcbase-$SLURM_JOB_ID-$k.log" 2>&1 &
done
wait
"$VP_ENV/bin/python" rl/lcbase.py merge --sampler "$SAMPLER"
