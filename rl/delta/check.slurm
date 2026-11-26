#!/bin/bash
# 20-minute check on 1 A40 that the whole reward stack runs on a compute node: CUDA, unit tests, both reward versions.
#     sbatch rl/delta/check.slurm
#SBATCH --account=bjop-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=62G
#SBATCH --time=00:20:00
#SBATCH --job-name=vp-check
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module unload pytorch-conda 2>/dev/null || true   # its PyTorch 2.8 libraries would shadow the env's torch 2.7.1
source config/env.sh
PY="$VP_ENV/bin/python"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
(cd vp_reward && "$PY" -m pytest -q tests -k "not repro")   # from the package dir: the repo-root vp_reward/ folder shadows the package
export PYTHONPATH="$VP_NFT_DIR${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1   # flow_grpo scorers; models are pre-fetched
"$PY" rl/delta/check_reward.py
