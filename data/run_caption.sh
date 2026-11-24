#!/bin/bash
# Caption one reference set on one free GPU with the vlm env (envs/setup_vlm.sh).  usage: GPU=1 bash data/run_caption.sh <ds>
# The three exports are needed on Blackwell (sm_120): FlashInfer JIT needs nvcc from the CUDA 13 wheel, ninja from the env, and
# FlashInfer's sampling kernel fails to build on sm_120 so vLLM's own sampler is used.
set -euo pipefail
source "$(dirname "$0")/../config/env.sh"
export CUDA_HOME="$VP_VLM_ENV/lib/python3.12/site-packages/nvidia/cu13" PATH="$VP_VLM_ENV/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU=<free gpu id>}
"$VP_VLM_ENV/bin/python" "$(dirname "$0")/caption.py" "$1"
