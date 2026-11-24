#!/bin/bash
# Captioning env "vlm" (vLLM 0.33.0, python 3.12, torch 2.14+cu130) for data/caption.py only. Separate from nft-vp
# because vLLM needs a much newer torch/transformers. Usage: source config/env.sh; bash envs/setup_vlm.sh
set -euo pipefail
: "${VP_DATA:?source config/env.sh first}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; E="${VP_VLM_ENV:?}"; MAMBA="${MAMBA:-micromamba}"
[ -x "$E/bin/python" ] || "$MAMBA" create -y -p "$E" -c conda-forge python=3.12.16
"$E/bin/python" -m pip install vllm==0.33.0
"$E/bin/python" -m pip install -r "$REPO/envs/requirements-vlm.lock.txt"   # pins the rest to the exact versions used
"$E/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__); print('VLM_ENV_OK')"
# Runtime settings needed on Blackwell (sm_120), set by data/caption.py's launcher data/run_caption.sh:
#   CUDA_HOME=$E/lib/python3.12/site-packages/nvidia/cu13   (FlashInfer JIT needs nvcc from the wheel)
#   PATH=$E/bin:$PATH                                       (ninja)
#   VLLM_USE_FLASHINFER_SAMPLER=0                           (FlashInfer sampling kernel JIT build fails on sm_120)
