#!/bin/bash
# 1-GPU pipeline test of the RL training loop (sampling -> reward -> training steps), with config.debug=True (no eval, no
# checkpoints) and a CONSTANT Lc_base map (pipeline test only, not a calibrated floor). Small epochs: 8 prompts x 6 images.
# TRAIN_BS (default 2) sets the micro-batch; peak GPU memory is sampled every 20 s. Run inside an interactive GPU job (≤ 50 min, 1 A100 = at most ~1.7 units):
#   srun -A bjop-delta-gpu -p gpuA100x4-interactive --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=16 --mem=62G \
#        --time=00:50:00 bash rl/delta/early_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
module unload pytorch-conda 2>/dev/null || true
source config/env.sh
D="$VP_NFT_DIR/dataset/vp_hypersim"; LB="$D/lc_base_constant_pipeline_test.json"
"$VP_ENV/bin/python" - "$D" "$LB" <<'EOF'
import json, sys
d, out = sys.argv[1:]
P = [l.rstrip("\n") for f in ("train.txt", "test.txt") for l in open(f"{d}/{f}") if l.strip()]
json.dump({p: 13000.0 for p in P}, open(out, "w"))
print("constant Lc_base = 13000 for", len(P), "prompts (pipeline test only)")
EOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export VP_REWARD="${VP_REWARD:-v2}" VP_LC_BASE="$LB" NFT_K=6 NFT_GROUPS=8 VP_NWORKERS=8 WANDB_MODE=offline HF_HUB_OFFLINE=1
TB="${TRAIN_BS:-2}"; export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > early_test_gpumem.log 2>&1 &
MP=$!
# always stop the memory logger (else it keeps the GPU job alive, and billed, until the time limit) and report the peak
trap 'kill $MP 2>/dev/null; echo "peak GPU memory (MiB): $(sort -n early_test_gpumem.log | tail -1)"' EXIT
GPUS=0 EPOCHS="${EPOCHS:-2}" RUN=early_test TRAIN_BS=$TB TRAIN_ACC=$((48 / TB)) EXTRA="--config.debug=True" bash rl/run_train.sh
echo EARLY_TEST_DONE
