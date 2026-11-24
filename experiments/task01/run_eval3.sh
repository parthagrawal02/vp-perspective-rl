#!/bin/bash
# Runs DrawBench eval (scripts/evaluation.py) for base / FlowGRPO / DiffusionNFT, sequentially, on GPUs in $GPUS.
source "$(dirname "$0")/../../config/env.sh"  # [repo] was the machine-specific env.sh
cd $VP_NFT_DIR
GPUS=${GPUS:-0,1,2,3,4,5,6,7}; N=$(echo $GPUS | tr ',' '\n' | wc -l)
OUT=$VP_DATA/outputs/eval_drawbench
run() { name=$1; shift
  echo "=== $name start $(date +%s) $(date) GPUs=$GPUS"; s=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --master_port=29611 scripts/evaluation.py \
     --model_type sd3 --dataset drawbench --mixed_precision fp16 --num_inference_steps 40 --resolution 512 \
     --save_images --output_dir $OUT/$name "$@"
  echo "=== $name end rc=$? wall_s=$(( $(date +%s)-s ))"; }
run base_cfg4.5          --guidance_scale 4.5
run flowgrpo_pick_cfg4.5 --guidance_scale 4.5 --lora_hf_path jieliu/SD3.5M-FlowGRPO-PickScore
run nft_multi_cfg1.0     --guidance_scale 1.0 --lora_hf_path worstcoder/SD3.5M-DiffusionNFT-MultiReward
echo ALLDONE
