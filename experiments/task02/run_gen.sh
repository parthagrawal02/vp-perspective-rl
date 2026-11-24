#!/bin/bash
# usage: GPUS="0 1 2" run_gen.sh   (one process per GPU)
source "$(dirname "$0")/../../config/env.sh"  # [repo] was the machine-specific env.sh
set -- $GPUS; N=$#; k=0; S=$(date +%s)
for g in $GPUS; do CUDA_VISIBLE_DEVICES=$g python $VP_REPO/experiments/task02/gen.py --shard $k --nshard $N > $VP_DATA/logs/task2_gen_shard$k.log 2>&1 & k=$((k+1)); done
wait; echo "GEN_ALL_DONE wall_s=$(( $(date +%s)-S )) ngpu=$N"
