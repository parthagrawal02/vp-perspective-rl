#!/bin/bash
# usage: GPUS="0 1" EXTRA="--res 1024 --seeds 0,1 --max_ids 300" run_gen3.sh <ds>
source "$(dirname "$0")/../../config/env.sh"  # [repo] was the machine-specific env.sh
DSN=$1; set -- $GPUS; N=$#; k=0; S=$(date +%s)
for g in $GPUS; do CUDA_VISIBLE_DEVICES=$g python $VP_REPO/experiments/task03/gen.py --ds $DSN --shard $k --nshard $N $EXTRA > $VP_DATA/logs/task3_gen_${DSN}_${RES:-512}_$k.log 2>&1 & k=$((k+1)); done
wait; echo "GEN_ALL_DONE $DSN wall_s=$(( $(date +%s)-S )) ngpu=$N"
