#!/bin/bash
# usage: GPUS="2 3 4" run_dlsd.sh <ds>
source "$(dirname "$0")/../../config/env.sh"  # [repo] was the machine-specific env.sh; export PYTHONPATH=
DSNAME=$1; set -- $GPUS; N=$#; k=0
for g in $GPUS; do CUDA_VISIBLE_DEVICES=$g $VP_DATA/envs/deeplsd/bin/python $VP_REPO/experiments/task03/pipe.py dlsd $DSNAME $k $N > $VP_DATA/logs/task3_dlsd_${DSNAME}_$k.log 2>&1 & k=$((k+1)); done; wait; echo DLSD_ALL_DONE $DSNAME
