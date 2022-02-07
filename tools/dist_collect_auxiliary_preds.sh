#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
THRESH=$3
ROUND=$4
GPUS=$5
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/collect_preds.py $CONFIG $CHECKPOINT $THRESH $ROUND --auxiliary --launcher pytorch ${@:6} \
