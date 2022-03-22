#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
EVALCLASS=$3
GPUS=$4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --evalclass $EVALCLASS --launcher pytorch ${@:5} \
    --eval bbox
