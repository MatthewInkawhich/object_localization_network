#!/bin/bash

bash tools/dist_train.sh configs/oln_box/baselines/coco_rpn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/coco_rpn.py out/oln_box/baselines/coco_rpn/latest.pth 'all' 4

