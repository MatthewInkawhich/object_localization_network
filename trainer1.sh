#!/bin/bash

#bash tools/dist_train.sh configs/openset_det/coco_faster_rcnn.py 4
bash tools/dist_test_bbox_evalclass.sh configs/openset_det/coco_faster_rcnn.py out/openset_det/coco_faster_rcnn/latest.pth 'all' 4
