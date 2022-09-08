#!/bin/bash

#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/baselines/coco_class_agn_faster_rcnn.py 3
#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/coco_class_agn_faster_rcnn.py out/oln_box/baselines/coco_class_agn_faster_rcnn/latest.pth 'all' 3

#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/baselines/coco_oln_2x.py 3
#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/coco_oln_2x.py out/oln_box/baselines/coco_oln_2x/latest.pth 'all' 3

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/round0/coco_cz_2x_r0.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/coco_cz_2x_r0.py out/oln_box/round0/coco_cz_2x_r0/latest.pth 'all' 3
