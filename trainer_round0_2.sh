#!/bin/bash

#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/baselines/allship_class_agn_faster_rcnn.py 3
#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/allship_class_agn_faster_rcnn.py out/oln_box_ships/baselines/allship_class_agn_faster_rcnn/latest.pth 'allship' 3

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/baselines/allship_oln.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/allship_oln.py out/oln_box_ships/baselines/allship_oln/latest.pth 'allship' 3

#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/round0/allship_cz_r0.py 3
#CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/allship_cz_r0.py out/oln_box_ships/round0/allship_cz_r0/latest.pth 'allship' 3
