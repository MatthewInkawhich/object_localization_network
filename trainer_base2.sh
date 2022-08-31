#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/baselines/merchant_class_agn_faster_rcnn.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/merchant_class_agn_faster_rcnn.py out/oln_box_ships/baselines/merchant_class_agn_faster_rcnn/latest.pth 'nonmerchant' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/merchant_class_agn_faster_rcnn.py out/oln_box_ships/baselines/merchant_class_agn_faster_rcnn/latest.pth 'merchant' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/merchant_class_agn_faster_rcnn.py out/oln_box_ships/baselines/merchant_class_agn_faster_rcnn/latest.pth 'all' 3
