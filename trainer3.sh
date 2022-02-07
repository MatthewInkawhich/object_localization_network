#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/voc_imagenet_combined_r1_s78.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_imagenet_combined_r1_s78.py out/oln_box/round1/voc_imagenet_combined_r1_s78/latest.pth 4
