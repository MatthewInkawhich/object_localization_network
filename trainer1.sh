#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/voc_imagenet_combined_r1_s74.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_imagenet_combined_r1_s74.py out/oln_box/round1/voc_imagenet_combined_r1_s74/latest.pth 4
