#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/voc_imagenet_r1_s80NA.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_imagenet_r1_s80NA.py out/oln_box/round1/voc_imagenet_r1_s80NA/latest.pth 4
