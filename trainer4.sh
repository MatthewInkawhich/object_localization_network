#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/voc_cz_lateqfl_2x_r1_p60.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cz_lateqfl_2x_r1_p60.py out/oln_box/round1/voc_cz_lateqfl_2x_r1_p60/latest.pth 4

