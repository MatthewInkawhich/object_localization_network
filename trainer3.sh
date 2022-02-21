#!/bin/bash


### Round 1
#bash tools/dist_train.sh configs/oln_box/round1/voc_imagenet_combined_r1_s78.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_imagenet_combined_r1_s78.py out/oln_box/round1/voc_imagenet_combined_r1_s78/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/voc_cz_lateqfl_2x_r1_p90.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cz_lateqfl_2x_r1_p90.py out/oln_box/round1/voc_cz_lateqfl_2x_r1_p90/latest.pth 4
bash tools/dist_train.sh configs/oln_box/round1/voc_cz_lateqfl_2x_r1_p100.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cz_lateqfl_2x_r1_p100.py out/oln_box/round1/voc_cz_lateqfl_2x_r1_p100/latest.pth 4
