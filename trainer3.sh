#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/voc_cz_lateqfl210wbbl2_2x_r1_p20.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cz_lateqfl210wbbl2_2x_r1_p20.py out/oln_box/round1/voc_cz_lateqfl210wbbl2_2x_r1_p20/latest.pth 4
