#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p28.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p28.py out/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p28/latest.pth 4


