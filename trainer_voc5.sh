#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p10.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p10.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p10/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p20.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p20.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p20/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p40.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p40.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p40/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p50.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p50.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p50/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p60.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p60.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p60/latest.pth 4

