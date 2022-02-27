#!/bin/bash


### Round 1
bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p10.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p10.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p10/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p20.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p20.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p20/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p40.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p40.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p40/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p50.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p50.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p50/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p60.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p60.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p60/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p70.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p70.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p70/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p80.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p80.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p80/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p90.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p90.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p90/latest.pth 4

bash tools/dist_train.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p100.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p100.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p100/latest.pth 4
