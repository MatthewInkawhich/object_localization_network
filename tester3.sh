#!/bin/bash

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/voc_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'all_nou' 4



bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'nonvoc5' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'voc5_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/voc5_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc5_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/voc5_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'nonvoc5' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc5_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/voc5_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'voc5_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc5_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/voc5_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'nonvoc5' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'voc5_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'all_nou' 4



bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'animal_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30.py out/oln_box/round1/animal_cz_lateqflwbbl2_2x_r1_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'animal_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/animal_cz_lateqflwbbl2_2x_r2_p30_p30.py out/oln_box/round2/animal_cz_lateqflwbbl2_2x_r2_p30_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'nonanimal' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'animal_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 'all_nou' 4
