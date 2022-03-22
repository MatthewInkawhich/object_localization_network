#!/bin/bash

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/animal_cz_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'animal_nou' 4
echo "FINISHED ID"
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/animal_cz_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/animal_cz_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'all_nou' 4
echo "FINISHED ALL"

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/animal_cz_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/animal_cz_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'animal_nou' 4
echo "FINISHED ID"
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/animal_cz_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/animal_cz_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'all_nou' 4
echo "FINISHED ALL"


