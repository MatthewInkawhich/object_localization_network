#!/bin/bash


# Round 1
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter25_merchant_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter25_merchant_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter50_merchant_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter50_merchant_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter75_merchant_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter75_merchant_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/filter100_merchant_cz_lateqflwbbl2_noft_r1_p100.py out/oln_box_ships/round1/filter100_merchant_cz_lateqflwbbl2_noft_r1_p100/latest.pth 'allship_nou' 4


# Round 2
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter25_merchant_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter25_merchant_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter50_merchant_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter50_merchant_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter75_merchant_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter75_merchant_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/filter100_merchant_cz_lateqflwbbl2_noft_r2_p100_p100.py out/oln_box_ships/round2/filter100_merchant_cz_lateqflwbbl2_noft_r2_p100_p100/latest.pth 'allship_nou' 4
