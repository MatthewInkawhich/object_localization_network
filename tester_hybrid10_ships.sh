#!/bin/bash


bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonwarship' 0 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'warship_nou' 0 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 0 4
echo "DONE warship lc=0"

bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonwarship' 0.25 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'warship_nou' 0.25 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 0.25 4
echo "DONE warship lc=0.25"

bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonwarship' 0.5 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'warship_nou' 0.5 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 0.5 4
echo "DONE warship lc=0.5"




bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonmerchant' 0 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'merchant_nou' 0 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 0 4
echo "DONE merchant lc=0"

bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonmerchant' 0.25 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'merchant_nou' 0.25 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 0.25 4
echo "DONE merchant lc=0.25"

bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'nonmerchant' 0.5 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'merchant_nou' 0.5 4
bash tools/dist_test_bbox_evalclass_testlc.sh configs/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_merchant_cz_hybrid_lc10_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 0.5 4
echo "DONE merchant lc=0.5"