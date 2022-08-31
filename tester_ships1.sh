#!/bin/bash

# Round 0
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/warship_class_agn_faster_rcnn.py out/oln_box_ships/baselines/warship_class_agn_faster_rcnn/latest.pth 'allship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/warship_oln.py out/oln_box_ships/baselines/warship_oln/latest.pth 'allship' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_r0.py out/oln_box_ships/round0/warship_cz_r0/latest.pth 'allship' 4


# Round 1
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_lateqflwbbl2_noft_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_lateqflwbbl2_noft_r1_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_noft_r1_p30.py out/oln_box_ships/round1/warship_cz_lateqflwbbl2_noft_r1_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/warship_cz_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 4

# Round 2
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_warship_cz_lateqflwbbl2_noft_r2_p30_p30.py out/oln_box_ships/round2/restricted_warship_cz_lateqflwbbl2_noft_r2_p30_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_noft_r2_p30_p30.py out/oln_box_ships/round2/warship_cz_lateqflwbbl2_noft_r2_p30_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_warship_cz_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_warship_cz_lateqflwbbl2_r2_p30_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/warship_cz_lateqflwbbl2_r2_p30_p30/latest.pth 'allship_nou' 4

# Round 3
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_warship_cz_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_warship_cz_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'allship_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/warship_cz_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'allship_nou' 4
