#!/bin/bash

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/vehicle_cz_2x_r0.py out/oln_box/round0/vehicle_cz_2x_r0/latest.pth 'vehicle' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/vehicle_cz_2x_r0.py out/oln_box/round0/vehicle_cz_2x_r0/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/vehicle_cz_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/vehicle_cz_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'vehicle_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/vehicle_cz_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/vehicle_cz_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'all_nou' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/vehicle_cz_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/vehicle_cz_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'vehicle_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/vehicle_cz_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/vehicle_cz_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'all_nou' 4
