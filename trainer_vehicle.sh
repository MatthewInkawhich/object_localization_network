#!/bin/bash


### Round1
#bash tools/dist_train.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s80.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s80.py out/oln_box/round1/vehicle_cropzoomx_r1_s80/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s78.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s78.py out/oln_box/round1/vehicle_cropzoomx_r1_s78/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s76.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s76.py out/oln_box/round1/vehicle_cropzoomx_r1_s76/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s74.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/vehicle_cropzoomx_r1_s74.py out/oln_box/round1/vehicle_cropzoomx_r1_s74/latest.pth 4

### Round2
#bash tools/dist_train.sh configs/oln_box/round2/vehicle_cropzoomx_r2_s78_s84.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/vehicle_cropzoomx_r2_s78_s84.py out/oln_box/round2/vehicle_cropzoomx_r2_s78_s84/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round2/vehicle_cropzoomx_r2_s78_s83.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/vehicle_cropzoomx_r2_s78_s83.py out/oln_box/round2/vehicle_cropzoomx_r2_s78_s83/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round2/vehicle_cropzoomx_r2_s78_s82.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/vehicle_cropzoomx_r2_s78_s82.py out/oln_box/round2/vehicle_cropzoomx_r2_s78_s82/latest.pth 4

### Round3
#bash tools/dist_train.sh configs/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s86.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s86.py out/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s86/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s85.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s85.py out/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s85/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s84.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s84.py out/oln_box/round3/vehicle_cropzoomx_r3_s78_s82_s84/latest.pth 4
