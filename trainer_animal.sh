#!/bin/bash


### Round2
#bash tools/dist_train.sh configs/oln_box/round2/animal_cropzoomx_r2_s78_s86.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/animal_cropzoomx_r2_s78_s86.py out/oln_box/round2/animal_cropzoomx_r2_s78_s86/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round2/animal_cropzoomx_r2_s78_s85.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/animal_cropzoomx_r2_s78_s85.py out/oln_box/round2/animal_cropzoomx_r2_s78_s85/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round2/animal_cropzoomx_r2_s78_s84.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/animal_cropzoomx_r2_s78_s84.py out/oln_box/round2/animal_cropzoomx_r2_s78_s84/latest.pth 4

### Round3
bash tools/dist_train.sh configs/oln_box/round3/animal_cropzoomx_r3_s78_s86_s87.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round3/animal_cropzoomx_r3_s78_s86_s87.py out/oln_box/round3/animal_cropzoomx_r3_s78_s86_s87/latest.pth 4
bash tools/dist_train.sh configs/oln_box/round3/animal_cropzoomx_r3_s78_s86_s86.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round3/animal_cropzoomx_r3_s78_s86_s86.py out/oln_box/round3/animal_cropzoomx_r3_s78_s86_s86/latest.pth 4
bash tools/dist_train.sh configs/oln_box/round3/animal_cropzoomx_r3_s78_s86_s85.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round3/animal_cropzoomx_r3_s78_s86_s85.py out/oln_box/round3/animal_cropzoomx_r3_s78_s86_s85/latest.pth 4
