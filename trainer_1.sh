#!/bin/bash


### TEST
#CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/oln_box/voc_test_focalloss.py 1
#bash tools/dist_train.sh configs/oln_box/voc_test_focalloss.py 4

### Round 0
#bash tools/dist_train.sh configs/oln_box/round0/voc_cropzoomx_qfl1.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/voc_cropzoomx_qfl1.py out/oln_box/round0/voc_cropzoomx_qfl1/latest.pth 4

### Round 1
#bash tools/dist_train.sh configs/oln_box/round1/voc_cropzoomx_qfl1_r1_p10.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_qfl1_r1_p10.py out/oln_box/round1/voc_cropzoomx_qfl1_r1_p10/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round1/voc5_cropzoomx_qfl_r1_p20.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cropzoomx_qfl_r1_p20.py out/oln_box/round1/voc5_cropzoomx_qfl_r1_p20/latest.pth 4


### Round 2
#bash tools/dist_train.sh configs/oln_box/round2/voc_cropzoomx_qfl1_r2_s80_p12.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/voc_cropzoomx_qfl1_r2_s80_p12.py out/oln_box/round2/voc_cropzoomx_qfl1_r2_s80_p12/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round2/voc5_cropzoomx_qfl_r2_p30_p35.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/voc5_cropzoomx_qfl_r2_p30_p35.py out/oln_box/round2/voc5_cropzoomx_qfl_r2_p30_p35/latest.pth 4

### Round 3
#bash tools/dist_train.sh configs/oln_box/round3/voc5_cropzoomx_qfl_r3_p30_p40_p50.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round3/voc5_cropzoomx_qfl_r3_p30_p40_p50.py out/oln_box/round3/voc5_cropzoomx_qfl_r3_p30_p40_p50/latest.pth 4

### Round 4
bash tools/dist_train.sh configs/oln_box/round4/voc5_cropzoomx_qfl_r4_p30_p40_p48_p50.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round4/voc5_cropzoomx_qfl_r4_p30_p40_p48_p50.py out/oln_box/round4/voc5_cropzoomx_qfl_r4_p30_p40_p48_p50/latest.pth 4
