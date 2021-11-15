#!/bin/bash


### TEST
#CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/oln_box/voc_test_focalloss.py 1
#bash tools/dist_train.sh configs/oln_box/voc_test_focalloss.py 4

### Round 0
#bash tools/dist_train.sh configs/oln_box/round0/voc_cropzoomx_qfl_b15.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/voc_cropzoomx_qfl_b15.py out/oln_box/round0/voc_cropzoomx_qfl_b15/latest.pth 4

### Round 1
#bash tools/dist_train.sh configs/oln_box/round1/voc_cropzoomx_qfl1_r1_p10.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_qfl1_r1_p10.py out/oln_box/round1/voc_cropzoomx_qfl1_r1_p10/latest.pth 4



### Round 3
#bash tools/dist_train.sh configs/oln_box/round3/voc_cropzoomx_qfl1_r3_s80_p12_p16.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round3/voc_cropzoomx_qfl1_r3_s80_p12_p16.py out/oln_box/round3/voc_cropzoomx_qfl1_r3_s80_p12_p16/latest.pth 4

### Round 4
#bash tools/dist_train.sh configs/oln_box/round4/voc_cropzoomx_qfl1_r4_s80_p14_p17_p20.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round4/voc_cropzoomx_qfl1_r4_s80_p14_p17_p20.py out/oln_box/round4/voc_cropzoomx_qfl1_r4_s80_p14_p17_p20/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round4/voc_cropzoomx_qfl1_r4_s80_p14_p17_p18.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round4/voc_cropzoomx_qfl1_r4_s80_p14_p17_p18.py out/oln_box/round4/voc_cropzoomx_qfl1_r4_s80_p14_p17_p18/latest.pth 4

### Round 5
#bash tools/dist_train.sh configs/oln_box/round5/voc_cropzoomx_qfl1_r5_s80_p14_p17_p20_p23.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round5/voc_cropzoomx_qfl1_r5_s80_p14_p17_p20_p23.py out/oln_box/round5/voc_cropzoomx_qfl1_r5_s80_p14_p17_p20_p23/latest.pth 4

### Round 6
bash tools/dist_train.sh configs/oln_box/round6/voc_cropzoomx_qfl1_r6_s80_p14_p17_p20_p23_p26.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round6/voc_cropzoomx_qfl1_r6_s80_p14_p17_p20_p23_p26.py out/oln_box/round6/voc_cropzoomx_qfl1_r6_s80_p14_p17_p20_p23_p26/latest.pth 4
