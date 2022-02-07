#!/bin/bash

#bash tools/dist_test_bbox.sh configs/oln_box/voc5_split.py out/oln_box/voc5_split/latest.pth 4
#bash tools/dist_test_bbox.sh configs/oln_box/round0/voc5_cropzoomx.py out/oln_box/round0/voc5_cropzoomx/latest.pth 4

#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc5_cropzoomx_qfl_r1_p30.py out/oln_box/round1/voc5_cropzoomx_qfl_r1_p30/latest.pth 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/voc5_cropzoomx_qfl_r2_p30_p40.py out/oln_box/round2/voc5_cropzoomx_qfl_r2_p30_p40/latest.pth 4

#bash tools/dist_test_bbox.sh configs/oln_box/round3/voc5_cropzoomx_qfl_r3_p30_p40_p48.py out/oln_box/round3/voc5_cropzoomx_qfl_r3_p30_p40_p48/latest.pth 4

bash tools/dist_test_bbox.sh configs/oln_box/round4/voc5_cropzoomx_qfl_r4_p30_p40_p48_p50.py out/oln_box/round4/voc5_cropzoomx_qfl_r4_p30_p40_p48_p50/latest.pth 4
