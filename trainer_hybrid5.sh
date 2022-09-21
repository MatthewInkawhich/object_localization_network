#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0.py out/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0/latest.pth 'nonvoc5' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0.py out/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0/latest.pth 'voc5' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0.py out/oln_box/round0/voc5_cz_hybrid_lc1_2x_r0/latest.pth 'all' 3

