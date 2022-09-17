#!/bin/bash

CUDA_VISIBLE_DEVICES=1 bash tools/dist_train.sh configs/oln_box/round0/voc_cz_hybrid_fl_lc10_2x_r0.py 1
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/hcoco_cz_hybrid_lc10_2x_r0.py out/oln_box/round0/hcoco_cz_hybrid_lc10_2x_r0/latest.pth 'nonhcoco' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/hcoco_cz_hybrid_lc10_2x_r0.py out/oln_box/round0/hcoco_cz_hybrid_lc10_2x_r0/latest.pth 'hcoco' 4
#bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/hcoco_cz_hybrid_lc10_2x_r0.py out/oln_box/round0/hcoco_cz_hybrid_lc10_2x_r0/latest.pth 'all' 4
