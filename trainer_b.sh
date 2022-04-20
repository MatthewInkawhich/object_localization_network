#!/bin/bash


### Round 2
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p30/preds_round1.bbox.json 492812 .40

bash tools/dist_train.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40.py out/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40.py out/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40.py out/oln_box/round2/voc_cz_lateqflwbbl2_noft_2x_r2_p30_p40/latest.pth 'all_nou' 4


