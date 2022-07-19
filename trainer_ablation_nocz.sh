#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/baselines/voc_oln_2x/preds_round0.bbox.json 'voc' .30

bash tools/dist_train.sh configs/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/voc_lateqflwbbl2_noft_2x_r1_p30/preds_round1.bbox.json 'voc' .30

bash tools/dist_train.sh configs/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/voc_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND2"

