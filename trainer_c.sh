#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round0/voc_cz_2x_r0/preds_round0.bbox.json 492812 .80

bash tools/dist_train.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80.py out/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80.py out/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80.py out/oln_box/round1/voc_cz_lateqflwbbl2_noft_2x_r1_p80/latest.pth 'all_nou' 4
echo "DONE ROUND1"


