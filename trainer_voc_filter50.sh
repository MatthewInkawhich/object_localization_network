#!/bin/bash


### Round 1
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round0/voc_cz_2x_r0/preds_round0.bbox.json 'voc' 1.0 --oracle-anns data/coco/annotations/instances_train2017.json --oracle-filter-percent 0.50

bash tools/dist_train.sh configs/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100.py out/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100.py out/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100.py out/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100.py out/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100/latest.pth 0.6 1 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/filter50_voc_cz_lateqflwbbl2_2x_r1_p100/preds_round1.bbox.json 'voc' 1.0 --oracle-anns data/coco/annotations/instances_train2017.json --oracle-filter-percent 0.50

bash tools/dist_train.sh configs/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100.py out/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100.py out/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100.py out/oln_box/round2/filter50_voc_cz_lateqflwbbl2_2x_r2_p100_p100/latest.pth 'all_nou' 4
echo "DONE ROUND2"

