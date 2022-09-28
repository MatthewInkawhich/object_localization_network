#!/bin/bash


### Round 0
bash tools/dist_train.sh configs/oln_box/round0/voc_cz_hybrid_lc10_1x_r0.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/voc_cz_hybrid_lc10_1x_r0.py out/oln_box/round0/voc_cz_hybrid_lc10_1x_r0/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/voc_cz_hybrid_lc10_1x_r0.py out/oln_box/round0/voc_cz_hybrid_lc10_1x_r0/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round0/voc_cz_hybrid_lc10_1x_r0.py out/oln_box/round0/voc_cz_hybrid_lc10_1x_r0/latest.pth 'all' 4
echo "DONE ROUND0"


### Round 1
bash tools/dist_collect_preds.sh configs/oln_box/round0/voc_cz_hybrid_lc10_1x_r0.py out/oln_box/round0/voc_cz_hybrid_lc10_1x_r0/latest.pth 0.5 0 4
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round0/voc_cz_hybrid_lc10_1x_r0/preds_round0.bbox.json 'voc' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30/latest.pth 0.5 1 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r1_p30/preds_round1.bbox.json 'voc' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND2"


### Round 3
bash tools/dist_collect_preds.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30/latest.pth 0.5 2 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r2_p30_p30/preds_round2.bbox.json 'voc' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_1x_r3_p30_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND3"
