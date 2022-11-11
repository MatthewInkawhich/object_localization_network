#!/bin/bash


### Round 1
#bash tools/dist_collect_preds.sh configs/oln_box/round0/voc_cz_hybrid_lc10_2x_r0.py out/oln_box/round0/voc_cz_hybrid_lc10_2x_r0/latest.pth 0.5 0 4
python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round0/voc_cz_hybrid_lc10_2x_r0/preds_round0.bbox.json 'voc' .90 --restricted

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90/latest.pth 'nonvoc' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90/latest.pth 'voc_nou' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90/latest.pth 'all_nou' 3
echo "DONE ROUND1"


### Round 2
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_collect_preds.sh configs/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90.py out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90/latest.pth 0.5 1 3
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r1_p90/preds_round1.bbox.json 'voc' .90 --restricted

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90/latest.pth 'nonvoc' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90/latest.pth 'voc_nou' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90/latest.pth 'all_nou' 3
echo "DONE ROUND2"


### Round 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_collect_preds.sh configs/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90.py out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90/latest.pth 0.5 2 3
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round2/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r2_p90_p90/preds_round2.bbox.json 'voc' .90 --restricted

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90/latest.pth 'nonvoc' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90/latest.pth 'voc_nou' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p90_p90_p90/latest.pth 'all_nou' 3
echo "DONE ROUND3"
