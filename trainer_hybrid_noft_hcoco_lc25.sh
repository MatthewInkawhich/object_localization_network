#!/bin/bash


### Round 1
#bash tools/dist_collect_preds.sh configs/oln_box/round0/hcoco_cz_hybrid_lc25_2x_r0.py out/oln_box/round0/hcoco_cz_hybrid_lc25_2x_r0/latest.pth 0.5 0 4
#python -u tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round0/hcoco_cz_hybrid_lc25_2x_r0/preds_round0.bbox.json 'hcoco' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'nonhcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'hcoco_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30/latest.pth 'all_nou' 4
echo "DONE ROUND1"


### Round 2
bash tools/dist_collect_preds.sh configs/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30.py out/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30/latest.pth 0.5 1 4
python tools/merge_annotations_by_percent.py data/coco/annotations/instances_train2017.json out/oln_box/round1/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r1_p30/preds_round1.bbox.json 'hcoco' .30 --restricted

bash tools/dist_train.sh configs/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'nonhcoco' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'hcoco_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30.py out/oln_box/round2/restricted_hcoco_cz_hybrid_lc25_lateqflwbbl2_noft_2x_r2_p30_p30/latest.pth 'all_nou' 4
echo "DONE ROUND2"
