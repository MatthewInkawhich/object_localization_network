#!/bin/bash


### Round 0
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc90_r0.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc90_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc90_r0/latest.pth 'nonwarship' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc90_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc90_r0/latest.pth 'warship' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc90_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc90_r0/latest.pth 'allship' 3


### Round 1
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_collect_preds.sh configs/oln_box_ships/round0/warship_cz_hybrid_lc90_r0.py out/oln_box_ships/round0/warship_cz_hybrid_lc90_r0/latest.pth 0.5 0 3
python -u tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round0/warship_cz_hybrid_lc90_r0/preds_round0.bbox.json 'warship' .30 --restricted

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30/latest.pth 'nonwarship' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30/latest.pth 'warship_nou' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30/latest.pth 'allship_nou' 3
echo "DONE ROUND1"


### Round 2
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_collect_preds.sh configs/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30.py out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30/latest.pth 0.5 1 3
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round1/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r1_p30/preds_round1.bbox.json 'warship' .30 --restricted

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30/latest.pth 'nonwarship' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30/latest.pth 'warship_nou' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30/latest.pth 'allship_nou' 3
echo "DONE ROUND2"


### Round 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_collect_preds.sh configs/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30.py out/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30/latest.pth 0.5 2 3
python tools/merge_annotations_by_percent.py data/ShipRSImageNet/COCO_Format/ShipRSImageNet_bbox_train_level_3.json out/oln_box_ships/round2/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r2_p30_p30/preds_round2.bbox.json 'warship' .30 --restricted

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh configs/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30.py 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'nonwarship' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'warship_nou' 3
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30.py out/oln_box_ships/round3/restricted_warship_cz_hybrid_lc90_lateqflwbbl2_r3_p30_p30_p30/latest.pth 'allship_nou' 3
echo "DONE ROUND3"
