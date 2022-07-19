#!/bin/bash

#bash tools/dist_collect_val_preds.sh configs/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 4
#echo "finished_voc5__collectvalpreds"

#bash tools/dist_collect_val_preds.sh configs/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/latest.pth 4
#echo "finished_animal__collectvalpreds"



python -u tools/merge_and_evaluate.py configs/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc5_cz_lateqflwbbl2_2x_r3_p30_p30_p30/val_preds.pkl out/oln_box/baselines/voc5_class_agn_faster_rcnn/val_preds.pkl 0.10
echo "finished_voc5___ftstpnp_frcnn"

python -u tools/merge_and_evaluate.py configs/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/animal_cz_lateqflwbbl2_2x_r3_p30_p30_p30/val_preds.pkl out/oln_box/baselines/animal_class_agn_faster_rcnn/val_preds.pkl 0.10
echo "finished_animal___ftstpnp_frcnn"


