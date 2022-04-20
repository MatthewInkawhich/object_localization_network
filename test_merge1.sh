#!/bin/bash

python -u tools/merge_and_evaluate.py configs/oln_box/baselines/hcoco_oln_2x.py out/oln_box/baselines/hcoco_oln_2x/val_preds.pkl out/oln_box/baselines/hcoco_class_agn_faster_rcnn/val_preds.pkl 0.10
echo "finished_hcoco___oln_frcnn"
python -u tools/merge_and_evaluate.py configs/oln_box/round3/hcoco_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/hcoco_cz_lateqflwbbl2_2x_r3_p30_p30_p30/val_preds.pkl out/oln_box/baselines/hcoco_oln_2x/val_preds.pkl 0.10
echo "finished_hcoco___ftstpnp_oln"
python -u tools/merge_and_evaluate.py configs/oln_box/round3/hcoco_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/hcoco_cz_lateqflwbbl2_2x_r3_p30_p30_p30/val_preds.pkl out/oln_box/baselines/hcoco_class_agn_faster_rcnn/val_preds.pkl 0.10
echo "finished_hcoco___ftstpnp_frcnn"


python -u tools/merge_and_evaluate.py configs/oln_box/baselines/voc_oln_2x.py out/oln_box/baselines/voc_oln_2x/val_preds.pkl out/oln_box/baselines/voc_class_agn_faster_rcnn/val_preds.pkl 0.10
echo "finished_voc___oln_frcnn"
python -u tools/merge_and_evaluate.py configs/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30/val_preds.pkl out/oln_box/baselines/voc_oln_2x/val_preds.pkl 0.10
echo "finished_voc___ftstpnp_oln"
python -u tools/merge_and_evaluate.py configs/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30.py out/oln_box/round3/voc_cz_lateqflwbbl2_2x_r3_p30_p30_p30/val_preds.pkl out/oln_box/baselines/voc_class_agn_faster_rcnn/val_preds.pkl 0.10
echo "finished_voc___ftstpnp_frcnn"

