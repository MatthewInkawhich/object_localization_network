#!/bin/bash

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p80_p80_p80.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p80_p80_p80/latest.pth 'nonvoc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p80_p80_p80.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p80_p80_p80/latest.pth 'voc_nou' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p80_p80_p80.py out/oln_box/round3/restricted_voc_cz_hybrid_lc10_lateqflwbbl2_2x_r3_p80_p80_p80/latest.pth 'all_nou' 4
echo "DONE ROUND3"
