#!/bin/bash

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_oln.py out/oln_box/baselines/voc_oln/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/baselines/voc_oln.py out/oln_box/baselines/voc_oln/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_cropzoom.py out/oln_box/aug_cropzoom/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_cropzoom.py out/oln_box/aug_cropzoom/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_discreterotate.py out/oln_box/aug_discreterotate/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_discreterotate.py out/oln_box/aug_discreterotate/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_randomaffine.py out/oln_box/aug_randomaffine/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_randomaffine.py out/oln_box/aug_randomaffine/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_gaussiannoise.py out/oln_box/aug_gaussiannoise/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_gaussiannoise.py out/oln_box/aug_gaussiannoise/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_photometricdistortion.py out/oln_box/aug_photometricdistortion/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_photometricdistortion.py out/oln_box/aug_photometricdistortion/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_cropzoomx_gaussiannoise.py out/oln_box/aug_cropzoomx_gaussiannoise/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_cropzoomx_gaussiannoise.py out/oln_box/aug_cropzoomx_gaussiannoise/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_cropzoomx_photometricdistortion.py out/oln_box/aug_cropzoomx_photometricdistortion/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_cropzoomx_photometricdistortion.py out/oln_box/aug_cropzoomx_photometricdistortion/latest.pth 'all' 4



bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_discreterotate_2x.py out/oln_box/aug_discreterotate_2x/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_discreterotate_2x.py out/oln_box/aug_discreterotate_2x/latest.pth 'all' 4

bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_randomaffine_2x.py out/oln_box/aug_randomaffine_2x/latest.pth 'voc' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box/aug_randomaffine_2x.py out/oln_box/aug_randomaffine_2x/latest.pth 'all' 4
