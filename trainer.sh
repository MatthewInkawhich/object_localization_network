#!/bin/bash

#bash tools/dist_train.sh configs/oln_box/person_split.py 4
#bash tools/dist_train.sh configs/oln_box/vehicle_split.py 4
#bash tools/dist_train.sh configs/oln_box/animal_split.py 4
#bash tools/dist_train.sh configs/oln_box/food_split.py 4

#bash tools/dist_train.sh configs/oln_box/aug_cropzoom.py 4
#bash tools/dist_train.sh configs/oln_box/aug_rotate.py 4
#bash tools/dist_train.sh configs/oln_box/aug_defocusblur.py 4
#bash tools/dist_train.sh configs/oln_box/aug_gaussiannoise.py 4
# Photometric distortion
#bash tools/dist_train.sh configs/oln_box/aug_photometricdistortion.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_photometricdistortion.py out/oln_box/aug_photometricdistortion/latest.pth 4
# CropzoomX
#bash tools/dist_train.sh configs/oln_box/aug_cropzoomx.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_cropzoomx.py out/oln_box/aug_cropzoomx/latest.pth 4
# RandomAffine
#bash tools/dist_train.sh configs/oln_box/aug_randomaffine.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_randomaffine.py out/oln_box/aug_randomaffine/latest.pth 4
# RandomAffineX
#bash tools/dist_train.sh configs/oln_box/aug_randomaffinex.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_randomaffinex.py out/oln_box/aug_randomaffinex/latest.pth 4
# CropzoomX & Gaussiannoise
#bash tools/dist_train.sh configs/oln_box/aug_cropzoomx_gaussiannoise.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_cropzoomx_gaussiannoise.py out/oln_box/aug_cropzoomx_gaussiannoise/latest.pth 4
# CropzoomX & PMD
#bash tools/dist_train.sh configs/oln_box/aug_cropzoomx_photometricdistortion.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/aug_cropzoomx_photometricdistortion.py out/oln_box/aug_cropzoomx_photometricdistortion/latest.pth 4
