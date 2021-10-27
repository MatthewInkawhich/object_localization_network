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
# VOC5 CZX
#bash tools/dist_train.sh configs/oln_box/voc5_cropzoomx.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/voc5_cropzoomx.py out/oln_box/voc5_cropzoomx/latest.pth 4


### Round2
#bash tools/dist_train.sh configs/oln_box/round2/aug_cropzoomx_r2_s76.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round2/aug_cropzoomx_r2_s76.py out/oln_box/round2/aug_cropzoomx_r2_s76/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round2/voc_split_r2_s78.py 4
#bash tools/dist_train.sh configs/oln_box/round2/voc_split_r2_s76.py 4
#bash tools/dist_train.sh configs/oln_box/round2/voc_split_r2_s74.py 4

### Round3
#bash tools/dist_train.sh configs/oln_box/round3/voc_cropzoomx_r3_s80_s84_s88.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round3/voc_cropzoomx_r3_s80_s84_s88.py out/oln_box/round3/voc_cropzoomx_r3_s80_s84_s88/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round3/voc_cropzoomx_r3_s80_s84_s86.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round3/voc_cropzoomx_r3_s80_s84_s86.py out/oln_box/round3/voc_cropzoomx_r3_s80_s84_s86/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round3/voc_cropzoomx_r3_s80_s84_s84.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round3/voc_cropzoomx_r3_s80_s84_s84.py out/oln_box/round3/voc_cropzoomx_r3_s80_s84_s84/latest.pth 4



### Finetune
### Round1
#bash tools/dist_train.sh configs/oln_box/round1/voc_cropzoomx_ft4_r1_s80.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_ft4_r1_s80.py out/oln_box/round1/voc_cropzoomx_ft4_r1_s80/latest.pth 4
#bash tools/dist_train.sh configs/oln_box/round1/voc_cropzoomx_ft6_r1_s80.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_ft6_r1_s80.py out/oln_box/round1/voc_cropzoomx_ft6_r1_s80/latest.pth 4

### Round2
bash tools/dist_train.sh configs/oln_box/round2/voc_cropzoomx_ft4_r2_s80_s84.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_ft4_r2_s80_s84.py out/oln_box/round2/voc_cropzoomx_ft4_r2_s80_s84/latest.pth 4
bash tools/dist_train.sh configs/oln_box/round2/voc_cropzoomx_ft4_r2_s80_s83.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_ft4_r2_s80_s83.py out/oln_box/round2/voc_cropzoomx_ft4_r2_s80_s83/latest.pth 4
bash tools/dist_train.sh configs/oln_box/round2/voc_cropzoomx_ft4_r2_s80_s82.py 4
bash tools/dist_test_bbox.sh configs/oln_box/round1/voc_cropzoomx_ft4_r2_s80_s82.py out/oln_box/round2/voc_cropzoomx_ft4_r2_s80_s82/latest.pth 4



