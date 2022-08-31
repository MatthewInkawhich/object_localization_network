#!/bin/bash

bash tools/dist_train.sh configs/oln_box_ships/baselines/merchant_oln.py 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/merchant_oln.py out/oln_box_ships/baselines/merchant_oln/latest.pth 'nonmerchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/merchant_oln.py out/oln_box_ships/baselines/merchant_oln/latest.pth 'merchant' 4
bash tools/dist_test_bbox_evalclass.sh configs/oln_box_ships/baselines/merchant_oln.py out/oln_box_ships/baselines/merchant_oln/latest.pth 'all' 4
