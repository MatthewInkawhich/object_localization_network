#!/bin/bash

### Faster R-CNN baselines
#bash tools/dist_train.sh configs/oln_box/baselines/animal_class_agn_faster_rcnn.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_class_agn_faster_rcnn.py out/oln_box/baselines/animal_class_agn_faster_rcnn/latest.pth 4

### Faster R-CNN baselines
bash tools/dist_train.sh configs/oln_box/baselines/animal_class_agn_faster_rcnn_2x.py 4
bash tools/dist_test_bbox.sh configs/oln_box/baselines/animal_class_agn_faster_rcnn_2x.py out/oln_box/baselines/animal_class_agn_faster_rcnn_2x/latest.pth 4





### Faster R-CNN baselines
#bash tools/dist_train.sh configs/oln_box/baselines/vehicle_class_agn_faster_rcnn.py 4
#bash tools/dist_test_bbox.sh configs/oln_box/baselines/vehicle_class_agn_faster_rcnn.py out/oln_box/baselines/vehicle_class_agn_faster_rcnn/latest.pth 4
