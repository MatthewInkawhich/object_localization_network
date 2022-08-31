# MatthewInkawhich

"""
This script merges the "new" pseudo annotations from tools/generate_pseudolabels.py
with the source annotations. The output of this script is a new annotation file that is
ready to train on in the next stage.
"""
import argparse
import os
import json
import copy
import random
from collections import Counter

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
HCOCO_CLASSES = (
           'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
           'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
           'train', 'tv', 'traffic light', 'stop sign', 'bench', 'backpack',
           'handbag', 'skis', 'sports ball', 'skateboard', 'surfboard', 
           'fork', 'bowl', 'apple', 'pizza', 'toilet', 'laptop', 'remote',
           'oven', 'sink', 'refrigerator', 'toothbrush')
VOC_CLASSES = (
           'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 
           'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
           'train', 'tv')
VOC5_CLASSES = (
           'bicycle', 'car', 'chair', 'dog', 'person')
ANIMAL_CLASSES = (
           'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe')
HANIMAL_CLASSES = (
           'bird', 'cat', 'dog', 'horse', 'cow')
VEHICLE_CLASSES = (
           'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat')

SHIP_CLASSES = ('Other Ship', 'Other Warship', 'Submarine', 'Other Aircraft Carrier', 'Enterprise', 'Nimitz', 'Midway',
           'Ticonderoga',
           'Other Destroyer', 'Atago DD', 'Arleigh Burke DD', 'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Other Frigate',
           'Perry FF',
           'Patrol', 'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 'Osumi LL',
           'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Commander', 'Other Auxiliary Ship', 'Medical Ship', 'Test Ship',
           'Training Ship',
           'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF', 'Other Merchant', 'Container Ship', 'RoRo', 'Cargo',
           'Barge', 'Tugboat', 'Ferry', 'Yacht', 'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft',
           'Motorboat', 'Dock',)

WARSHIP_CLASSES = ('Other Warship', 'Submarine', 'Other Aircraft Carrier', 'Enterprise', 'Nimitz', 'Midway',
           'Ticonderoga',
           'Other Destroyer', 'Atago DD', 'Arleigh Burke DD', 'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Other Frigate',
           'Perry FF',
           'Patrol', 'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL', 'Austin LL', 'Osumi LL',
           'Wasp LL', 'LSD 41 LL', 'LHA LL', 'Commander', 'Other Auxiliary Ship', 'Medical Ship', 'Test Ship',
           'Training Ship',
           'AOE', 'Masyuu AS', 'Sanantonio AS', 'EPF',)

MERCHANT_CLASSES = ('Other Merchant', 'Container Ship', 'RoRo', 'Cargo',
           'Barge', 'Tugboat', 'Ferry', 'Yacht', 'Sailboat', 'Fishing Vessel', 'Oil Tanker', 'Hovercraft',
           'Motorboat',)

class_names_dict = {
    'all': CLASSES,
    'hcoco': HCOCO_CLASSES,
    'voc': VOC_CLASSES,
    'voc5': VOC5_CLASSES,
    'animal': ANIMAL_CLASSES,
    'hanimal': HANIMAL_CLASSES,
    'vehicle': VEHICLE_CLASSES,
    'warship': WARSHIP_CLASSES,
    'merchant': MERCHANT_CLASSES,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudolabels')
    parser.add_argument('source', help='source annotation path')
    parser.add_argument('preds', help='preds annotation path')
    parser.add_argument('split', type=str, help='ID class split')
    parser.add_argument('percent_new', type=float, help='percent increase from new pseudo-labels relative to original_id_ann_count')
    parser.add_argument('--restricted', action='store_true', help='non-(+): only generate use PLs from images already containing ID objects')
    parser.add_argument('--oracle-anns', type=str, help='use oracle pseudo-label filtering (human-in-the-loop), this is the path to ALL GT labels (ID&OOD)')
    parser.add_argument('--oracle-filter-percent', type=float, help='percent of bad pseudo-labels to filter out')
    args = parser.parse_args()
    return args


def xywh2xyxy(b):
    return [b[0], b[1], b[0]+b[2], b[1]+b[3]]


def bb_intersection_over_union(boxA, boxB):
    """
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def gt_overlap_exists(p, img2ann_map, iou_thr=0.7):
    """
    p: annotation dict
    img2ann_map: dict mapping image_ids to list of anns corresponding to that image_id
    """
    p_bbox_xyxy = xywh2xyxy(p['bbox'])
    if p['image_id'] in img2ann_map:
        overlap_exists = False
        for ann in img2ann_map[p['image_id']]:
            gt_box_xyxy = xywh2xyxy(ann['bbox'])
            iou = bb_intersection_over_union(p_bbox_xyxy, gt_box_xyxy)
            if iou >= iou_thr:
                #print('pred filtered by GT category:', ann['category_id'])
                return True
    return False
            

def subset_ann_count(annotations, subset):
    cats = []
    for ann in annotations['annotations']:
        curr_id = ann['category_id']
        for cat in annotations['categories']:
            if curr_id == cat['id']:
                cats.append(cat['name'])
        
    # Create dictionary of category and counts
    count_dict = dict(Counter(cats))

    # Count total annotations in subset
    total = 0
    for c in subset:
        total += count_dict[c]
    return total





def main():
    # Parse args
    args = parse_args()

    next_round = int(args.preds.split('/')[-1].split('_round')[-1].split('.bbox.json')[0]) + 1
    if args.percent_new < 1.0:
        str_percent_new = "{:.2f}".format(args.percent_new).split('.')[-1]
    else:
        str_percent_new = "{:.3f}".format(args.percent_new * .1).split('.')[-1]

    if "robustpreds" in args.preds.split('/')[-1]:
        jitter = args.preds.split('/')[-1].split('robustpreds')[-1].split('_round')[0]
        new_filepath = os.path.join(args.preds.split('/robustpreds')[0], f'robust{jitter}_annotations_for_round{next_round}_p{str_percent_new}.json')
    elif args.restricted:
        new_filepath = os.path.join(args.preds.split('/preds_round')[0], f'restricted_{args.split}_annotations_for_round{next_round}_p{str_percent_new}.json')
    elif args.oracle_anns:
        assert args.oracle_filter_percent >= 0 and args.oracle_filter_percent <= 1.0, "Error: oracle_filter_percent argument must be in [0,1]"
        if args.oracle_filter_percent < 1.0:
            str_oracle_filter_percent = "{:.2f}".format(args.oracle_filter_percent).split('.')[-1]
        else:
            str_oracle_filter_percent = "{:.3f}".format(args.oracle_filter_percent * .1).split('.')[-1]

        new_filepath = os.path.join(args.preds.split('/preds_round')[0], f'filter{str_oracle_filter_percent}_annotations_for_round{next_round}_p{str_percent_new}.json')
    else:
        new_filepath = os.path.join(args.preds.split('/preds_round')[0], f'annotations_for_round{next_round}_p{str_percent_new}.json')
    print("new_filepath:", new_filepath)


    # Read source and preds files
    with open(args.source, 'r') as f:
        source_contents = json.load(f)
    with open(args.preds, 'r') as f:
        preds_contents = json.load(f)
    if args.oracle_anns:
        with open(args.oracle_anns, 'r') as f:
            oracle_anns_contents = json.load(f)

    # Initialize new_contents
    new_contents = copy.deepcopy(source_contents)
    # Add new UNKNOWN category
    new_contents['categories'].append({
        'supercategory': 'UNKNOWN',
        'id': 999999999,
        'name': 'UNKNOWN',
    })
    for c in new_contents['categories']:
        print(c)
    # Add score item to each existing annotation
    for i in range(len(new_contents['annotations'])):
        new_contents['annotations'][i]['score'] = 1.0


    # Create category_id2name_map
    category_id2name_map = {}
    category_name2id_map = {}
    for i in range(len(new_contents['categories'])):
        category_id2name_map[new_contents['categories'][i]['id']] = new_contents['categories'][i]['name']
        category_name2id_map[new_contents['categories'][i]['name']] = new_contents['categories'][i]['id']

    # Create id_class sets
    id_classname_set = class_names_dict[args.split]
    id_classid_set = [category_name2id_map[name] for name in id_classname_set]
    print("id_classname_set:", id_classname_set)
    print("id_classid_set:", id_classid_set)

    # Compute original_id_ann_count and max_allowed_new_preds
    print("\n\nsource_contents['images']:", len(source_contents['images']))
    original_id_ann_count = subset_ann_count(source_contents, id_classname_set)
    print("Total original ID anns:", original_id_ann_count)
    max_allowed_new_preds = round(original_id_ann_count * args.percent_new)
    print("max allowed new preds:", max_allowed_new_preds)


    """
    Add new annotations
    """
    # Create source_img2idann_map
    source_img2idann_map = {}
    for ann in source_contents['annotations']:
        # Check if current ann is ID object
        if ann['category_id'] in id_classid_set:
            curr_image_id = ann['image_id']
            if curr_image_id in source_img2idann_map:
                source_img2idann_map[curr_image_id].append(ann)
            else:
                source_img2idann_map[curr_image_id] = [ann]

        
    if args.oracle_anns:
        all_img2allann_map = {}
        for ann in oracle_anns_contents['annotations']:
            curr_image_id = ann['image_id']
            if curr_image_id in all_img2allann_map:
                all_img2allann_map[curr_image_id].append(ann)
            else:
                all_img2allann_map[curr_image_id] = [ann]

        
    #for k, v in source_img2idann_map.items():
    #    print("\n", k)
    #    for i in range(len(v)):
    #        print(v[i])

    # If we have an args.restricted argument, filter preds from images
    # that don't contain an ID instance
    if args.restricted:
        # Filter preds_contents
        print("\nFiltering restricted preds_contents...")
        restricted_preds_contents = []
        for i in range(len(preds_contents)):
            if preds_contents[i]['image_id'] in source_img2idann_map: 
                restricted_preds_contents.append(preds_contents[i])

        print("preds_contents (before):", len(preds_contents))
        preds_contents = restricted_preds_contents
        print("preds_contents (after):", len(preds_contents))


    # Initialize candidate PLs
    candidate_pseudolabels = []
    
    # Iterate over preds anns
    curr_id = 1000000000
    for i, pred in enumerate(preds_contents):
        # Filter out utter garbage (< 0.7). This isn't necessary, but it speeds up runtime
        #if pred['score'] >= 0.70:
        # Check if pred has significant overlap with an existing annotation
        if not gt_overlap_exists(pred, source_img2idann_map):
            # Make copy
            pseudo_label = copy.deepcopy(pred)
            # Round bbox ann to ints
            #pseudo_label['bbox'] = [round(x) for x in pred['bbox']]
            # Add annotation id
            pseudo_label['id'] = curr_id
            curr_id += 1
            # Change category
            pseudo_label['category_id'] = 999999999
            # Add 'segmentation'
            pseudo_label['segmentation'] = [xywh2xyxy(pseudo_label['bbox'])]
            # Add 'area'
            pseudo_label['area'] = float(pseudo_label['bbox'][2] * pseudo_label['bbox'][3])
            # Add iscrowd
            pseudo_label['iscrowd'] = 0
            #print("\npseudo_label:")
            #for k, v in pseudo_label.items():
            #    print(k, v)
            #exit()
            # Append pseudolabel to candidate_pls
            candidate_pseudolabels.append(pseudo_label)
            #new_contents['annotations'].append(pseudo_label)
        if i!=0 and i%10000==0:
            print(f"Completed {i}/{len(preds_contents)} candidate pseudolabels")

    # Sort candidates by descending confidence score
    candidate_pseudolabels.sort(key=lambda x: x['score'], reverse=True)
   
    print("len(candidate_pseudolabels):", len(candidate_pseudolabels))
    # Take the top sorted candidate pseudolabels
    if len(candidate_pseudolabels) <= max_allowed_new_preds:
        #new_contents['annotations'].extend(candidate_pseudolabels)
        top_candidate_pseudolabels = candidate_pseudolabels
    else:
        #new_contents['annotations'].extend(candidate_pseudolabels[:max_allowed_new_preds])
        top_candidate_pseudolabels = candidate_pseudolabels[:max_allowed_new_preds]

    


    # If we are passed oracle filtering options, filter out R% of the bad pseudo-labels
    if args.oracle_anns:
        print("\n\nPerforming oracle filtering...")
        print("Before filtering:", len(top_candidate_pseudolabels))
        filtered_top_candidate_pseudolabels = []
        for i, pl in enumerate(top_candidate_pseudolabels):
            # If this pl overlaps an object with a hidden label keep it, otherwise remove it 
            # with probability args.oracle_filter_percent
            # ** Note that allann == hiddenann because we've already removed overlaps with
            #    labeled ID instances
            if gt_overlap_exists(pl, all_img2allann_map):
                filtered_top_candidate_pseudolabels.append(pl)
            else:
                if random.random() >= args.oracle_filter_percent:
                    filtered_top_candidate_pseudolabels.append(pl)
        top_candidate_pseudolabels = filtered_top_candidate_pseudolabels
        print("After filtering:", len(top_candidate_pseudolabels))
                


    # Add to new_contents
    new_contents['annotations'].extend(top_candidate_pseudolabels)

    # Summarize
    print("\n\ntotal source anns:", len(source_contents['annotations']))
    print("total new anns", len(new_contents['annotations']))
    diff = len(new_contents['annotations'])-len(source_contents['annotations'])
    print("diff:", diff)
    print("total new ID anns:", original_id_ann_count + diff)

    """
    Write new_contents to file
    """
    print("\nWriting outfile:", new_filepath)
    with open(new_filepath, 'w') as out:
        json.dump(new_contents, out)
    print("Finished writing outfile!")

            
    



if __name__ == '__main__':
    main()
