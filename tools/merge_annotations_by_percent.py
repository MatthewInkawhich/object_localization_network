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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pseudolabels')
    parser.add_argument('source', help='source annotation path')
    parser.add_argument('preds', help='preds annotation path')
    parser.add_argument('original_id_anns', type=int, help='number of original ID annotations')
    parser.add_argument('percent_new', type=float, help='percent increase from new pseudo-labels relative to original_id_anns')
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
                return True
    return False
            



def main():
    # Parse args
    args = parse_args()

    next_round = int(args.preds.split('/')[-1].split('_round')[-1].split('.bbox.json')[0]) + 1
    str_percent_new = "{:.2f}".format(args.percent_new).split('.')[-1]
    if "robustpreds" in args.preds.split('/')[-1]:
        jitter = args.preds.split('/')[-1].split('robustpreds')[-1].split('_round')[0]
        new_filepath = os.path.join(args.preds.split('/robustpreds')[0], f'robust{jitter}_annotations_for_round{next_round}_p{str_percent_new}.json')
    else:
        new_filepath = os.path.join(args.preds.split('/preds_round')[0], f'annotations_for_round{next_round}_p{str_percent_new}.json')
    print("new_filepath:", new_filepath)

    max_allowed_new_preds = round(args.original_id_anns * args.percent_new)
    print("max allowed new preds:", max_allowed_new_preds)

    # Read source and preds files
    with open(args.source, 'r') as f:
        source_contents = json.load(f)
    with open(args.preds, 'r') as f:
        preds_contents = json.load(f)

    # Initialize new_contents
    new_contents = copy.deepcopy(source_contents)
    
    """
    Add new UNKNOWN category
    """
    new_contents['categories'].append({
        'supercategory': 'UNKNOWN',
        'id': 999999999,
        'name': 'UNKNOWN',
    })

    for c in new_contents['categories']:
        print(c)

    """
    Add new annotations
    """
    # Create source source_img2ann_map
    source_img2ann_map = {}
    for ann in source_contents['annotations']:
        curr_image_id = ann['image_id']
        if curr_image_id in source_img2ann_map:
            source_img2ann_map[curr_image_id].append(ann)
        else:
            source_img2ann_map[curr_image_id] = [ann]

    #for k, v in source_img2ann_map.items():
    #    print("\n", k)
    #    for i in range(len(v)):
    #        print(v[i])

    # Initialize candidate PLs
    candidate_pseudolabels = []
    
    # Iterate over preds anns
    curr_id = 1000000000
    for i, pred in enumerate(preds_contents):
        # Filter out utter garbage (< 0.7). This isn't necessary, but it speeds up runtime
        #if pred['score'] >= 0.70:
        # Check if pred has significant overlap with an existing annotation
        if not gt_overlap_exists(pred, source_img2ann_map):
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
            #print("\npseudo_label:", pseudo_label)
            # Append pseudolabel to candidate_pls
            candidate_pseudolabels.append(pseudo_label)
            #new_contents['annotations'].append(pseudo_label)
        if i!=0 and i%1000==0:
            print(f"Completed {i}/{len(preds_contents)} candidate pseudolabels")

    # Sort candidates by descending confidence score
    candidate_pseudolabels.sort(key=lambda x: x['score'], reverse=True)
   
    print("len(candidate_pseudolabels):", len(candidate_pseudolabels))
    # Add the top sorted candidate pseudolabels to new_contents['annotations']
    if len(candidate_pseudolabels) <= max_allowed_new_preds:
        new_contents['annotations'].extend(candidate_pseudolabels)
    else:
        new_contents['annotations'].extend(candidate_pseudolabels[:max_allowed_new_preds])


    print("total source anns:", len(source_contents['annotations']))
    print("total new anns", len(new_contents['annotations']))
    diff = len(new_contents['annotations'])-len(source_contents['annotations'])
    print("diff:", diff)
    print("total new ID anns:", args.original_id_anns + diff)

    """
    Write new_contents to file
    """
    print("\nWriting outfile:", new_filepath)
    with open(new_filepath, 'w') as out:
        json.dump(new_contents, out)
    print("Finished writing outfile!")

            
    





if __name__ == '__main__':
    main()
