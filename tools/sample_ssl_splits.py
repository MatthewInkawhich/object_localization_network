# MatthewInkawhich

import argparse
import os
import shutil
import warnings
import random
import json
import copy

"""
This script randomly subsamples a certain percentage of each class 
instance annotations for use in an SSL-style scenario.

The idea is to train models on reduced annotation sets (while 
keeping the number of images the same).

The output of this file is a new annotation file in the 
"new_annotation_dir". 
"""


##########################################################
### Argument parsing
##########################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample subset of coco annotations')
    parser.add_argument('percentage', type=float, help='percentage of original labels to include')
    parser.add_argument('seed', type=int, default=1, help='random seed to use')
    args = parser.parse_args()
    return args




##########################################################
### Main
##########################################################
def main():
    # Parse args
    args = parse_args()

    assert args.percentage > 0 and args.percentage < 1.0, "args.percentage NOT in range (0,1)"
    str_percentage = "{:.2f}".format(args.percentage).split('.')[-1]

    # Set random seed
    random.seed(args.seed)

    # File/Directory names
    coco_annotation_file = 'data/coco/annotations/instances_train2017.json'
    new_annotation_dir = os.path.join('data', 'coco', 'ssl_annotations')
    new_annotation_file = os.path.join(new_annotation_dir, 'ssl_annotations_p{}_s{}.json'.format(str_percentage, args.seed))

    print("new_annotation_dir:", new_annotation_dir)
    print("new_annotation_file:", new_annotation_file)

    # Create data directory if necessary
    if not os.path.exists(new_annotation_dir):
        print("Creating", new_annotation_dir)
        os.makedirs(new_annotation_dir)
        
    # Load coco annotations
    print("\nOpening coco annotations file...")
    with open(coco_annotation_file, 'r') as f:
        coco_contents = json.load(f)

    # Collect category ids
    category_ids = []
    print("\ncategories:")
    for i in range(len(coco_contents['categories'])):
        category_ids.append(coco_contents['categories'][i]['id'])
        print("\n")
        for k, v in coco_contents['categories'][i].items():
            print(k, "\t", v)
    print("\ncategory_ids:", category_ids, len(category_ids))

    # Create category id to annotation idx map
    # Organize annotation indexes (NOT IDs) by class
    print("\n\nCreating categoryid_to_annotationidx_map...")
    categoryid_to_annotationidx_map = {}
    for cat_id in category_ids:
        categoryid_to_annotationidx_map[cat_id] = []

    print("\nFilling categoryid_to_annotationidx_map...")
    for i in range(len(coco_contents['annotations'])):
        cat_id = coco_contents['annotations'][i]['category_id']
        categoryid_to_annotationidx_map[cat_id].append(i)

    # Print results
    print("\nCompleted categoryid_to_annotationidx_map:")
    counted_annotations = 0
    for k, v in categoryid_to_annotationidx_map.items():
        print(k, "\t", len(v))
        counted_annotations += len(v)
    print("\ntrue num annotation:", len(coco_contents['annotations']))
    print("counted_annotations:", counted_annotations)


    # Sample args.percentage of each class
    print("\nSampling percentage of each class...")
    all_sampled_annotation_idxs = []
    for k, v in categoryid_to_annotationidx_map.items():
        num_to_sample = round(len(v) * args.percentage)
        all_sampled_annotation_idxs.extend(random.sample(v, num_to_sample))
        print(k, "\t", num_to_sample)
    print("\ntrue total*percentage:", round(len(coco_contents['annotations']) * args.percentage))
    print("num sampled annotation idxs:", len(all_sampled_annotation_idxs))

    # Sort list (to maintain original order) (I don't think this is necessary)
    all_sampled_annotation_idxs.sort()

    # Generate new annotations
    new_annotations = []
    for original_ann_idx in all_sampled_annotation_idxs:
        new_annotations.append(coco_contents['annotations'][original_ann_idx])
    print("new_annotations:", len(new_annotations))



    ### Prepare new contents
    # Copy coco_contents to new dict
    print("\nCopying over coco_contents to new_contents")
    new_contents = copy.deepcopy(coco_contents)
    # Add seed to info
    new_contents['info']['seed'] = args.seed
    # Clear annotations
    new_contents['annotations'] = new_annotations

    
    ### Save annotations file to disk
    print("\nWriting new annotation file...")
    with open(new_annotation_file, 'w') as out:
        json.dump(new_contents, out)
    print("\nWrote new updated annotations to:", new_annotation_file)
    


if __name__ == '__main__':
    main()
