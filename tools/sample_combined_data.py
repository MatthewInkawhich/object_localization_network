# MatthewInkawhich

import argparse
import os
import glob
import shutil
import warnings
import numpy as np
import random
import json
import copy
import datetime
from PIL import Image



##########################################################
### Argument parsing
##########################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample new subset of auxiliary dataset')
    parser.add_argument('samples', type=int, help='number of auxiliary samples to include')
    parser.add_argument('--seed', type=int, default=-1, help='random seed to use')
    args = parser.parse_args()
    return args




##########################################################
### Main
##########################################################
def main():
    # Parse args
    args = parse_args()

    # File/Directory names
    new_data_dir = os.path.join('data', 'coco_imagenet')
    new_annotations_file = os.path.join(new_data_dir, 'annotations.json')
    new_image_dir = os.path.join(new_data_dir, 'images')

    ### Reset data directory
    if os.path.exists(new_annotations_file):
        print("Removing", new_annotations_file)
        os.remove(new_annotations_file)
    if os.path.exists(new_image_dir):
        print("Clearing", new_image_dir)
        image_files = glob.glob(os.path.join(new_image_dir, '*'))
        for f in image_files:
            os.remove(f)
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
        
    ### Sample auxiliary samples and merge them with coco
    # COCO paths
    print("\nGlobbing coco image filepaths...")
    coco_image_filepaths = glob.glob('/raid/inkawhmj/WORK/data/coco/images2017/train2017/*')
    coco_annotation_file = 'data/coco/annotations/instances_train2017.json'
    # Auxiliary paths
    print("\nGlobbing auxiliary image filepaths...")
    auxiliary_image_sizes_file = os.path.join(new_data_dir, 'imagenet_image_sizes.json')
    auxiliary_image_filepaths = glob.glob('/zero1/data1/ILSVRC2012/train/original/*/*')
    auxiliary_image_filepaths.remove('/zero1/data1/ILSVRC2012/train/original/n04357314/n04357314_1467.JPEG')
    print("Total (usable) auxiliary images:", len(auxiliary_image_filepaths))

    # Randomly sample args.samples paths
    print("\nRandomly sampling auxiliary paths...")
    if args.seed > 0:
        seed = args.seed
    else:
        seed = random.randint(1, 1000000)
    print("seed:", seed)
    random.seed(seed)
    random.shuffle(auxiliary_image_filepaths) 
    auxiliary_image_subset = auxiliary_image_filepaths[:args.samples]
    print("auxiliary subset size:", len(auxiliary_image_subset))

    # Create symlinks for each image in coco
    print("\nCreating symlinks for COCO data...")
    for i in range(len(coco_image_filepaths)):
        f = coco_image_filepaths[i]
        symlink_target = os.path.join(new_image_dir, f.split('/')[-1])
        os.symlink(f, symlink_target)    
        if i % 1000 == 0:
            print("{} / {}".format(i, len(coco_image_filepaths)))

    # Create symlinks for each image in the auxiliary_image_subset
    print("\nCreating symlinks for auxiliary data...")
    new_auxiliary_image_paths = []
    for i in range(len(auxiliary_image_subset)):
        f = auxiliary_image_subset[i]
        symlink_target = os.path.join(new_image_dir, 'aux_' + f.split('/')[-1])
        new_auxiliary_image_paths.append(symlink_target)
        #print(f, symlink_target)
        os.symlink(f, symlink_target)    
        if i % 1000 == 0:
            print("{} / {}".format(i, len(auxiliary_image_subset)))

    ### Initialize annotations file
    # Load coco annotations (to use as template)
    print("\nOpening coco annotations file...")
    with open(coco_annotation_file, 'r') as f:
        coco_contents = json.load(f)
    new_contents = copy.deepcopy(coco_contents)
    # Modify contents for imagenet
    new_contents['categories'].append({
        'supercategory': 'UNKNOWN',
        'id': 999999999,
        'name': 'UNKNOWN',
    })
    # Add seed to info
    new_contents['info']['seed'] = seed
    # Add auxiliary images to 'images' list
    print("\nAdding auxiliary images to 'images' list...")
    with open(auxiliary_image_sizes_file, 'r') as f:
        image_sizes = json.load(f)
    curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for i in range(len(new_auxiliary_image_paths)):
        file_name = new_auxiliary_image_paths[i].split('/')[-1]
        #img = Image.open(new_auxiliary_image_paths[i])
        #w, h = img.size
        curr_size = image_sizes[file_name.split('aux_')[-1]]
        new_contents['images'].append(
            {
                "id": 1000000000+i,
                "width": curr_size['w'],
                "height": curr_size['h'],
                "file_name": file_name,
                "license": 2,
                "flickr_url": "n/a",
                "coco_url": "n/a",
                "date_captured": curr_datetime,
            }   
        )
        if i % 1000 == 0:
            print("{} / {}".format(i, len(new_auxiliary_image_paths)))

    # Save annotations file to disk
    print("\nWritingn new annotation file...")
    with open(new_annotations_file, 'w') as out:
        json.dump(new_contents, out)
    print("\nWrote new updated annotations to:", new_annotations_file)
    


if __name__ == '__main__':
    main()
