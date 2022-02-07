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
### Reset imagenet sample selection
##########################################################
def reset_imagenet_samples(args):
    data_dir = os.path.join('data', 'imagenet')
    annotations_file = os.path.join(data_dir, 'empty_annotations.json')
    image_sizes_file = os.path.join(data_dir, 'image_sizes.json')
    image_dir = os.path.join(data_dir, 'images')
    ### Reset data/imagenet directory
    if os.path.exists(annotations_file):
        print("Removing", annotations_file)
        os.remove(annotations_file)
    if os.path.exists(image_dir):
        print("Clearing", image_dir)
        image_files = glob.glob(os.path.join(image_dir, '*'))
        for f in image_files:
            os.remove(f)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    ### Sample imagenet images and link them
    imagenet_files = glob.glob('/zero1/data1/ILSVRC2012/train/original/*/*')
    # Remove corrupted image
    imagenet_files.remove('/zero1/data1/ILSVRC2012/train/original/n04357314/n04357314_1467.JPEG')
    print("Total (usable) imagenet images:", len(imagenet_files))

    # Randomly sample args.samples paths
    if args.seed > 0:
        seed = args.seed
    else:
        seed = random.randint(1, 10000)
    print("seed:", seed)
    random.seed(seed)
    random.shuffle(imagenet_files) 
    imagenet_subset = imagenet_files[:args.samples]
    print("subset size:", len(imagenet_subset))

    # Create symlinks for each image in the subset
    print("\nCreating symlinks...")
    new_image_paths = []
    for i in range(len(imagenet_subset)):
        f = imagenet_subset[i]
        symlink_target = os.path.join(image_dir, f.split('/')[-1])
        new_image_paths.append(symlink_target)
        #print(f, symlink_target)
        os.symlink(f, symlink_target)    
        if i % 1000 == 0:
            print("{} / {}".format(i, len(imagenet_subset)))

    ### Initialize fresh annotations file
    # Load coco annotations (to use as template)
    with open('data/coco/annotations/instances_val2017.json', 'r') as f:
        coco_contents = json.load(f)
    new_contents = copy.deepcopy(coco_contents)
    # Modify contents for imagenet
    new_contents['categories'].append({
        'supercategory': 'UNKNOWN',
        'id': 999999999,
        'name': 'UNKNOWN',
    })
    new_contents['annotations'] = []
    # Add seed to info
    new_contents['info']['seed'] = seed
    # Fill images list
    print("\nFilling images dict...")
    new_contents['images'] = []
    with open('data/imagenet/image_sizes.json', 'r') as f:
        image_sizes = json.load(f)
    curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for i in range(len(new_image_paths)):
        file_name = new_image_paths[i].split('/')[-1]
        #img = Image.open(new_image_paths[i])
        #w, h = img.size
        curr_size = image_sizes[file_name]
        new_contents['images'].append(
            {
                "id": i,
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
            print("{} / {}".format(i, len(new_image_paths)))
        #print("\n")
        #for k, v in new_contents['images'][i].items():
        #    print(k, v)
    
    # Save annotations file to disk
    with open(annotations_file, 'w') as out:
        json.dump(new_contents, out)
    print("\nWrote new (empty) annotations to:", annotations_file)




##########################################################
### Main
##########################################################
def main():
    args = parse_args()
    reset_imagenet_samples(args)

    


if __name__ == '__main__':
    main()
