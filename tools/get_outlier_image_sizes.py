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
### Main
##########################################################
def main():
    out_file = os.path.join('data', 'imagenet', 'image_sizes.json')
    imagenet_files = glob.glob('/zero1/data1/ILSVRC2012/train/original/*/*')

    # Remove corrupted image
    imagenet_files.remove('/zero1/data1/ILSVRC2012/train/original/n04357314/n04357314_1467.JPEG')
    print("Total images:", len(imagenet_files))

    # Generate path_to_size dict
    print("Generating path_to_size dict...")
    path_to_size = {}
    for i in range(len(imagenet_files)):
        file_name = imagenet_files[i].split('/')[-1]
        img = Image.open(imagenet_files[i])
        w, h = img.size
        path_to_size[file_name] = {'h': h, 'w': w}
        if i % 1000 == 0:
            print("{} / {}".format(i, len(imagenet_files)))

    
    # Save annotations file to disk
    with open(out_file, 'w') as out:
        json.dump(path_to_size, out)
    print("\nDumped output to:", out_file)

    


if __name__ == '__main__':
    main()
