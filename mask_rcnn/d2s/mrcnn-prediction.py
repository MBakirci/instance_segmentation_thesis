import cv2
import os
import sys
import argparse

# Import Mask RCNN
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from d2s import d2sDataset
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize

# load the class label names from disk, one label per line
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
args = parser.parse_args()

dataset_val = d2sDataset()
dataset_val.load_d2s(args.dataset, subset="val_set")
dataset_val.prepare()
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = dataset_val.class_names
print(CLASS_NAMES)

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
# Download the mask_rcnn_coco.h5 file from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
model.load_weights(filepath="../mask_rcnn_d2s_1.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("/home/mehmet/Documents/tilburg-uni-23-24/data/d2s_images_v1/images/D2S_000308.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
