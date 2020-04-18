import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imageio
import tensorflow as tf
import glob
import wget

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/mapillary/"))  # To find local version
import mapillary

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class InferenceConfig(mapillary.mapvistas):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    NUM_CLASSES = 17 + 1

class Inference():
    def __init__(self, weight_dir):
        self.config = InferenceConfig()
        # Create model in training mode
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                  model_dir=MODEL_DIR)
        # if weights are not there download epoch 47 from onedrive                          
        if not os.path.exists(weight_dir):
            url = "https://onedrive.live.com/download?cid=EA356294C6263A37&resid=EA356294C6263A37%21100479&authkey=AJdjSPnfz5shCvk"
            wget.download(url, out=weight_dir)
        self.model.load_weights(weight_dir, by_name=True)
    def predict(self, image_path, save_vis=False, save_dir=None):
        """ Returns segentation results as dict and binary masks
        # Inputs:
            image_path: path to a single image
            save_vis: Boolean if segmentation results are to be saved
            save_dir: directory where results are to be saved (name stays the same)
        # Return:
            dict with results in format {'image_id': image_id, 'classes': class_ids, 'boxes': bboxes}
            binary masks in format [im_h, im_w, num_instances] with the same ordering as in the dict
        """
        print(image_path)
        image = cv2.imread(image_path)
        results = self.model.detect([image], verbose=0)
        r = results[0]
        image_id=os.path.split(image_path)[1][0:-4]
        if save_vis:
            visualize.save_image(image = image[:,:,::-1], image_name=image_id, boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'], class_names=class_names[1:], scores=r['scores'], save_dir=save_dir)
        features = {'image_id': image_id, 'classes': r['class_ids'].tolist(), 'boxes': r['rois'].tolist()}
        return features, r['masks']




