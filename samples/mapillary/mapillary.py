"""
Mask R-CNN
Configurations and data loading code for the Mapillary Vistas Dataset.
"""

import os
import glob
import sys
import math
import random
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import json
import time
import imgaug


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class mapvistas(Config):
    """Configuration for training on the Mapillary Vistas dataset.
    Derives from the base Config class and overrides values specific
    to the Mapillary Vistas dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mapvistas"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 65  # background + 65 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640


class MapillaryDataset(utils.Dataset):
    def load_vistas(self, dataset_dir, subset, class_ids=None):
        """Load a subset of the Mapillary Vistas dataset.
        dataset_dir: The root directory of the Mapillary Vistas dataset.
        subset: What to load (training, testing, validation) subset should be in the file ~/dataset_dir/subset
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        # Read config.json file 
        with open("{}/config.json".format(dataset_dir)) as json_file:
          config = json.load(json_file)

        dataset_dir = "{}/{}".format(dataset_dir, subset)
        self.dataset_dir = dataset_dir

        color_to_classid = {(0, 0, 0): 0}
        classid_to_name = ['BG']

        if not class_ids:
            # All classes
            class_ids = list(range(1, len(classid_to_name)))

        for i in range(len(config['labels'])-1):
          if i+1 in class_ids:
              classid_to_name.append(config['labels'][i]['readable'])
              color_ = tuple(config['labels'][i]['color'])
              color_to_classid.update({color_: class_ids.index(i+1)})
          else:
              color_ = tuple(config['labels'][i]['color'])
              color_to_classid.update({color_: 0})

        self.color_to_classid = color_to_classid
        self.classid_to_name = classid_to_name
        
        # Add classes
        for i in range(1, len(classid_to_name)):
            self.add_class("mapillary_vistas", i, classid_to_name[i])

        # Add class_ids in an easy accessible way
        self.class_number = class_ids

        # Iterate trough all images in subset path
        image_paths = glob.iglob(os.path.join(dataset_dir,'images', '*.*'))
        for image_path in image_paths:
          head, tail = os.path.split(image_path)

          # Add images
          image_id = tail[0:-4]
          self.add_image(
              "mapillary_vistas", image_id=image_id,
              path=image_path,
              )

            
    def image_reference(self, image_id):
        """Return the path of the image"""
        
        info = self.image_info[image_id]
        if info["source"] == "mapillary_vistas":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
            
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = super(MapillaryDataset, self).load_image(image_id)
        return cv2.resize(image, (768, 768), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Time if necessary.
        # tic = time.perf_counter()

        # If not an InteriorNet image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "mapillary_vistas":
            return super(self._class_, self).load_mask(image_id)
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        instance_mask_path = os.path.join(self.dataset_dir,
                                          'instances', '{}.png'.format(image_info['id']))
        instance_to_class_color_path = os.path.join(self.dataset_dir,
                                                    'instances', '{}..txt'.format(image_info['id']))

        instance_im = imageio.imread(instance_mask_path)
        instance_im = cv2.resize(instance_im, (768, 768), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

        class_ids = []
        instance_ids = []

        with open(instance_to_class_color_path) as f:
            lines = f.readlines()
            for line in lines:
                values = line.split(" ")
                color_ = (int(values[1]), int(values[2]), int(values[3]))
                class_id = self.color_to_classid[color_]
                if class_id !=0:
                    class_ids.append(class_id)
                    instance_ids.append(int(values[0]))

        instance_masks = []

        for i in range(len(instance_ids)):
            instance_masks.append(np.where(instance_im == instance_ids[i], True, False))

        # Pack instance masks into an array
        if class_ids:
            class_ids = np.array(class_ids, dtype=np.int32)
            # tic = time.perf_counter()
            instance_mask = np.stack(instance_masks, axis=2)

            #print(class_ids)
            # toc = time.perf_counter()
            # print("Time to create mask: {}".format(toc-tic))

            return instance_mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MapillaryDataset, self).load_mask(image_id)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Mapillary Vistas.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Mapillary Vistas")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/mapillary_vistas/",
                        help='Directory of the Mapillary Vistas dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=True,
                        default='/logs',
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')                
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Selected classes
    selected_classes = selected_classes = [34, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52]


    # Configurations
    if args.command == "train":
        class TrainConfig(mapvistas):
            NUM_CLASSES = len(selected_classes) + 1
            STEPS_PER_EPOCH = 1500
            VALIDATION_STEPS = 100
            IMAGE_MAX_DIM = 768
            IMAGE_MIN_DIM = 768
            LEARNING_RATE = 0.001
            USE_MINI_MASK = False
            #MINI_MASK_SHAPE = (64, 64)
            IMAGES_PER_GPU = 2
        config = TrainConfig()
    else:
        class mapvistas(mapvistas):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    #model.load_weights(COCO_MODEL_PATH, by_name=True,
    #                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
    #                            "mrcnn_bbox", "mrcnn_mask"])
    model.load_weights("/tf/logs/mapvistas20200403T0216/mask_rcnn_mapvistas_0009.h5")

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = MapillaryDataset()
        dataset_train.load_vistas(args.dataset, "training", class_ids=selected_classes)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = MapillaryDataset()
        dataset_val.load_vistas(args.dataset, subset='validation', class_ids=selected_classes)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        #augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)