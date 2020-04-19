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
import datetime
import tensorflow as tf
import keras
from keras.callbacks import LearningRateScheduler

from cytools import mask_tools

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model_felix as modellib


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
    def load_vistas(self, dataset_dir, subset, class_ids=None, config):
        """Load a subset of the Mapillary Vistas dataset.
        dataset_dir: The root directory of the Mapillary Vistas dataset.
        subset: What to load (training, testing, validation) subset should be in the file ~/dataset_dir/subset
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        # Save configuragions
        self.config = config
        
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
        image, _, scale, padding, _ = utils.resize_image(image, min_dim=self.config.IMAGE_MIN_DIM, max_dim=self.config.IMAGE_MAX_DIM, min_scale=None, mode="square")
        self.MASK_SCALE = scale
        self.MASK_PADDING = padding
        return image
    

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

        instance_im = cv2.imread(instance_mask_path, cv2.IMREAD_UNCHANGED)
        instance_im = resize_mask(instance_im, self.MASK_SCALE, self.MASK_PADDING):

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

        #instance_masks = []

        #for i in range(len(instance_ids)):
        #    instance_masks.append(np.where(instance_im == instance_ids[i], True, False))

        # Pack instance masks into an array
        if class_ids:
            class_ids = np.array(class_ids, dtype=np.int32)
            # tic = time.perf_counter()
            #instance_mask = np.stack(instance_masks, axis=2)
            instance_mask = np.zeros([instance_im.shape[0], instance_im.shape[1], len(instance_ids)], dtype=bool)
            instance_sizes = np.zeros([len(instance_ids)], dtype=np.int32)
            mask_tools.get_binary_masks(instance_mask, instance_sizes, np.array(instance_ids, dtype=np.uint16), instance_im)
            shape_before = instance_mask.shape[2]
            small_masks = np.where(instance_sizes < 80)
            instance_mask = np.delete(instance_mask, small_masks, axis=2)
            class_ids = np.delete(class_ids, small_masks)
            
            #if shape_before - instance_mask.shape[2] > 0:
                #print("")
                #print("Removed {}/{} masks because they are too small.".format(shape_before - instance_mask.shape[2], shape_before))

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
            STEPS_PER_EPOCH = 9000
            VALIDATION_STEPS = 1000
            IMAGE_MAX_DIM = 1024
            IMAGE_MIN_DIM = 1024
            LEARNING_RATE = 0.001
            USE_MINI_MASK = False
            #MINI_MASK_SHAPE = (64, 64)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = TrainConfig()
    else:
        class InferenceConfig(mapvistas):
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
        # Path to trained weights file
        COCO_MODEL_PATH = os.path.join("/cluster/scratch/erbachj/", "mask_rcnn_coco.h5")
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
        model_path = COCO_MODEL_PATH
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True,
                          exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                   "mrcnn_bbox", "mrcnn_mask"])
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True,
                          exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                   "mrcnn_bbox", "mrcnn_mask"])
    else:
        model_path = args.model
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

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
        augmentation = imgaug.augmenters.Sometimes(
            0.1,
            [imgaug.augmenters.Affine(rotate=(-4, 4)),
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Affine(scale=(1, 1.3)),
            imgaug.augmenters.MultiplyBrightness((0.9, 1.1)),
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 0.3))]
        )

        # Add custom tensorboard callback    
        logdir = os.path.join(
            "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(args.logs, update_freq = 1000)

        # Learning Rate Scheduler
        def step_decay_1(epoch):
            initial_lrate = config.LEARNING_RATE
            drop = 0.5
            epochs_drop = 20.0
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        lrate = LearningRateScheduler(step_decay_1)
        #Learning Rate Scheduler for training stage 3
        def step_decay_2(epoch):
            initial_lrate = config.LEARNING_RATE/5
            drop = 0.4
            epochs_drop = 10.0
            lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
            return lrate
        lrate2 = LearningRateScheduler(step_decay_2)
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=29,
                    layers='heads', 
                    augmentation=augmentation,
                    custom_callbacks = [tensorboard_callback, lrate])

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=109,
                    layers='4+',
                    augmentation=augmentation, 
                    custom_callbacks = [tensorboard_callback, lrate])

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=149,
                    layers='all',
                    augmentation=augmentation, 
                    custom_callbacks = [tensorboard_callback, lrate2])
