import os
import sys
import math
import random
import numpy as np
import cv2
import glob
import scenenet_pb2 as sn


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils



WNID_PATH = 'pySceneNetRGBD/semantic_classes/wnid_to_class.txt'


class ScenenetConfig(Config):
    """Configuration for training on the SceneNet dataset.
    Derives from the base Config class and overrides values specific
    to the SceneNet dataset.
    """
    # Give the configuration a recognizable name
    NAME = "scenenet"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # background + 13 

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 240
    IMAGE_MAX_DIM = 320

class ScenenetDataset(utils.Dataset):
    def load_scene(self, dataset_dir, subset, class_ids=None, return_coco=False):
        """Load a subset of the SceneNet dataset.
        dataset_dir: The root directory of the SceneNet dataset.
        subset: What to load (train, test, val)
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        self.dataset_dir = os.path.join(dataset_dir, subset)
        
        # Add classes
        # Add additional class lists here and select which one to use in 
        # the main method below.  The class list order is important, the 
        # script below gives 'Unknown' an id of 0, 'Bed' and id of 1 etc.
        NYU_13_CLASSES = ['Unknown', 'Bed', 'Books', 'Ceiling', 'Chair',
                  'Floor', 'Furniture', 'Objects', 'Picture',
                  'Sofa', 'Table', 'TV', 'Wall', 'Window'
                 ]
        
        for i in len(NYU_13_CLASSES):
            self.add_class("scenenet", i, NYU_13_CLASSES[i])
        """
        # Could change to "name" here (all classes)
        column_name = '13_classes'
        # This is the list of classes, with the index in the list denoting the # class_id
        class_list = NYU_13_CLASSES

        wnid_to_classid = {}
        with open(WNID_PATH,'r') as f:
            class_lines = f.readlines()
            column_headings = class_lines[0].split()
            for class_line in class_lines[1:]:
                wnid = class_line.split()[0].zfill(8)
                classid = class_list.index(class_line.split()[column_headings.index(column_name)])
                wnid_to_classid[wnid] = classid
                self.add_class("scenenet", classid, class_list[classid])
                
        print(wnid_to_classid)
        self.add_class(wnid_to_classid=wnid_to_classid)
        """
        
        NYU_WNID_TO_CLASS = {
            '04593077':4, '03262932':4, '02933112':6, '03207941':7, '03063968':10, '04398044':7, '04515003':7,
            '00017222':7, '02964075':10, '03246933':10, '03904060':10, '03018349':6, '03786621':4, '04225987':7,
            '04284002':7, '03211117':11, '02920259':1, '03782190':11, '03761084':7, '03710193':7, '03367059':7,
            '02747177':7, '03063599':7, '04599124':7, '20000036':10, '03085219':7, '04255586':7, '03165096':1,
            '03938244':1, '14845743':7, '03609235':7, '03238586':10, '03797390':7, '04152829':11, '04553920':7,
            '04608329':10, '20000016':4, '02883344':7, '04590933':4, '04466871':7, '03168217':4, '03490884':7,
            '04569063':7, '03071021':7, '03221720':12, '03309808':7, '04380533':7, '02839910':7, '03179701':10,
            '02823510':7, '03376595':4, '03891251':4, '03438257':7, '02686379':7, '03488438':7, '04118021':5,
            '03513137':7, '04315948':7, '03092883':10, '15101854':6, '03982430':10, '02920083':1, '02990373':3,
            '03346455':12, '03452594':7, '03612814':7, '06415419':7, '03025755':7, '02777927':12, '04546855':12,
            '20000040':10, '20000041':10, '04533802':7, '04459362':7, '04177755':9, '03206908':7, '20000021':4,
            '03624134':7, '04186051':7, '04152593':11, '03643737':7, '02676566':7, '02789487':6, '03237340':6,
            '04502670':7, '04208936':7, '20000024':4, '04401088':7, '04372370':12, '20000025':4, '03956922':7,
            '04379243':10, '04447028':7, '03147509':7, '03640988':7, '03916031':7, '03906997':7, '04190052':6,
            '02828884':4, '03962852':1, '03665366':7, '02881193':7, '03920867':4, '03773035':12, '03046257':12,
            '04516116':7, '00266645':7, '03665924':7, '03261776':7, '03991062':7, '03908831':7, '03759954':7,
            '04164868':7, '04004475':7, '03642806':7, '04589593':13, '04522168':7, '04446276':7, '08647616':4,
            '02808440':7, '08266235':10, '03467517':7, '04256520':9, '04337974':7, '03990474':7, '03116530':6,
            '03649674':4, '04349401':7, '01091234':7, '15075141':7, '20000028':9, '02960903':7, '04254009':7,
            '20000018':4, '20000020':4, '03676759':11, '20000022':4, '20000023':4, '02946921':7, '03957315':7,
            '20000026':4, '20000027':4, '04381587':10, '04101232':7, '03691459':7, '03273913':7, '02843684':7,
            '04183516':7, '04587648':13, '02815950':3, '03653583':6, '03525454':7, '03405725':6, '03636248':7,
            '03211616':11, '04177820':4, '04099969':4, '03928116':7, '04586225':7, '02738535':4, '20000039':10,
            '20000038':10, '04476259':7, '04009801':11, '03909406':12, '03002711':7, '03085602':11, '03233905':6,
            '20000037':10, '02801938':7, '03899768':7, '04343346':7, '03603722':7, '03593526':7, '02954340':7,
            '02694662':7, '04209613':7, '02951358':7, '03115762':9, '04038727':6, '03005285':7, '04559451':7,
            '03775636':7, '03620967':10, '02773838':7, '20000008':6, '04526964':7, '06508816':7, '20000009':6,
            '03379051':7, '04062428':7, '04074963':7, '04047401':7, '03881893':13, '03959485':7, '03391301':7,
            '03151077':12, '04590263':13, '20000006':1, '03148324':6, '20000004':1, '04453156':7, '02840245':2,
            '04591713':7, '03050864':7, '03727837':5, '06277280':11, '03365592':5, '03876519':8, '03179910':7,
            '06709442':7, '03482252':7, '04223580':7, '02880940':7, '04554684':7, '20000030':9, '03085013':7,
            '03169390':7, '04192858':7, '20000029':9, '04331277':4, '03452741':7, '03485997':7, '20000007':1,
            '02942699':7, '03231368':10, '03337140':7, '03001627':4, '20000011':6, '20000010':6, '20000013':6,
            '04603729':10, '20000015':4, '04548280':12, '06410904':2, '04398951':10, '03693474':9, '04330267':7,
            '03015149':9, '04460038':7, '03128519':7, '04306847':7, '03677231':7, '02871439':6, '04550184':6,
            '14974264':7, '04344873':9, '03636649':7, '20000012':6, '02876657':7, '03325088':7, '04253437':7,
            '02992529':7, '03222722':12, '04373704':4, '02851099':13, '04061681':10, '04529681':7,
        }

        
        #potobuf_path = glob.iglob(os.path.join(dataset_dir, subset, subset + '_protobufs')
        protobuf_path = os.path.join(subset + '_protobufs', 'senenet_rgbd_train_0.pb')
        # Add images
        
        trajectories = sn.Trajectories()
        try:
        with open(protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
        except IOError:
            print('Scenenet protobuf data not found at location:{0}'.format(protobuf_path))
            print('Please ensure you have copied the pb file to the data directory')
        
        for traj in trajectories.trajectories:
            for view in traj.views:
                photo_path = photo_path_from_view(traj.render_path,view)
                self.add_image("scenenet", image_id = os.path.join(traj.render_path, view), path=photo_path, protobuf_path=protobuf_path )
                
            instance_class_map = {}
            for instance in traj.instances:
                instance_type = sn.Instance.InstanceType.Name(instance.instance_type)

                if instance.instance_type != sn.Instance.BACKGROUND:
                    instance_class_map[instance.instance_id] = NYU_WNID_TO_CLASS[instance.semantic_wordnet_id]
            self.add_traj(traj_id=traj.render_path, instance_class_map=instance_class_map)
                
                
    def image_reference(self, image_id):
        """Return the path of the image"""
        
        info = self.image_info[image_id]
        if info["source"] == "scenenet":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
        
        
        
            
    def load_mask(self, image_id):
       """ Load instance masks for the given image.
       Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        protobuf_path = image_info['protobuf_path']
        
        
        instance_masks = []
        class_ids = []
        render_path, view = os.path.split(image_id)
        path = self.image_info[image_id]["path"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        
        instance_mask_path = os.path.join(self.dataset_dir, render_path, 'instance' , "{}.png".format(view))
        # Load mapping between instance id and class id
        instance_class_map = self.traj_info[render_path]["instance_class_map"]

        instance_im = imageio.imread(instance_mask_path)
        
        instance_ids = np.unique(instance_im)
        for instance_id in instance_ids:
            binary_mask = np.where(instance_im == instance_id , True, False)
            class_ids.append(instance_class_map[instance_id])
            instance_masks.append(binary_mask)
            
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)
                           
                           
    def add_traj(self, traj_id, instance_class_map, **kwargs):
        traj_info = {
            "id": traj_id,
            "instance_class_map" = instance_class_map
        }
        traj_info.update(kwargs)
        self.traj_info.append(traj_info)                       
                           
                           
    def photo_path_from_view(render_path,view):
        photo_path = os.path.join(render_path,'photo')
        image_path = os.path.join(photo_path,'{0}.jpg'.format(view.frame_num))
        return os.path.join(dataset_dir,image_path)

    def instance_path_from_view(render_path,view):
        photo_path = os.path.join(render_path,'instance')
        image_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
        return os.path.join(dataset_dir,image_path)

    def depth_path_from_view(render_path,view):
        photo_path = os.path.join(render_path,'depth')
        image_path = os.path.join(photo_path,'{0}.png'.format(view.frame_num))
        return os.path.join(dataset_dir,image_path)

