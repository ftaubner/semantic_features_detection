#!usr/bin/env python3

import os

HOME_DIR = '/cluster/home/erbachj'
LOCAL_SCRATCH = "/scratch"  
SCRATCH = '/cluster/scratch/erbachj'
# Install required packages
!pip install -r /cluster/home/erbachj/requirements.txt

os.chdir('/cluster/home/erbachj/semantic_features/semantic_features_detection')
!python setup.py install

# Get dataset and unzip it
!wget --no-check-certificate "https://onedrive.live.com/download?cid=EA356294C6263A37&resid=EA356294C6263A37%2199865&authkey=ALxjJOloi9Wjk80" -O /scratch/mapillary_vistas_sample.zip
!unzip -qq /scratch/mapillary_vistas_sample.zip -d /scratch

# run mapillary.py script
!python3 samples/mapillary/mapillary.py evaluate --dataset=/scratch/mapillary_vistas/ --model=coco --logs=/cluster/scratch/erbachj/semantic_features/logs