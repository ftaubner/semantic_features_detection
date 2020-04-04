
#HOME_DIR = '/cluster/home/erbachj'
#LOCAL_SCRATCH = "/scratch"  
#SCRATCH = '/cluster/scratch/erbachj'
# Install required packages
pip install --user -r /cluster/home/erbachj/semantic_features/semantic_features_detection/requirements.txt

cd /cluster/home/erbachj/semantic_features/semantic_features_detection
python setup.py install --prefix=$HOME/python

# Get dataset and unzip it
wget --no-check-certificate "https://onedrive.live.com/download?cid=EA356294C6263A37&resid=EA356294C6263A37%21100024&authkey=ADLJzehr2ENRggw" -O $TMPDIR/mapillary_vistas.zip
unzip -qq $TMPDIR/mapillary_vistas.zip -d $TMPDIR

# run mapillary.py script
python3 samples/mapillary/mapillary_leo.py train --dataset=$TMPDIR/mapillary_vistas --model=coco --logs=/cluster/scratch/erbachj/semantic_features/logs