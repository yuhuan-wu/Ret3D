#!/bin/bash


echo " "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "starting pgnet train script"
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo " "

#CONDA_ENV_PATH=/root/miniconda3/
#PYTHON_MAIN=${CONDA_ENV_PATH}/bin/python
#export PATH=${CONDA_ENV_PATH}/bin:$PATH


#N_GPUS=`${PYTHON_MAIN} -c "import torch; print(torch.cuda.device_count())"`
WORKSPACE=./

cd ${WORKSPACE}

#ln -s /mnt/wyh/Dataset/WOD_CenterPoint ${WORKSPACE}/data/WOD_CenterPoint

#N_GPUS=`${PYTHON_MAIN} -c "import torch; print(torch.cuda.device_count())"`

echo "number of gpus"

#echo $N_GPUS

echo "print nvidia-smi"

nvidia-smi

echo "start to train the script"

# CONFIG_NAME: ./configs/center_graph_original_voxel.py
CONFIG_NAME=$1 
METHOD=$2
SPECIFIED_PATH=$3
NUM_UPDATE=4
RADIUS=2.0
TAG=$4

PYTHONPATH=$(pwd):$PYTHONPATH \
python dist_test.py $CONFIG_NAME \
    --data_path data/Waymo/WOD_$2 --root_dir ./ \
    --tag $TAG --ckpt_epoch 10 --ckpt_specified_path $3  --num_updates $NUM_UPDATE --radius_t $RADIUS --pcdet2det3d 0
