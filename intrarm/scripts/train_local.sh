#!/bin/bash
set -e
set -x

echo " "
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "starting pgnet train script"
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo " "

CONDA_ENV_PATH=/root/miniconda3/
PYTHON_MAIN=${CONDA_ENV_PATH}/bin/python
export PATH=${CONDA_ENV_PATH}/bin:$PATH


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

MASTER_PORT=$((RANDOM + 10000))


# CONFIG_NAME: ./configs/center_graph_original_voxel.py
CONFIG_NAME=$1 
NUM_UPDATE=$2
RADIUS=$3
TAG=ablation_$4_${NUM_UPDATE}_$RADIUS

PYTHONPATH=$(pwd):$PYTHONPATH \
python dist_train.py   $CONFIG_NAME \
    --data_path data/WOD_CenterPoint --root_dir ./pami_r1/ \
    --tag $TAG --num_updates $NUM_UPDATE --radius_t $RADIUS --pcdet2det3d 0

echo "epoch 10"

PYTHONPATH=$(pwd):$PYTHONPATH \
python dist_test.py $CONFIG_NAME \
    --data_path data/WOD_CenterPoint --root_dir ./pami_r1/ \
    --tag $TAG --ckpt_epoch 10  --num_updates $NUM_UPDATE --radius_t $RADIUS --pcdet2det3d 0
