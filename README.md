## Ret3D: Rethinking Object Relations for Efficient 3D Object Detection

### Introduction 

Current efficient LiDAR-based detection frameworks are lacking in exploiting object relations, which
naturally present in both spatial and temporal manners. To this end, we introduce a simple, efficient, and effective
two-stage detector, termed as Ret3D. At the core of Ret3D is the utilization of novel intra-frame and inter-frame
relation modules to capture the spatial and temporal relations accordingly. More Specifically, intra-frame relation
module (InterRM) encapsulates the intra-frame objects into a sparse graph and thus allows us to refine the object
features through efficient message passing. On the other hand, inter-frame relation module (IntraRM) densely
connects each object in its corresponding tracked sequences dynamically, and leverages such temporal information
to further enhance its representations efficiently through a lightweight transformer network. We instantiate our
novel designs of IntraRM and InterRM with general center-based or anchor-based detectors and evaluate them on
Waymo Open Dataset (WOD). With negligible extra overhead, Ret3D achieves the state-of-the-art performance,
being 2.9% and 3.2% higher than the recent competitor in terms of the LEVEL_1 and LEVEL_2 mAPH metrics
on vehicle detection, respectively.

### Requirements & Installtion

- python >= 3.5
- torch == 1.7.1
- torch-geometric == 1.6.3

For torch-geometric, use the following code to install the precompiled packages, which can speed up the installtion:
```
pip install --no-index torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && \
pip install --no-index torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && \
pip install --no-index torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && \
pip install --no-index torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html && \
pip install torch-geometric==1.6.3
```

After installation, run the following code:
```
cd intrarm
bash setup.sh
```

### Train

On Waymo Open Dataset, our code supports three types of model for training: SECOND, CenterPoint, and CenterPoint++.

Taking CenterPoint as the example, you need to prepare a pretrained model of two-stage of CenterPoint first, and put it to `./data/pretrained_weights/epoch_6.pth`. This model is utilized to intialize the MLP feature extraction layers.
Besides, you are required to pre-compute the object features and detection results of the one-stage CenterPoint. Check the dataset files `intrarm/centergraph/datasets/waymo/multi_sweep.py` for pre-computed data format.

Then, use the following bash script for training `intrarm/scripts/train_local.sh`.

```
cd intrarm
bash scripts/train_local.sh $CONFIG_NAME $NUM_UPDATE $RADIUS

# As default we set:
# CONFIG_NAME=configs/center_graph_original_voxel.py
# NUM_UPDATE=4
# RADIUS=2.0
```

If you would like to train IntraRM on SECOND, you can use the official `OpenPCDet` repository to pre-compute the object features and detection results.

### Evaluation

After training,
`intrarm/scripts/test_local.sh` controls the evaluation process.

```
bash scripts/test_local.sh $CONFIG_NAME $METHOD $SPECIFIED_PATH $TAG


# CONFIG_NAME: ./configs/center_graph_original_voxel.py
# $METHOD CenterPoint
# $SPECIFIED_PATH is the pretrained model path
# TAG is save_path
```

The evaluation code uses the toolkit of `OpenPCDet` to evaluate the performance.

### Citation

````
@article{wu2025ret3d,
  author  = {Wu, Yu-Huan and Zhang, Da and Liu, Yun and Zhang, Le and Cheng, Ming-Ming},
  title   = {Ret3D: rethinking object relations for efficient 3D object detection},
  journal = {Sci~Sin~Inform},
  year    = {2025},
  volume  = {55},
  number  = {4},
  pages   = {887--901},
  doi     = {10.1360/SSI-2024-0295},
  url     = {https://doi.org/10.1360/SSI-2024-0295}
}
````
