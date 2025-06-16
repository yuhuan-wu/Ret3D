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

### Train

Under construction. Code will be released soon.

### Evaluation

Under construction.

### Citation

````
@article{wu2025ret3d,
  author  = {Wu, Yu-Huan and Zhang, Da and Liu, Yun and others},
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
