import os
import torch
import glob
import numpy as np
from tqdm import tqdm

import torch.utils.data as torch_data
import torch.nn.functional as F

class WaymoDataset(torch_data.Dataset):
    def __init__(self, info_path, split, logger, interval=10):
        super(WaymoDataset, self).__init__()
        self._info_path = info_path
        self.split = split
        self.logger = logger
        self.interval = interval
        self.load_infos(self._info_path / self.split)
        self.pcdet2det3d = False

    def load_infos(self, npz_path):
        data_dirs = os.listdir(str(npz_path))
        datas = []
        print("loading {} sequences".format(len(data_dirs)))
        for data_dir in tqdm(data_dirs):
            data = glob.glob(str(npz_path / data_dir / '*.npz'))
            data.sort()
            datas.extend(data)
        #data = glob.glob(str(npz_path / '*.npz'))
        datas.sort()
        datas = datas[::self.interval]
        # data = data[:32]
        self._waymo_infos = datas
        self.logger.info("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):
        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path / self.split)
        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]
        data = np.load(info, allow_pickle=True)
        res = dict()
        res['frame_id'] = os.path.dirname(str(info)) + '_' + os.path.basename(str(info))
        res['rois'] = torch.Tensor(data['rois']).squeeze(0)
        # yuhuan: pad boxes [N, 7] to [N, 9], add zero velocity for fitting current code!
        if res['rois'].shape[-1] == 7:
            res['rois'] = F.pad(res['rois'], (0, 2)) 
        if self.pcdet2det3d:
            # waymo to kitti
            res['rois'][:, [3,4]] = res['rois'][:, [4,3]]
            res['rois'][:, 6] = - np.pi / 2 - res['rois'][:, 6]
            
        res['roi_features'] = torch.Tensor(data['roi_features']).squeeze(0)
        res['roi_labels'] = torch.Tensor(data['roi_labels']).long().squeeze(0)
        res['roi_scores'] = torch.Tensor(data['roi_scores']).squeeze(0)
        if self.split == 'train':
            res['gt_boxes_and_cls'] = torch.Tensor(data['gt_boxes_and_cls']).squeeze(0)
        return res

    def get_sensor_data_by_name(self, info):
        data = np.load(info, allow_pickle=True)
        res = dict()
        res['frame_id'] = os.path.dirname(str(info)) + '_' + os.path.basename(str(info))
        res['rois'] = torch.Tensor(data['rois']).squeeze(0)
        if res['rois'].shape[-1] == 7:
            res['rois'] = F.pad(res['rois'], (0, 2)) 
        if self.pcdet2det3d:
            res['rois'][:, [3,4]] = res['rois'][:, [4,3]]
            res['rois'][:, 6] = - np.pi / 2 - res['rois'][:, 6]
        res['roi_features'] = torch.Tensor(data['roi_features']).squeeze(0)
        res['roi_labels'] = torch.Tensor(data['roi_labels']).long().squeeze(0)
        res['roi_scores'] = torch.Tensor(data['roi_scores']).squeeze(0)
        if self.split == 'train':
            res['gt_boxes_and_cls'] = torch.Tensor(data['gt_boxes_and_cls']).squeeze(0)
        return res

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)
