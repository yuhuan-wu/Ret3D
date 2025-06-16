import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from centergraph.datasets.waymo.base import WaymoDataset
from centergraph.datasets.waymo.single_sweep import get_valid_length
from centergraph.openpcdet_eval import load_waymo_pickle_data


class MultiSweepWaymoDataset(WaymoDataset):
    def __init__(self, info_path, split, logger, radius_t, knn=999, sample_interval=10, pose_d=None, pcdet2det3d=False):
        super(MultiSweepWaymoDataset, self).__init__(info_path, split, logger, interval=sample_interval)
        self.radius_t = radius_t
        self.knn = knn
        self.num_sweeps = 2
        if pose_d is None:
            pose_d = dict()
            infos = load_waymo_pickle_data(data_dir=info_path, split=split, logger=logger)
            for i in range(len(infos)):
                frame_id = infos[i]['frame_id']
                pose = infos[i]['pose']
                pose_d[frame_id] = pose
            self.pose_d = pose_d
        else:
            self.pose_d = pose_d
        self.pcdet2det3d = pcdet2det3d

    def __getitem__(self, idx):
        cur_res = self.get_sensor_data(idx)
        cur_valid_len = get_valid_length(cur_res['rois'])
        for key in cur_res:
            if key in ['rois', 'roi_features', 'roi_labels', 'roi_scores']:
                cur_res[key] = cur_res[key][:cur_valid_len]
        seq_name = cur_res['frame_id'][:-8]
        frame_id = int(cur_res['frame_id'][-7:-4])
        ret_dict = dict()
        if self.split == 'train':
            ret_dict['gt_boxes_and_cls'] = cur_res['gt_boxes_and_cls']
        ret_dict['frame_id'] = cur_res['frame_id']
        if frame_id != 0 and self.num_sweeps == 2:
            pre_file = '%s/%03d.npz' % (seq_name, frame_id - 1)
            pre_res = self.get_sensor_data_by_name(pre_file)
            pre_valid_len = get_valid_length(pre_res['rois'])
            for key in pre_res:
                if key in ['rois', 'roi_features', 'roi_labels', 'roi_scores']:
                    pre_res[key] = pre_res[key][:pre_valid_len]
            cur_frame_id = cur_res['frame_id'].split('/')[-1][:-4]
            pre_frame_id = pre_res['frame_id'].split('/')[-1][:-4]

            # mask = pre_res['roi_scores'] >= 0.3
            # pre_res['rois'] = pre_res['rois'][mask]
            # pre_res['roi_features'] = pre_res['roi_features'][mask]
            # pre_res['roi_labels'] = pre_res['roi_labels'][mask]
            # pre_res['roi_scores'] = pre_res['roi_scores'][mask]
            try: 
                pre_res['rois'][:, 0] += pre_res['rois'][:, 7] * 0.1
                pre_res['rois'][:, 1] += pre_res['rois'][:, 8] * 0.1
            except:
                pass

            pre_rois = self.boxes_projection(cur_frame_id, pre_frame_id, pre_res['rois'][:, :7])
            pre_res['rois'][:, :7] = torch.Tensor(pre_rois)

            rois = torch.cat([cur_res['rois'], pre_res['rois']], dim=0)
            roi_features = torch.cat([cur_res['roi_features'], pre_res['roi_features']], dim=0)
            roi_labels = torch.cat([cur_res['roi_labels'], pre_res['roi_labels']], dim=0)
            roi_scores = torch.cat([cur_res['roi_scores'], pre_res['roi_scores']], dim=0)
            cl = cur_res['rois'].shape[0]
            jl = cl + pre_res['rois'].shape[0]

            data = Data(x=roi_features, edge_index=None, pos=rois[:, :3], box=rois[:, 3:], cl=cl, jl=jl,
                        roi_labels=roi_labels, roi_scores=roi_scores)
            #transform = T.RadiusGraph(r=self.radius_t)
            # transform = T.KNNGraph(k=6)
            #data = transform(data)
            ret_dict['graph_data'] = data
        elif frame_id > 1 and self.num_sweeps == 3:
            pre_file = '%s/%03d.npz' % (seq_name, frame_id - 1)
            pre_res = self.get_sensor_data_by_name(pre_file)
            pre_valid_len = get_valid_length(pre_res['rois'])
            for key in pre_res:
                if key in ['rois', 'roi_features', 'roi_labels', 'roi_scores']:
                    pre_res[key] = pre_res[key][:pre_valid_len]
            cur_frame_id = cur_res['frame_id'].split('/')[-1][:-4]
            pre_frame_id = pre_res['frame_id'].split('/')[-1][:-4]

            # mask = pre_res['roi_scores'] >= 0.3
            # pre_res['rois'] = pre_res['rois'][mask]
            # pre_res['roi_features'] = pre_res['roi_features'][mask]
            # pre_res['roi_labels'] = pre_res['roi_labels'][mask]
            # pre_res['roi_scores'] = pre_res['roi_scores'][mask]
            try: 
                pre_res['rois'][:, 0] += pre_res['rois'][:, 7] * 0.1
                pre_res['rois'][:, 1] += pre_res['rois'][:, 8] * 0.1
            except:
                pass

            pre_rois = self.boxes_projection(cur_frame_id, pre_frame_id, pre_res['rois'][:, :7])
            pre_res['rois'][:, :7] = torch.Tensor(pre_rois)

            rois = torch.cat([cur_res['rois'], pre_res['rois']], dim=0)
            roi_features = torch.cat([cur_res['roi_features'], pre_res['roi_features']], dim=0)
            roi_labels = torch.cat([cur_res['roi_labels'], pre_res['roi_labels']], dim=0)
            roi_scores = torch.cat([cur_res['roi_scores'], pre_res['roi_scores']], dim=0)
            cl = cur_res['rois'].shape[0]
            jl = cl + pre_res['rois'].shape[0]

            data = Data(x=roi_features, edge_index=None, pos=rois[:, :3], box=rois[:, 3:], cl=cl, jl=jl,
                        roi_labels=roi_labels, roi_scores=roi_scores)
            #transform = T.RadiusGraph(r=self.radius_t)
            # transform = T.KNNGraph(k=6)
            #data = transform(data)
            ret_dict['graph_data'] = data
        else:
            cl = cur_res['rois'].shape[0]
            jl = cl
            data = Data(x=cur_res['roi_features'], edge_index=None, pos=cur_res['rois'][:, :3],
                        box=cur_res['rois'][:, 3:], cl=cl, jl=jl,
                        roi_labels=cur_res['roi_labels'], roi_scores=cur_res['roi_scores'])
            #transform = T.RadiusGraph(r=self.radius_t)
            # transform = T.KNNGraph(k=6)
            #data = transform(data)
            ret_dict['graph_data'] = data

        return ret_dict

    def boxes_projection(self, cur_frame_id, pre_frame_id, previous_box):
        previous_pose = self.pose_d[pre_frame_id]
        current_pose = self.pose_d[cur_frame_id]
        new_p = transform_p_to_c(previous_box[:, :3], previous_pose, current_pose)
        x = np.expand_dims(np.cos(previous_box[:, -1]), axis=-1)
        y = np.expand_dims(np.sin(previous_box[:, -1]), axis=-1)
        dir = np.concatenate([x, y, np.zeros([x.shape[0], 1])], axis=-1)
        ori = np.zeros([x.shape[0], 3])
        new_dir = transform_p_to_c(dir, previous_pose, current_pose)
        new_ori = transform_p_to_c(ori, previous_pose, current_pose)
        tan_theta = (new_dir[:, 1] - new_ori[:, 1]) / (new_dir[:, 0] - new_ori[:, 0])
        theta = np.arctan(tan_theta)
        theta = np.expand_dims(theta, axis=-1)
        new_box = np.concatenate([new_p[:, :3], previous_box[:, 3:6], theta], axis=-1)
        return new_box


def transform_p_to_c(pp, t_p_w, t_c_w):
    """
    :param pp: (N, 3)
    :param t_p_w: (4,4)
    :param t_c_w: (4,4)
    :return: new_p： (N, 3)
    """
    N = pp.shape[0]
    t_p_w = np.tile(t_p_w, [N, 1, 1])
    t_c_w = np.tile(t_c_w, [N, 1, 1])

    pp = np.concatenate([pp, np.ones([N, 1])], axis=1)
    pp = np.expand_dims(pp, axis=-1)
    t_c_w_inv = np.linalg.inv(t_c_w)
    new_p = np.matmul(t_c_w_inv, np.matmul(t_p_w, pp))
    return new_p[:, :, 0]


def combine_valid_data(data1, data2, idx1, idx2):
    if len(data1.shape) == 2:
        ret = data1.new_zeros(data1.shape[0] * 2, data1.shape[1])
        res = torch.cat([data1[:idx1, :], data2[:idx2, :]], dim=0)
        ret[:res.shape[0], :] = res
    else:
        ret = data1.new_zeros(data1.shape[0] * 2)
        res = torch.cat([data1[:idx1], data2[:idx2]], dim=0)
        ret[:res.shape[0]] = res
    return ret