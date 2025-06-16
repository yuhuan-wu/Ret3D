import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

from centergraph.datasets.waymo.base import WaymoDataset


class SingleSweepWaymoDataset(WaymoDataset):
    def __init__(self, info_path, split, logger, radius_t, sample_interval=10):
        super(SingleSweepWaymoDataset, self).__init__(info_path, split, logger, interval=sample_interval)
        self.radius_t = radius_t

    def __getitem__(self, idx):
        res = self.get_sensor_data(idx)
        valid_len = get_valid_length(res['rois'])
        for key in res:
            if key in ['rois', 'roi_features', 'roi_labels', 'roi_scores']:
                res[key] = res[key][:valid_len]
        data = Data(x=res['roi_features'], edge_index=None, pos=res['rois'][:, :3], box=res['rois'][:, 3:],
                    roi_labels=res['roi_labels'], roi_scores=res['roi_scores'])
        # transform = T.RadiusGraph(r=self.radius_t)
        transform = T.KNNGraph(k=6)
        data = transform(data)
        res['graph_data'] = data
        res.pop('rois')
        res.pop('roi_features')
        res.pop('roi_labels')
        res.pop('roi_scores')
        return res


def get_valid_length(data):
    cnt = len(data) - 1
    while cnt > -1 and torch.sum(data[cnt, :]) == 0:
        cnt -= 1
    return cnt + 1
