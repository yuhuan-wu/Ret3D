import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from centergraph.datasets.waymo import SingleSweepWaymoDataset
from centergraph.openpcdet_eval import load_waymo_pickle_data
from centergraph.torchie import Config

from centergraph.utils import create_logger
from centergraph.utils.visualization import draw_point_cloud_and_color_boxes, draw_point_cloud_and_color_boxes_with_score


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("config", help="config file path")

    parser.add_argument('--data_path', type=str, default='/mnt/wyh/Dataset/WOD_CenterPoint',
                        help='specify the data location')
    parser.add_argument('--point_cloud_path', type=str, default='/mnt/wyh/Dataset/waymo/waymo_processed_data',
                        help='specify the data location')
    parser.add_argument('--visual_dir', type=str, default='/mnt/wyh/Codebase/pgnet/vis',
                        help='specify the data location')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    logger = create_logger(log_file='tmp.txt', log_level=cfg.log_level)
    data_path = Path(args.data_path)
    split = 'val'
    dataset = SingleSweepWaymoDataset(data_path, split, logger, 0.0)

    # some constant variables
    COLORMAP = {"white": (255, 255, 255),
                "pupple": (255, 0, 255),
                "skublue": (255, 255, 0),
                "red": (0, 0, 255),
                "yellow": (0, 255, 255), }

    FWD_RANGE = np.array([-80, 80], dtype=np.double)  # move from velodyne to camera
    SIDE_RANGE = np.array([-80, 80], dtype=np.double)
    VOXEL_SIZE = np.array([0.05, 0.05, 1.0])

    infos = load_waymo_pickle_data(data_dir=data_path, split=split, logger=logger)
    gt_dict = dict()
    for i in range(len(infos)):
        f_id = infos[i]['frame_id']
        gt_boxes_lidar = infos[i]['annos']['gt_boxes_lidar']
        gt_dict[f_id] = gt_boxes_lidar

    with open("./prediction_nms_old.pkl", "rb") as f:
        pic_data = pickle.load(f)

    for i in range(10):
        idx = i
        roi_boxes = pic_data[idx]['boxes_lidar']
        score = np.expand_dims(pic_data[idx]['score'], axis=1)
        roi_boxes = np.concatenate([roi_boxes, score], axis=-1)
        frame_id = pic_data[idx]['frame_id']
        segment_id = frame_id[:-4]
        # item = dataset[idx]
        # frame_id = item['frame_id']
        # frame_id = frame_id.split('/')[-1][:-4]
        # segment_id = frame_id[:-4]
        # if split == 'train':
        #     gt_boxes_and_cls = item['gt_boxes_and_cls']
        # graph_data = item['graph_data']

        # roi_boxes = np.concatenate([graph_data.pos, graph_data.box[:, :4]], axis=1)
        if split == 'train':
            cnt = len(gt_boxes_and_cls) - 1
            while cnt > 0 and torch.sum(gt_boxes_and_cls[cnt, :]) == 0:
                cnt -= 1
            aug_gt_boxes = np.array(gt_boxes_and_cls[:cnt + 1, :7])
        gt_boxes = gt_dict[frame_id]
        tmp = np.ones([gt_boxes.shape[0], 1])
        gt_boxes = np.concatenate([gt_boxes, tmp], axis=-1)

        color_box_pairs = [(COLORMAP["pupple"], roi_boxes),
                           (COLORMAP["skublue"], gt_boxes)]
        if split == 'train':
            color_box_pairs = [(COLORMAP["pupple"], roi_boxes),
                               (COLORMAP["skublue"], gt_boxes),
                               (COLORMAP["yellow"], aug_gt_boxes)]
        pcd_file = os.path.join(args.point_cloud_path, segment_id, '%04d.npy' % int(frame_id[-3:]))
        pcd_current_np = np.load(pcd_file)

        img = draw_point_cloud_and_color_boxes_with_score(pcd_current_np, color_box_pairs,
                                               FWD_RANGE, SIDE_RANGE, VOXEL_SIZE)

        img_name = os.path.join(args.visual_dir, "nms_old_%s_%s.png" % (split, frame_id))
        cv2.imwrite(img_name, img)


if __name__ == "__main__":
    main()
