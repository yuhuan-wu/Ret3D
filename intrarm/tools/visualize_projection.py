import argparse
import os
import torch
from pathlib import Path

import cv2
import numpy as np

from centergraph.datasets.waymo import SingleSweepWaymoDataset
from centergraph.openpcdet_eval import load_waymo_pickle_data
from centergraph.utils import create_logger
from centergraph.utils.visualization import draw_point_cloud_and_color_boxes
from centergraph.torchie import Config


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


def boxes_projection(pose_d, cur_frame_id, pre_frame_id, previous_box):
    previous_pose = pose_d[pre_frame_id]
    current_pose = pose_d[cur_frame_id]
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    logger = create_logger(log_file='tmp.txt', log_level=cfg.log_level)
    data_path = Path(args.data_path)
    split = 'train'
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
    # gt_dict = dict()
    pose_d = dict()
    for i in range(len(infos)):
        f_id = infos[i]['frame_id']
        # gt_boxes_lidar = infos[i]['annos']['gt_boxes_lidar']
        pose = infos[i]['pose']
        # gt_dict[f_id] = gt_boxes_lidar
        pose_d[f_id] = pose

    # cur_gt_boxes = gt_dict['segment-10203656353524179475_7625_000_7645_000_with_camera_labels_001']
    # pre_gt_boxes = gt_dict['segment-10203656353524179475_7625_000_7645_000_with_camera_labels_000']
    # pre_gt_boxes = boxes_projection(pose_d, 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels_001',
    #                                 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels_000',
    #                                 pre_gt_boxes)

    cur_item = dataset[9822]
    cur_frame_id = cur_item['frame_id']
    cur_frame_id = cur_frame_id.split('/')[-1][:-4]
    cur_segment_id = cur_frame_id[:-4]
    cur_gt_boxes_and_cls = cur_item['gt_boxes_and_cls']
    cur_graph_data = cur_item['graph_data']

    pre_item = dataset[9821]
    pre_frame_id = pre_item['frame_id']
    pre_frame_id = pre_frame_id.split('/')[-1][:-4]
    pre_segment_id = pre_frame_id[:-4]
    pre_gt_boxes_and_cls = pre_item['gt_boxes_and_cls']
    pre_graph_data = pre_item['graph_data']

    cnt = len(cur_gt_boxes_and_cls) - 1
    while cnt > 0 and torch.sum(cur_gt_boxes_and_cls[cnt, :]) == 0:
        cnt -= 1
    cur_gt_boxes_and_cls = np.array(cur_gt_boxes_and_cls[:cnt + 1])
    cnt = len(pre_gt_boxes_and_cls) - 1
    while cnt > 0 and torch.sum(pre_gt_boxes_and_cls[cnt, :]) == 0:
        cnt -= 1
    pre_gt_boxes_and_cls = np.array(pre_gt_boxes_and_cls[:cnt + 1])

    # pre
    # [ 2.39495635e+00  7.21078873e+00  5.14274120e-01  1.98289609e+00
    #    4.76023531e+00  1.67999995e+00 -1.57296431e+00  2.01910591e+01
    #   -2.00295020e-02  1.00000000e+00]
    # after projection
    # [  0.78917167   7.21742472   0.51708853   1.98289609   4.76023531
    #     1.67999995   1.56874393]
    # cur
    # [ 2.8007905e+00  7.2200875e+00  5.1905364e-01  1.9828961e+00
    #    4.7602353e+00  1.6799999e+00 -1.5737985e+00  2.0171104e+01
    #   -4.7054053e-03  1.0000000e+00]

    cur_gt_boxes_and_cls = np.concatenate([cur_graph_data.pos, cur_graph_data.box], axis=1)
    pre_gt_boxes_and_cls = np.concatenate([pre_graph_data.pos, pre_graph_data.box], axis=1)

    # print(np.min(pre_gt_boxes_and_cls[:, 7]), np.max(pre_gt_boxes_and_cls[:, 7]))
    # print(np.min(pre_gt_boxes_and_cls[:, 8]), np.max(pre_gt_boxes_and_cls[:, 8]))

    # mask = np.abs(pre_gt_boxes_and_cls[:,7]) <= 5.0
    # pre_gt_boxes_and_cls = pre_gt_boxes_and_cls[mask]
    # mask = np.abs(pre_gt_boxes_and_cls[:, 8]) <= 5.0
    # pre_gt_boxes_and_cls = pre_gt_boxes_and_cls[mask]

    pre_gt_boxes_and_cls[:, 0] += pre_gt_boxes_and_cls[:, 7] * 0.1
    pre_gt_boxes_and_cls[:, 1] += pre_gt_boxes_and_cls[:, 8] * 0.1

    p_boxes = boxes_projection(pose_d, cur_frame_id, pre_frame_id, pre_gt_boxes_and_cls[:,:7])
    c_boxes = cur_gt_boxes_and_cls[:,:7]

    color_box_pairs = [(COLORMAP["pupple"], c_boxes),
                       (COLORMAP["skublue"], p_boxes)]
    pcd_file = os.path.join(args.point_cloud_path, cur_segment_id, '%04d.npy' % int(1))
    pcd_current_np = np.load(pcd_file)

    img = draw_point_cloud_and_color_boxes(pcd_current_np, color_box_pairs,
                                           FWD_RANGE, SIDE_RANGE, VOXEL_SIZE)

    img_name = os.path.join(args.visual_dir, "%s_%s.png" % (split, cur_frame_id))
    cv2.imwrite(img_name, img)


if __name__ == "__main__":
    main()
