import argparse
import time
import os
import pickle
from pathlib import Path

import tensorflow as tf # fix the bug in 

import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from centergraph.datasets.waymo import SingleSweepWaymoDataset, MultiSweepWaymoDataset
from centergraph.models.detectors import GraphDetector
from centergraph.openpcdet_eval import generate_prediction_dicts, load_waymo_pickle_data, evaluation
from centergraph.torchie import Config
from centergraph.utils import create_logger, load_data_to_gpu

from copy import deepcopy

from time import time

def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("config", help="config file path")
 
    parser.add_argument('--data_path', type=str, default='./data/WOD_CenterPoint',
                        help='specify the data location')
    parser.add_argument('--root_dir', type=str, default='./',
                        help='specify the data location')

    parser.add_argument('--pcdet2det3d', type=int, default=0, required=False)
    parser.add_argument('--save-path', default="./pkls/cp_voxel_onestage_train_kittiformat.pkl", type=str, help='saving path for the pkl file')

    args = parser.parse_args()
    return args



def post_process(cfg, batch_dict):
    batch_dict['batch_size'] = len(batch_dict['frame_id'])
    batch_dict['rois'] = torch.cat([batch_dict['graph_data'].pos, batch_dict['graph_data'].box], dim=-1)
    batch_dict['roi_features'] = batch_dict['graph_data'].x
    
    batch_size = batch_dict['batch_size']
    pred_dicts = []

    for index in range(batch_size):
        batch_mask = batch_dict['graph_data'].batch == index
        box_preds = batch_dict['rois'][batch_mask]
        cls_preds = batch_dict['graph_data'].roi_scores[batch_mask]
        label_preds = batch_dict['graph_data'].roi_labels[batch_mask]

        if cfg.head.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'recls':
            cls_preds = torch.sigmoid(cls_preds)
            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            box_preds = box_preds
            scores = cls_preds
            labels = label_preds
        elif cfg.head.model_cfg.TARGET_CONFIG.CLS_SCORE_TYPE == 'roi_iou':
            cls_preds = cls_preds
            box_preds = box_preds
            scores = cls_preds
            label_preds = label_preds - 1
            labels = label_preds

        if box_preds.shape[-1] == 9:
            # move rotation to the end (the create submission file will take elements from 0:6 and -1)
            box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

        #box_preds = box_preds + 0.001 - 0.001 # speed test
        
        # currently don't need nms
        pred_dict = {
            'box3d_lidar': box_preds,
            'scores': scores,
            'label_preds': labels,
            "metadata": batch_dict["frame_id"][index]
        }

        pred_dicts.append(pred_dict)

    return pred_dicts


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    root_dir = Path(args.root_dir)
    data_path = Path(args.data_path)
    log_file = 'log_gen_onestage_%s.txt'
    
    logger = create_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('**********************Start logging**********************')
    logger.info(args)

    split = 'train'
    if cfg.head.type == 'RoIHead':
        dataset = SingleSweepWaymoDataset(data_path, split, logger, 0.0,
                                          sample_interval=cfg.data.val.sampling_interval)
    elif cfg.head.type in ['GraphHead', 'MiXHead', 'DevHead'] and cfg.head.model_cfg.GRAPH_TYPE == 's':
        dataset = SingleSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                          sample_interval=cfg.data.val.sampling_interval)
    elif cfg.head.type in ['GraphHead', 'MiXHead', 'DevHead'] and cfg.head.model_cfg.GRAPH_TYPE == 't':
        dataset = MultiSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                         sample_interval=cfg.data.val.sampling_interval, pcdet2det3d=args.pcdet2det3d)
    else:
        logger.info('NO IMPLEMENTED DATASET TYPE!')
        exit()
    dataloader = DataLoader(dataset, batch_size=cfg.data.val.batch_size, pin_memory=False,
                            num_workers=cfg.data.val.num_workers, shuffle=False, drop_last=False)

    # fix_weight_path = cfg.fix_weight_path
    # model.load_params_from_file(fix_weight_path)
    
    logger.info('*************** EVALUATION *****************')

    with torch.no_grad():
        det_annos = list()
        det_annos_kitti = list()
        total_time = 0
        total_times = 0
        for i, batch_dict in enumerate(tqdm(dataloader)):
            load_data_to_gpu(batch_dict)
            t0 = time()
            results = post_process(cfg, batch_dict)
            annos = generate_prediction_dicts(results)
            det_annos += annos
            det_annos_kitti += results

        # yes, kitti format!!!
        with open(args.save_path, "wb") as f:
             pickle.dump(det_annos_kitti, f)

        infos = load_waymo_pickle_data(data_dir=data_path, split=split, logger=logger, pic_data=dataset._waymo_infos)
        result_str, ap_dict = evaluation(det_annos, infos=infos)
        
        classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        level1_aph = 0.
        level2_aph = 0.
        level1_ap = 0.
        level2_ap = 0.
        for key in list(ap_dict.keys()):
            if classes[0] in key or classes[1] in key or classes[2] in key:
                if 'LEVEL_1/APH' in key:
                    level1_aph += ap_dict[key]
                elif 'LEVEL_2/APH' in key:
                    level2_aph += ap_dict[key]
                elif 'LEVEL_1/AP' in key and 'LEVEL_1/APL' not in key:
                    level1_ap += ap_dict[key]
                elif 'LEVEL_2/AP' in key and 'LEVEL_2/APL' not in key:
                    level2_ap += ap_dict[key]
        level1_aph /= 3
        level2_aph /= 3
        level1_ap /= 3
        level2_ap /= 3
        logger.info("level2_aph: {:.4f}, level2_ap: {:.4f}, level1_aph: {:.4f}, level1_ap: {:.4f}".format(level2_aph, level2_ap, level1_aph, level1_ap))
        #logger.info('Dataset evaluation finished: %.1f seconds).' % sec_eval)
        logger.info(result_str)
        logger.info("level2_aph: {:.4f}, level2_ap: {:.4f}, level1_aph: {:.4f}, level1_ap: {:.4f}".format(level2_aph, level2_ap, level1_aph, level1_ap))


if __name__ == "__main__":
    main()
