import argparse
import time
import pickle
import numpy as np
import os
import glob
from pathlib import Path

import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from centergraph.datasets.waymo import SingleSweepWaymoDataset, MultiSweepWaymoDataset
from centergraph.models.detectors import GraphDetector
from centergraph.openpcdet_eval import generate_prediction_dicts, load_waymo_pickle_data, evaluation
from centergraph.torchie import Config
from centergraph.utils import create_logger, load_data_to_gpu


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("config", help="config file path")
    parser.add_argument('--ckpt_epoch', type=int, required=True, default=None,
                        help='checkpoint epoch number, the whole path is determined by root_dir and tag')
    parser.add_argument('--tag', type=str, default=None, required=True, help='experiment tag')
    parser.add_argument('--split', type=str, required=True, default=None,
                        help='create submission file for val or test set')

    parser.add_argument('--data_path', type=str, default='/mnt/wyh/Dataset/WOD_CenterPoint',
                        help='specify the data location')
    parser.add_argument('--root_dir', type=str, default='/mnt/wyh/Codebase/pgnet',
                        help='specify the data location')

    args = parser.parse_args()
    return args


def save_pred(pred, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(pred, f)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    if args.root_dir is not None:
        root_dir = Path(args.root_dir)
    else:
        root_dir = Path(cfg.root_dir)
    output_dir = root_dir / 'output' / 'centergraph' / args.tag
    if args.data_path is not None:
        data_path = Path(args.data_path)
    else:
        data_path = Path(cfg.data_path)
    ckpt_dir = output_dir / 'ckpt'

    log_file = output_dir / ('log_%s_%s.txt' % (args.split, args.tag))
    logger = create_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('**********************Start logging**********************')
    logger.info('{:16} {}'.format('TAG', args.tag))

    # create metadata pickle files BEGIN
    split = args.split
    target_file = data_path / (split + '_annos_for_submission.pkl')
    if not os.path.isfile(target_file):
        submission_annos_dict = dict()
        logger.info('Creating submission annotation file for %s split...' % split)
        center_point_anno_root = Path('/mnt/wyh/Dataset/waymo_center_point')
        center_point_anno_root = center_point_anno_root / split / 'annos'
        center_point_pkl_files = glob.glob(str(center_point_anno_root / '*.pkl'))
        for i, pkl_file in enumerate(tqdm(center_point_pkl_files)):
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
            frame_id = 'segment-%s_with_camera_labels_%03d' % (obj['scene_name'], obj['frame_id'])
            centerpoint_token = pkl_file.split('/')[-1].strip()[:-4]
            context_name = obj['scene_name']
            frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])
            submission_annos_dict[frame_id] = {'scene_name': context_name, 'timestamp': frame_timestamp_micros,
                                               'veh_to_global': obj['veh_to_global'],
                                               'centerpoint_token': centerpoint_token}
        with open(target_file, 'wb') as f:
            pickle.dump(submission_annos_dict, f)
        logger.info('submission annotation file saved at %s.' % target_file)
    else:
        with open(target_file, 'rb') as f:
            submission_annos_dict = pickle.load(f)
        logger.info('submission annotation file existed at %s.' % target_file)
    # create metadata pickle files END

    model = GraphDetector(cfg.head.input_channels, cfg.head.model_cfg, cfg.head.type)

    if cfg.head.type == 'RoIHead':
        dataset = SingleSweepWaymoDataset(data_path, split, logger, 0.0,
                                          sample_interval=cfg.data.val.sampling_interval)
    elif cfg.head.type in ['GraphHead', 'MiXHead'] and cfg.head.model_cfg.GRAPH_TYPE == 's':
        dataset = SingleSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                          sample_interval=cfg.data.val.sampling_interval)
    elif cfg.head.type in ['GraphHead', 'MiXHead'] and cfg.head.model_cfg.GRAPH_TYPE == 't':
        if split != 'test':
            dataset = MultiSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                             sample_interval=cfg.data.val.sampling_interval)
        else:
            pose_d = dict()
            for key in submission_annos_dict:
                pose_d[key] = np.reshape(submission_annos_dict[key]['veh_to_global'], [4, 4])
            dataset = MultiSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                             sample_interval=cfg.data.val.sampling_interval,
                                             pose_d=pose_d)
    else:
        logger.info('NO IMPLEMENTED DATASET TYPE!')
        exit()
    dataloader = DataLoader(dataset, batch_size=cfg.data.val.batch_size, pin_memory=False,
                            num_workers=cfg.data.val.num_workers, shuffle=False, drop_last=False)

    # fix_weight_path = cfg.fix_weight_path
    # model.load_params_from_file(fix_weight_path)

    ckpt_file = ckpt_dir / ('checkpoint_epoch_%d.pth' % args.ckpt_epoch)
    model.load_params_from_file(ckpt_file)

    model.cuda()
    model.eval()
    logger.info('*************** EVALUATION *****************')

    with torch.no_grad():
        det_results = list()
        for i, batch_dict in enumerate(tqdm(dataloader)):
            load_data_to_gpu(batch_dict)
            results = model(batch_dict, return_loss=False)
            # annos = generate_prediction_dicts(results)
            det_results += results

        out_file = 'prediction_%s.pkl' % split
        out_path = output_dir / out_file
        save_pred(det_results, out_path)
        logger.info("results saved to {}".format(out_path))


if __name__ == "__main__":
    main()
