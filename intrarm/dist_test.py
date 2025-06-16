import argparse
import time
import os
import pickle
from pathlib import Path

#import tensorflow as tf # fix the bug in 

import torch
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from torch_geometric.data import DataLoader
from tqdm import tqdm

from centergraph.datasets.waymo import SingleSweepWaymoDataset, MultiSweepWaymoDataset
from centergraph.models.detectors import GraphDetector
from centergraph.openpcdet_eval import generate_prediction_dicts, load_waymo_pickle_data, evaluation
from centergraph.torchie import Config
from centergraph.utils import create_logger, load_data_to_gpu

from copy import deepcopy

#from mindfree_utils import evaluation_utils

from time import time


import psutil, random

# Determine the number of CPU cores you want to use
num_cores = 8

# Get the current process ID
pid = os.getpid()

# Get the available CPU cores
cpu_count = psutil.cpu_count()

# Create a CPU affinity mask
cpu_affinity_mask = random.sample(range(cpu_count), num_cores)

# Set the CPU affinity for the current process
os.sched_setaffinity(pid, cpu_affinity_mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("config", help="config file path")
    parser.add_argument('--ckpt_epoch', type=int, required=True, default=None,
                        help='checkpoint epoch number, the whole path is determined by root_dir and tag')
    parser.add_argument('--ckpt_specified_path', type=str, default=None, required=False, help="ckpt_specified_path")
    parser.add_argument('--tag', type=str, default=None, required=True, help='experiment tag')
 
    parser.add_argument('--data_path', type=str, default='./data/Waymo/WOD_CenterPoint++',
                        help='specify the data location')
    parser.add_argument('--root_dir', type=str, default='./',
                        help='specify the data location')

    parser.add_argument('--radius_t', type=float, default=2.0, help='radius or knn number of constructing graph')
    parser.add_argument('--use_knn', type=int, default=0, help='use knn')
    parser.add_argument('--num_updates', type=int, default=4, help='number of gnns in IntraRM')

    parser.add_argument('--hadflow', action='store_true', default=False, help='whether to train on AutoDrive')
    parser.add_argument('--pcdet2det3d', type=int, default=1, required=False)

    parser.add_argument('--submission', type=int, default=0, required=False)

    args = parser.parse_args()
    return args


def merge_config_from_args(cfg, args):
    cfg.head.model_cfg.EDGE_RADIUS = args.radius_t
    cfg.radius_t = args.radius_t
    cfg.head.model_cfg.USE_KNN = args.use_knn
    cfg.head.model_cfg.NUM_UPDATES = args.num_updates


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    merge_config_from_args(cfg, args)

    root_dir = Path(args.root_dir)
    output_dir = root_dir / 'output/' / args.tag
    data_path = Path(args.data_path)
    ckpt_dir = output_dir / 'ckpt'
    log_file = output_dir / ('log_val_%s.txt' % args.tag)

    if args.hadflow:
        log_file = '/evaluation_result/log.txt'

    if args.ckpt_specified_path is not None:
        print("using ckpt_specified_path", args.ckpt_specified_path)
        base_dir = Path(os.path.dirname(args.ckpt_specified_path))
        output_dir = base_dir
        log_file = output_dir / ('log_val_%s.txt' % args.tag)
    
    logger = create_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('**********************Start logging**********************')
    logger.info(args)
    logger.info('{:16} {}'.format('TAG', args.tag))

    model = GraphDetector(cfg.head.input_channels, cfg.head.model_cfg, cfg.head.type)

    split = 'val'
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

    ckpt_file = ckpt_dir / ('checkpoint_epoch_%d.pth' % args.ckpt_epoch)
    
    
    if args.ckpt_specified_path is not None:
        ckpt_file = args.ckpt_specified_path
    
    logger.info("loading ckpt_file: {}".format(ckpt_file))

    logger.info(model)
    if args.hadflow:
        ckpt_file = args.tag
    model.load_params_from_file(ckpt_file)
    logger.info("loaded ckpt_file: {}".format(ckpt_file))


    model.cuda()
    model.eval()
    logger.info('*************** EVALUATION *****************')

    with torch.no_grad():
        det_annos = list()
        det_annos_kitti = list()
        total_time = 0
        total_times = 0
        for i, batch_dict in enumerate(tqdm(dataloader)):
            load_data_to_gpu(batch_dict)
            t0 = time()
            try:
                results = model(batch_dict, return_loss=False)
            except RuntimeError:
                print("the following batch_dict raises RuntimeError", batch_dict)
                continue
            t1 = time() - t0
            total_time += t1
            total_times += 1
            #print(t1, total_time / total_times)
            if total_times == 1000:
                total_times = 0
                total_time = 0
            annos = generate_prediction_dicts(results)
            det_annos += annos
            det_annos_kitti += results

        # yes, kitti format!!!
        with open("./cp++_intrarm_val_kittiformat.pkl", "wb") as f:
             pickle.dump(det_annos_kitti, f)

        infos = load_waymo_pickle_data(data_dir=data_path, split=split, logger=logger, pic_data=dataset._waymo_infos)

        infos = infos[:len(det_annos)]
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
        #evaluation_utils.write_evaluation_result()


if __name__ == "__main__":
    main()
