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


def _create_pd_detection(detections, submission_annos_dict, tracking=False):
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    LABEL_TO_TYPE = {0: 1, 1: 2, 2: 4}

    objects = metrics_pb2.Objects()
    for i, detection in enumerate(tqdm(detections)):
        box3d = detection["box3d_lidar"].detach().cpu().numpy()
        scores = detection["scores"].detach().cpu().numpy()
        labels = detection["label_preds"].detach().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]
        frame_id = detection['metadata'].split('/')[-1][:-4]

        for i in range(box3d.shape[0]):
            det = box3d[i]
            score = scores[i]
            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = submission_annos_dict[frame_id]['scene_name']
            o.frame_timestamp_micros = submission_annos_dict[frame_id]['timestamp']

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label]

            objects.objects.append(o)
    return objects


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

        objects = _create_pd_detection(det_results, submission_annos_dict)
        out_file = 'detection_pred_%s.bin' % split
        out_path = output_dir / out_file
        logger.info("results saved to {}".format(out_path))
        f = open(out_path, 'wb')
        f.write(objects.SerializeToString())
        f.close()
        # with open("./prediction_nms_new.pkl", "wb") as f:
        #     pickle.dump(det_annos, f)

        # infos = load_waymo_pickle_data(data_dir=data_path, split=split, logger=logger, pic_data=dataset._waymo_infos)
        # start_time = time.time()
        # result_str, _ = evaluation(det_annos, infos=infos)
        # sec_eval = time.time() - start_time
        # logger.info('Dataset evaluation finished: %.1f seconds).' % sec_eval)
        # logger.info(result_str)


if __name__ == "__main__":
    main()
