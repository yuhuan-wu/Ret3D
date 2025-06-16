import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.optim as optim
from time import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

from centergraph.datasets.waymo import SingleSweepWaymoDataset, MultiSweepWaymoDataset
from centergraph.models.detectors import GraphDetector
from centergraph.torchie import Config
from centergraph.utils import create_logger, load_data_to_gpu

from centergraph.builder import _create_learning_rate_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("config", help="config file path")
    parser.add_argument('--tag', type=str, default=None, required=True, help='experiment tag')

    parser.add_argument('--data_path', type=str, default='/mnt/wyh/Dataset/WOD_CenterPoint',
                        help='specify the data location')
    parser.add_argument('--root_dir', type=str, default='/mnt/wyh/Codebase/pgnet',
                        help='specify the data location')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')

    parser.add_argument('--radius_t', type=float, default=2.0, help='radius of constructing graph')
    parser.add_argument('--num_updates', type=int, default=4, help='number of gnns in IntraRM')

    parser.add_argument('--hadflow', action='store_true', default=False, help='whether to train on AutoDrive')
    parser.add_argument('--hadflow_step', type=int, default=1, required=False, help='checkpoint evaluation step')
    parser.add_argument('--pcdet2det3d', type=int, default=1, required=False)

    args = parser.parse_args()
    return args

def train_one_epoch(model, optimizer, train_loader, lr_scheduler, accumulated_iter, tb_logger, logger=None, total_steps=10000, print_freq=25):    
    print("start to train!")
    for cur_it, batch in enumerate(train_loader):
        start = time()

        optimizer.zero_grad()
        load_data_to_gpu(batch)
        loss, tb_dict = model(batch, return_loss=True)
        loss.backward()
        # TODO include in config
        
        clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        lr_scheduler.step(accumulated_iter)

        accumulated_iter += 1
        disp_dict = {'loss': loss.item()}

        if cur_it % print_freq == 0:
            if logger is not None:
                logger.info("iter: [{}]/[{}], lr: {:.6f}, loss: {:.2f}, rcnn_loss_dict: {}, cls_loss: {:.2f}, time: {:.2f}".format(accumulated_iter, total_steps, optimizer.lr, loss.item(), 
                tb_dict['rcnn_loss_dict'], tb_dict['rcnn_loss_cls'], time() - start))
            else:
                print("iter: [{}]/[{}], lr: {:.6f}, loss: {:.2f}, rcnn_loss_dict: {}, cls_loss: {:.2f}, time: {:.2f}".format(accumulated_iter, total_steps, optimizer.lr, loss.item(), 
                tb_dict['rcnn_loss_dict'], tb_dict['rcnn_loss_cls'], time() - start))

        if tb_logger is not None:
            tb_logger.add_scalar('train/loss', loss, accumulated_iter)
            for key, val in tb_dict.items():
                tb_logger.add_scalar('train/' + key, val, accumulated_iter)

    return accumulated_iter



def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def merge_config_from_args(cfg, args):
    cfg.head.model_cfg.EDGE_RADIUS = args.radius_t
    cfg.radius_t = args.radius_t
    cfg.head.model_cfg.NUM_UPDATES = args.num_updates

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1
    merge_config_from_args(cfg, args)

    root_dir = Path(args.root_dir)
    output_dir = root_dir / 'output/' / args.tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    log_file = output_dir / ('log_train_%s.txt' % args.tag)
    if args.hadflow:
        log_file = '/summary/log_train_%s.txt' % args.tag

    logger = create_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('**********************Start logging**********************')
    logger.info('{:16} {}'.format('LEARNING_RATE', args.lr))

    if args.hadflow:
        tb_logger = SummaryWriter(log_dir=str('/summary/'))
    else:
        tb_logger = None #SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    model = GraphDetector(cfg.head.input_channels, cfg.head.model_cfg, cfg.head.type)
    fix_weight_path = cfg.fix_weight_path
    if fix_weight_path is not None and os.path.exists(fix_weight_path):
        model.load_params_from_file(fix_weight_path)
    
    split = 'train'
    
    if cfg.head.type == 'RoIHead':
        dataset = SingleSweepWaymoDataset(data_path, split, logger, 0.0,
                                          sample_interval=cfg.data.train.sampling_interval)
    elif cfg.head.type in ['GraphHead', 'MiXHead', 'DevHead'] and cfg.head.model_cfg.GRAPH_TYPE == 's':
        dataset = SingleSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                          sample_interval=cfg.data.train.sampling_interval)
    elif cfg.head.type in ['GraphHead', 'MiXHead', 'DevHead'] and cfg.head.model_cfg.GRAPH_TYPE == 't':
        dataset = MultiSweepWaymoDataset(data_path, split, logger, cfg.radius_t,
                                         sample_interval=cfg.data.train.sampling_interval, pcdet2det3d=args.pcdet2det3d)
    else:
        logger.info('NO IMPLEMENTED DATASET TYPE!')
        exit()

    dataloader = DataLoader(dataset, batch_size=cfg.data.train.batch_size, pin_memory=False,
                            num_workers=cfg.data.train.num_workers, shuffle=True, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.total_epochs * len(dataloader)

    lr_scheduler = _create_learning_rate_scheduler(
        optimizer, cfg.lr_config, total_steps
    )

    model.cuda()
    model.train()

    logger.info('********************** START TRAINING **********************')
    accumulated_iter = 0
    for epoch in range(cfg.total_epochs):

        accumulated_iter = train_one_epoch(model, optimizer, dataloader, lr_scheduler, accumulated_iter=accumulated_iter, total_steps=total_steps, logger=logger, tb_logger=tb_logger)
        ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % (epoch + 1))
        logger.info("saving epoch {} ckpt to {}".format(epoch+1, ckpt_name))
        save_checkpoint(
            model.state_dict(), filename=ckpt_name,
        )
        if args.hadflow and epoch % args.hadflow_step == 0:
            filename = '{}.pth'.format(ckpt_name)
            shutil.copy(filename, '/model')
            with open("/result/iterations", "a") as step_writer:
                automl_step = dict()
                automl_step["step"] = trained_epoch + 1
                automl_step["weights"] = os.path.basename(filename)
                step_writer.write(";".join(["{}:{}".format(k, v) for k, v in automl_step.items()]) + "\n")
                step_writer.flush()

    logger.info('********************** END TRAINING **********************\n\n\n')


if __name__ == "__main__":
    main()
