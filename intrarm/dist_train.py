import argparse
import os
import os.path as osp
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

from centergraph.utils.scheduler import create_iter_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import centergraph.utils.miscs as utils 

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("config", help="config file path")
    parser.add_argument('--tag', type=str, default=None, required=True, help='experiment tag')

    parser.add_argument('--data_path', type=str, default='/mnt/wyh/Dataset/WOD_CenterPoint',
                        help='specify the data location')
    parser.add_argument('--root_dir', type=str, default='/mnt/wyh/Codebase/pgnet',
                        help='specify the data location')
    
    parser.add_argument('--fp32-resume', action='store_true', default=True)
    parser.add_argument('--batch-size', default=16, type=int)

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--use-sched', default=1, type=int, help='whether use lr scheduler')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=3e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-iters', type=float, default=30000, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-iters', type=int, default=1000, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-iters', type=int, default=1000, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-iters', type=int, default=1000, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--radius_t', type=float, default=2.0, help='radius of constructing graph')
    parser.add_argument('--use_knn', type=int, default=0, help='use knn')
    parser.add_argument('--num_updates', type=int, default=4, help='number of gnns in IntraRM')

    parser.add_argument('--hadflow', action='store_true', default=False, help='whether to train on AutoDrive')
    parser.add_argument('--hadflow_step', type=int, default=1, required=False, help='checkpoint evaluation step')
    parser.add_argument('--pcdet2det3d', type=int, default=1, required=False)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def train_one_epoch(model, optimizer, train_loader, lr_scheduler, accumulated_iter, loss_scaler, tb_logger=None, logger=None, fp32=False, total_steps=10000, print_freq=25):    
    print("start to train!")
    for cur_it, batch in enumerate(train_loader):
        start = time()

        optimizer.zero_grad()
        load_data_to_gpu(batch)
        
        with torch.cuda.amp.autocast(enabled=not fp32):
            loss, tb_dict = model(batch, return_loss=True)
        
        # TODO include in config
        
        if loss_scaler is not None:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=10,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        
        if lr_scheduler is not None:
            lr_scheduler.step(accumulated_iter)

        accumulated_iter += 1
        disp_dict = {'loss': loss.item()}


        if cur_it % print_freq == 0:
            if logger is not None:
                logger.info("iter: [{}]/[{}], lr: {:.6f}, loss: {:.2f}, rcnn_loss_dict: {}, cls_loss: {:.2f}, time: {:.2f}".format(accumulated_iter-1, total_steps, optimizer.param_groups[0]["lr"], loss.item(), 
                tb_dict['rcnn_loss_dict'], tb_dict['rcnn_loss_cls'], time() - start))
            else:
                print("iter: [{}]/[{}], lr: {:.6f}, loss: {:.2f}, rcnn_loss_dict: {}, cls_loss: {:.2f}, time: {:.2f}".format(accumulated_iter-1, total_steps, optimizer.param_groups[0]["lr"], loss.item(), 
                tb_dict['rcnn_loss_dict'], tb_dict['rcnn_loss_cls'], time() - start))

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
    cfg.head.model_cfg.USE_KNN = args.use_knn
    cfg.head.model_cfg.NUM_UPDATES = args.num_updates

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    merge_config_from_args(cfg, args)

    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    
    torch.backends.cudnn.benchmark = False

    root_dir = Path(args.root_dir)
    output_dir = root_dir / 'output/' / args.tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)

    shutil.copy(args.config, output_dir / osp.basename(args.config))

    #cfg.dump(osp.join(output_dir, osp.basename(args.config)))

    log_file = output_dir / ('log_train_%s.txt' % args.tag)
    if args.hadflow:
        log_file = '/summary/log_train_%s.txt' % args.tag

    logger = create_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info('**********************Start logging**********************')
    logger.info('{:16} {}'.format('LEARNING_RATE', args.lr))
    logger.info(args)

    if args.hadflow:
        tb_logger = SummaryWriter(log_dir=str('/summary/'))
    else:
        tb_logger = None #SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    model = GraphDetector(cfg.head.input_channels, cfg.head.model_cfg, cfg.head.type)
    fix_weight_path = cfg.fix_weight_path
    if fix_weight_path is not None and os.path.exists(fix_weight_path):
        logger.info("load fix weight from {}".format(fix_weight_path))
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

    if args.distributed:
        num_tasks = utils.get_world_size()
        print("num_of_node:", num_tasks)
        global_rank = utils.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank, shuffle=True
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False,
                            num_workers=cfg.data.train.num_workers, sampler=sampler, drop_last=True)

    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.total_epochs * len(dataloader)
    args.iters = total_steps

    #lr_scheduler = _create_learning_rate_scheduler(
    #    optimizer, cfg.lr_config, total_steps
    #)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    if args.use_sched:
        lr_scheduler, _ = create_iter_scheduler(args, optimizer)
    else:
        lr_scheduler = None

    model.train()
    print(model)

    logger.info('********************** START TRAINING **********************')
    accumulated_iter = 0
    for epoch in range(cfg.total_epochs):
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)
        
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        
        
        accumulated_iter = train_one_epoch(model, optimizer, dataloader, lr_scheduler, loss_scaler=loss_scaler, accumulated_iter=accumulated_iter, total_steps=total_steps, logger=logger, fp32=args.fp32_resume, tb_logger=tb_logger)
        ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % (epoch + 1))
        logger.info("saving epoch {} ckpt to {}".format(epoch+1, ckpt_name))
        utils.save_on_master(model_without_ddp.state_dict(), '{}.pth'.format(ckpt_name))
        if args.hadflow and epoch % args.hadflow_step == 0:
            filename = '{}.pth'.format(ckpt_name)
            shutil.copy(filename, '/model')
            with open("/result/iterations", "a") as step_writer:
                automl_step = dict()
                automl_step["step"] = epoch + 1
                automl_step["weights"] = os.path.basename(filename)
                step_writer.write(";".join(["{}:{}".format(k, v) for k, v in automl_step.items()]) + "\n")
                step_writer.flush()

    logger.info('********************** END TRAINING **********************\n\n\n')


if __name__ == "__main__":
    main()
