import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader

from centergraph.datasets.waymo import SingleSweepWaymoDataset, MultiSweepWaymoDataset
from centergraph.models.detectors import GraphDetector
from centergraph.torchie import Config
from centergraph.utils import create_logger, load_data_to_gpu


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("config", help="config file path")
    parser.add_argument('--tag', type=str, default=None, required=True, help='experiment tag')

    parser.add_argument('--data_path', type=str, default='/mnt/wyh/Dataset/WOD_CenterPoint',
                        help='specify the data location')
    parser.add_argument('--root_dir', type=str, default='/mnt/wyh/Codebase/pgnet',
                        help='specify the data location')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')

    parser.add_argument('--hadflow', action='store_true', default=False, help='whether to train on AutoDrive')
    parser.add_argument('--hadflow_step', type=int, default=1, required=False, help='checkpoint evaluation step')

    args = parser.parse_args()
    return args


def train_one_epoch(model, optimizer, train_loader, total_it_each_epoch, leave_pbar, accumulated_iter, dataloader_iter,
                    tbar, tb_logger):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        optimizer.zero_grad()
        load_data_to_gpu(batch)
        loss, tb_dict = model(batch, return_loss=True)
        loss.backward()
        # TODO include in config
        clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        accumulated_iter += 1
        disp_dict = {'loss': loss.item()}

        pbar.update()
        pbar.set_postfix(dict(total_it=accumulated_iter))
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        if tb_logger is not None:
            tb_logger.add_scalar('train/loss', loss, accumulated_iter)
            for key, val in tb_dict.items():
                tb_logger.add_scalar('train/' + key, val, accumulated_iter)

    pbar.close()
    return accumulated_iter


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1

    #assert cfg.head.model_cfg.GRAPH_TYPE == 't'
    #assert cfg.head.model_cfg.REDUCE_NODE is True

    root_dir = Path(args.root_dir)
    output_dir = root_dir / 'output' / 'centergraph' / args.tag
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
        tb_logger = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))

    model = GraphDetector(cfg.head.input_channels, cfg.head.model_cfg, cfg.head.type)
    fix_weight_path = cfg.fix_weight_path
    model.load_params_from_file(fix_weight_path)

    # create train+val for training START
    ds_train = MultiSweepWaymoDataset(data_path, 'train', logger, cfg.radius_t, sample_interval=1)
    ds_val_w_anno = MultiSweepWaymoDataset(data_path, 'val_w_anno', logger, cfg.radius_t, sample_interval=1, pose_d=[])
    ds_val = MultiSweepWaymoDataset(data_path, 'val', logger, cfg.radius_t, sample_interval=1)

    pose_d = dict()
    pose_d.update(ds_train.pose_d)
    pose_d.update(ds_val.pose_d)

    dataset = MultiSweepWaymoDataset(data_path, 'train', logger, cfg.radius_t, sample_interval=1, pose_d=pose_d)
    rm_list = ds_val_w_anno._waymo_infos[::10]
    remain_list = ds_val_w_anno._waymo_infos
    for item in rm_list:
        remain_list.remove(item)
    dataset._waymo_infos = ds_train._waymo_infos + remain_list
    # create train+val for training END

    dataloader = DataLoader(dataset, batch_size=cfg.data.train.batch_size, pin_memory=False,
                            num_workers=cfg.data.train.num_workers, shuffle=True, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    model.cuda()
    model.train()

    logger.info('********************** START TRAINING **********************')
    accumulated_iter = 0
    with tqdm.trange(0, cfg.total_epochs, desc='epochs', dynamic_ncols=True, leave=True) as tbar:
        total_it_each_epoch = len(dataloader)
        dataloader_iter = iter(dataloader)

        for cur_epoch in tbar:
            accumulated_iter = train_one_epoch(model, optimizer, dataloader, total_it_each_epoch,
                                               leave_pbar=(cur_epoch + 1 == cfg.total_epochs),
                                               accumulated_iter=accumulated_iter, dataloader_iter=dataloader_iter,
                                               tbar=tbar, tb_logger=tb_logger)
            trained_epoch = cur_epoch + 1
            ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % trained_epoch)
            save_checkpoint(
                model.state_dict(), filename=ckpt_name,
            )
            if args.hadflow and trained_epoch % args.hadflow_step == 0:
                filename = '{}.pth'.format(ckpt_name)
                shutil.copy(filename, '/model')
                with open("/result/iterations", "a") as step_writer:
                    automl_step = dict()
                    automl_step["step"] = trained_epoch
                    automl_step["weights"] = os.path.basename(filename)
                    step_writer.write(";".join(["{}:{}".format(k, v) for k, v in automl_step.items()]) + "\n")
                    step_writer.flush()

    logger.info('********************** END TRAINING **********************\n\n\n')


if __name__ == "__main__":
    main()
