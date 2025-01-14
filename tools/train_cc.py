# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import json
import os
import time

import _init_paths
import pprint
import logging
import timeit
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from mmcv import Config, DictAction
import lib.datasets as datasets
from cc_function import train, validate
from lib.utils.KPI_pool import Task_KPI_Pool
from lib.utils.utils import create_logger, random_seed_setting, copy_cur_env
from bisect import bisect_right
from lib.datasets.utils.collate import default_collate
from tools.build_counter import Baseline_Counter
import argparse
from lib.solver.build import build_optimizer_cls
from lib.solver.lr_scheduler_cls import build_scheduler
from tools.data import ForeverDataIterator


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Crowd Counting network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # dg setting
    parser.add_argument('--testset', type=str, default='SHA,SHB,QNRF', help="the test datasets")
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')

    args = parser.parse_args()


    return args


def main():
    args = parse_args()

    config = Config.fromfile(args.cfg)
    # cudnn related and random seed setting
    random_seed_setting(config)


    logger, train_log_dir = create_logger(
        config, args.cfg, 'train')

    writer_dict = {
        'writer': SummaryWriter(train_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    device = torch.device('cuda')

    # build model

    model = Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size, device)

    # provide the summary of model
    logger.info(pprint.pformat(args))
    logger.info(config)

    optimizer = build_optimizer_cls(config.optimizer, model)

    model.cuda()

    last_epoch = 0

    if config.train.resume_path is not None:
        model_state_file = os.path.join(config.train.resume_path,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    # prepare data
    source_train_dataset = eval('datasets.' + config.dataset.name)(
        root=config.dataset.root,
        list_path=config.dataset.train_set,
        num_samples=None,
        num_classes=config.dataset.num_classes,
        multi_scale=config.train.multi_scale,
        flip=config.train.flip,
        ignore_label=None,
        base_size=config.train.base_size,
        crop_size=config.train.image_size,
        min_unit=config.train.route_size,
        scale_factor=config.train.scale_factor)

    source_trainloader = torch.utils.data.DataLoader(
        source_train_dataset,
        batch_size=config.train.batch_size_per_gpu,
        shuffle=config.train.shuffle,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=default_collate)

    # 测试集
    config_dict = {
        "SHHA": 'configs/SHA_final.py',
        "SHHB": 'configs/SHB_final.py',
        "QNRF": 'configs/QNRF_final.py',
        "JHU": 'configs/JHU_final.py',
        "NWPU": 'configs/NWPU_final.py',
    }
    target_dataset = args.testset.strip()
    target_dataset_config = Config.fromfile(config_dict[target_dataset])

    target_train_dataset = eval('datasets.' + target_dataset_config.dataset.name)(
        root=target_dataset_config.dataset.root,
        list_path=target_dataset_config.dataset.train_set,
        num_samples=None,
        num_classes=config.dataset.num_classes,
        multi_scale=config.train.multi_scale,
        flip=config.train.flip,
        ignore_label=None,
        base_size=config.train.base_size,
        crop_size=config.train.image_size,
        min_unit=config.train.route_size,
        scale_factor=config.train.scale_factor)

    target_trainloader = torch.utils.data.DataLoader(
        target_train_dataset,
        batch_size=config.train.batch_size_per_gpu,
        shuffle=config.train.shuffle,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=default_collate)
    # test_names.remove(config.dataset.name)

    train_source_iter = ForeverDataIterator(source_trainloader)
    train_target_iter = ForeverDataIterator(target_trainloader)

    test_dataset = eval('datasets.' + target_dataset_config.dataset.name)(
        root=target_dataset_config.dataset.root,
        list_path=target_dataset_config.dataset.test_set,
        num_samples=None,
        num_classes=target_dataset_config.dataset.num_classes,
        multi_scale=False,
        flip=False,
        base_size=target_dataset_config.test.base_size,
        crop_size=(None, None),
        min_unit=target_dataset_config.train.route_size,
        downsample_rate=1)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=target_dataset_config.test.batch_size_per_gpu,
        shuffle=False,
        num_workers=target_dataset_config.workers,
        pin_memory=True,
        collate_fn=default_collate)


    epoch_iters = len(source_trainloader)
    start = timeit.default_timer()
    end_epoch = config.train.end_epoch
    num_iters = epoch_iters

    if config.dataset.extra_train_set:
        epoch_iters = len(source_trainloader)
        extra_iters = config.train.extra_epoch * epoch_iters

    task_KPI = Task_KPI_Pool(
        task_setting={
            'x4': ['gt', 'error'],
            'x8': ['gt', 'error'],
            'x16': ['gt', 'error'],
            'x32': ['gt', 'error'],
            'acc1': ['gt', 'error']},
        maximum_sample=1000, device=device)
    scheduler = build_scheduler(config.lr_config, optimizer, epoch_iters, config.train.end_epoch)

    best_mae = 1e20
    best_mse = 1e20
    for epoch in range(last_epoch, end_epoch):
        train(config, epoch, config.train.end_epoch,
              epoch_iters, num_iters,
              train_source_iter, train_target_iter, optimizer, scheduler, model, writer_dict,
              device, train_log_dir + '/../', source_train_dataset.mean,
              source_train_dataset.std, task_KPI)
        if epoch % args.val_epoch == 0 and epoch >= args.val_start:

            logger.info('=> saving checkpoint to {}'.format(
                train_log_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(train_log_dir, 'checkpoint.pth.tar'))

            t1 = time.time()
            valid_loss, mae, mse, nae = validate(config,
                                                 test_loader, model, writer_dict, device,
                                                 train_log_dir + '/../val', test_dataset.mean,
                                                 test_dataset.std)
            t2 = time.time()

            if mae < best_mae:
                best_mae = mae
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        train_log_dir,
                        target_dataset + '_' +
                        str(epoch) +
                        '_mae_' +
                        f'{mae:.2f}' +
                        '_mse_' +
                        f'{mse:.2f}' +
                        '.pth'))
            if mse < best_mse:
                best_mse = mse
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        train_log_dir,
                        target_dataset + '_' +
                        str(epoch) +
                        '_mae_' +
                        f'{mae:.2f}' +
                        '_mse_' +
                        f'{mse:.2f}' +
                        '.pth'))
            msg = '{} time: {:.2f}, Loss: {:.3f}, MAE: {: 4.2f}, Best_MAE: {: 4.2f} ' \
                  'MSE: {: 4.2f},Best_MSE: {: 4.2f}'.format(
                target_dataset, t2 - t1, valid_loss, mae, best_mae, mse,
                best_mse)
            logging.info(msg)

            if epoch == end_epoch - 1:
                torch.save(model.state_dict(),
                           os.path.join(train_log_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int32((end - start) / 3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
