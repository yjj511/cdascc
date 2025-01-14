# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
import os
import pprint
import _init_paths
from lib.utils.utils import create_logger, random_seed_setting
from cc_function import test_cc
import datasets
import torch
import numpy as np
import timeit
import logging
import argparse
from build_counter import Baseline_Counter
from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='Test crowd counting network')

    parser.add_argument('--testset', type=str, default='SHA', help="the test datasets")
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = Config.fromfile(args.cfg)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    random_seed_setting(config)

    # build model
    device = torch.device('cuda')

    model = Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size, device)

    model_state_file = args.checkpoint

    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)

    model.load_state_dict(pretrained_dict, strict=False)

    model = model.to(device)

    config_dict = {
        "SHHA": 'configs/SHA_final.py',
        "SHHB": 'configs/SHB_final.py',
        "QNRF": 'configs/QNRF_final.py',
        "JHU": 'configs/JHU_final.py',
        "NWPU": 'configs/NWPU_final.py',
    }
    target_dataset = args.testset.strip()
    target_dataset_config = Config.fromfile(config_dict[target_dataset])

    # prepare data
    test_dataset = eval('datasets.' + config.dataset.name)(
        root=target_dataset_config.dataset.root,
        list_path=target_dataset_config.dataset.test_set,
        num_samples=None,
        multi_scale=False,
        flip=False,
        base_size=target_dataset_config.test.base_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,  # config.WORKERS,
        pin_memory=True)

    start = timeit.default_timer()
    if 'test' in config.dataset.test_set or 'val' in config.dataset.test_set:
        mae, mse, nae, save_count_txt = test_cc(config, test_dataset, testloader, model,
                                                test_dataset.mean, test_dataset.std,
                                                sv_dir=final_output_dir, sv_pred=False, logger=logger
                                                )

        msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
               nae: {: 4.4f}, Class IoU: '.format(mae,
                                                  mse, nae)
        logging.info(msg)

    with open(final_output_dir + '_submit_SIC.txt', 'w') as f:
        f.write(save_count_txt)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int32((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
