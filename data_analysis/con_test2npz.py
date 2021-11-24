import pdb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.PCAModelNetDataLoader import PCAModelNetDataLoader
from data_utils.ScanNetDataLoader import ScanNetDataLoader
import argparse
import numpy as np
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import provider
import importlib
import shutil

from utils import get_backbone #, get_cifar_training, get_cifar_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    # parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    # parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    # parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--pca', action='store_true', default=False, help='Whether to use PCA to rotate data [default: False]')
    # parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
    parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    return parser.parse_args()

def test(loader, num_class=40):

    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        all_adv_pc.append(points.data.numpy())
        all_real_lbl.append(target.squeeze(dim=1).data.numpy())

    all_adv_pc = np.concatenate(all_adv_pc, axis=0)
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)
    # pdb.set_trace()

    return all_adv_pc, all_real_lbl



def main(args):

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    dump_dir = Path('./dump/')
    dump_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()



    if args.dataset == 'ModelNet40':
        DATA_PATH = 'data/modelnet40_normal_resampled/'

        if args.pca:
            TEST_DATASET = PCAModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)
        else:
            TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)
    elif args.dataset == 'ScanNetCls':
        assert (not args.pca), 'ScanNetCls with PCA is not supported yet'

        # TEST_PATH  = 'dump/scannet_test_data8316.npz'
        TEST_PATH  = 'data/scannet/test_files.txt'
        TEST_DATASET = ScanNetDataLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)



    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)


    num_class = 40

    with torch.no_grad():
        pc, lb = test(testDataLoader, num_class)
        np.savez(os.path.join(dump_dir, 'orig_dataset_%s.npz' % (args.dataset)),
                 test_pc=pc.astype(np.float32),
                 test_label=lb.astype(np.uint8),
                 target_label=lb.astype(np.uint8))

if __name__ == '__main__':
    args = parse_args()
    main(args)
