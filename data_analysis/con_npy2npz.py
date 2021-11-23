import pdb
import sys
import os

import argparse
import numpy as np
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    # parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    # parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    # parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    # parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    # parser.add_argument('--pca', action='store_true', default=False, help='Whether to use PCA to rotate data [default: False]')
    # parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
    # parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    return parser.parse_args()


def main(args):
    pc     = np.load( os.path.join(args.log_dir, 'fgsm_pert.npy') )
    pc_ori = np.load( os.path.join(args.log_dir, 'orig.npy') )
    lb     = np.load( os.path.join(args.log_dir, 'lable.npy') )

    np.savez(os.path.join(args.log_dir, 'model_jgba.npz'),
             test_pc=pc.astype(np.float32),
             test_label=lb.astype(np.uint8),
             target_label=lb.astype(np.uint8),
             ori_pc=pc_ori.astype(np.float32))

if __name__ == '__main__':
    args = parse_args()
    main(args)
