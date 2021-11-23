import os

import pdb

import argparse
import numpy as np

import torch
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from pytorch3d.loss import chamfer

parser = argparse.ArgumentParser(
    description='test shape net show image')
parser.add_argument('--attack_data', default='', type=str)
parser.add_argument('--defend_data', default='', type=str)

args = parser.parse_args()

def evaluate():

    assert os.path.exists(args.attack_data)

    npz = np.load(args.attack_data, allow_pickle=True)
    pc_ori = npz['test_ori_pc']
    pc_att = npz['test_pc']

    if 'is_succ' in npz:
        is_succ = npz['is_succ']
    else:
        is_succ = np.array([True]*pc_ori.shape[0]).astype(np.bool)

    assign_pts(pc_ori[is_succ], pc_att[is_succ])

    if os.path.exists(args.defend_data):
        print('now for defense data:')
        npz = np.load(args.defend_data, allow_pickle=True)
        pc_def = npz['test_pc']
        assign_pts(pc_ori, pc_def[:, :1024])


def assign_pts(pc_x, pc_y):
    b = pc_x.shape[0]
    n = pc_x.shape[1]
    m_cost = np.zeros((b, n, n))
    l_row_ind = np.zeros((b, n), dtype=int)
    l_col_ind = np.zeros((b, n), dtype=int)

    l_cost = np.zeros(b)

    for i in range(b):
        m_cost[i] = distance_matrix(pc_x[i], pc_y[i])
        l_row_ind[i], l_col_ind[i] = linear_sum_assignment(m_cost[i])
        l_cost[i] = m_cost[i, l_row_ind[i], l_col_ind[i]].mean()

    print('bipartite assignment norm: ', l_cost.mean())
    print('1on1 norm: ', m_cost.diagonal(axis1=1, axis2=2).mean())



    # calculate mean pert norm
    # m_cost.diagonal(axis1=1, axis2=2).mean()
    pdb.set_trace()


if __name__=='__main__':
    evaluate()
