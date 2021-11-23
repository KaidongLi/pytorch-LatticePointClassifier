import os

import pdb

import argparse
import numpy as np
import torch
from pytorch3d.loss import chamfer

PEND_ORIG = 'orig'
# PEND_NEW = 'splat'
PEND_IND_ORI = 'indpts_orig'
PEND_ATT = 'indpts'
PEND_ATT_FAIL = 'indpts_f'
PEND_ATT2D = '2dimg'
PEND_PRED = 'pred'

parser = argparse.ArgumentParser(
    description='test shape net show image')
parser.add_argument('--img_dir', default='../log', type=str)
# parser.add_argument('--check_success', action='store_true', default=False, help='Whether to check success attack [default: False]')
# parser.add_argument('--plot', action='store_true', default=False, help='Whether to show image [default: False]')

args = parser.parse_args()

def evaluate():

    # pend_attack = PEND_ATT if args.check_success else PEND_ATT_FAIL
    norm_succ = []
    norm_fail = []

    ct_sample = {}
    ct_succ = {}
    l_cls = []

    all_ori_pc = []
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    all_succ = []

    for file in os.listdir(args.img_dir):
        if file.endswith(PEND_ORIG + '.npy') and not file.endswith(PEND_IND_ORI + '.npy'):
            pt_cld_ori = np.load( os.path.join(args.img_dir, file) )
            pt_cld_ind = np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_IND_ORI+'.npy') )
            pt_cld_atts= np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_ATT+'.npy') )
            pt_cld_attf= np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_ATT_FAIL+'.npy') )
            # pt_cld_a2d = np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_ATT2D+'.npy') )
            pt_cld_prd = np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_PRED+'.npy') )

            tgt = file.split('_')
            vic = tgt[0]
            tgt = tgt[1]

            vic = add_blank(vic, 2)
            tgt = add_blank(tgt, 2)

            if vic not in ct_succ:
                ct_succ[vic] = {vic:0}
                ct_sample[vic] = {vic:1}
                l_cls.append(vic)

            if tgt not in ct_succ[vic]:
                ct_succ[vic][tgt] = 0
                ct_sample[vic][tgt] = pt_cld_ori.shape[0]
            else:
                ct_sample[vic][tgt] += pt_cld_ori.shape[0]


            # print(pend_attack)

            # if pt_cld_att.max()==1 and pt_cld_att.min()==1:

            for i in range(pt_cld_ori.shape[0]):

                if int(file.split('_')[1]) == pt_cld_prd[i] \
                and (pt_cld_atts[i].max() - pt_cld_atts[i].min() > 0) :
                    print(file)
                    flg = True
                    norm_succ.append(1)
                    np_att_pc = np.concatenate((pt_cld_atts[i], pt_cld_ori[i]), axis=0)
                    # np_att_pc = np.concatenate(((pt_cld_ind[i] + pt_cld_atts[i]), pt_cld_ori[i]), axis=0)
                    all_adv_pc.append(np_att_pc[None])

                    norm_succ.append(
                            chamfer.chamfer_distance(torch.Tensor(pt_cld_atts[i][None]), torch.Tensor(pt_cld_ori[i][None]), batch_reduction=None)[0].mean()
                        )

                    ct_succ[vic][tgt] += 1
                elif (not int(file.split('_')[1]) == pt_cld_prd[i]) \
                and (pt_cld_attf[i].max() - pt_cld_attf[i].min() > 0) \
                and (pt_cld_atts[i].max()==1 and pt_cld_atts[i].min()==1) :
                    flg = False
                    norm_fail.append(1)
                    np_att_pc = np.concatenate((pt_cld_attf[i], pt_cld_ori[i]), axis=0)
                    # np_att_pc = np.concatenate(((pt_cld_ind[i] + pt_cld_attf[i]), pt_cld_ori[i]), axis=0)
                    all_adv_pc.append(np_att_pc[None])
                    
                    norm_fail.append(
                            chamfer.chamfer_distance(torch.Tensor(pt_cld_attf[i][None]), torch.Tensor(pt_cld_ori[i][None]), batch_reduction=None)[0].mean()
                        )
                else:
                    print('conflict!!!')
                    pdb.set_trace()

                all_ori_pc.append(pt_cld_ori[None, i])
                all_real_lbl.append(int(vic))
                all_target_lbl.append(int(tgt))
                all_succ.append(flg)

    print('attack succ rate: ', len(norm_succ)/(len(norm_succ)+len(norm_fail)) )
    print('suc norm: ', sum(norm_succ)/(len(norm_succ)+1e-5), ', f norm: ', sum(norm_fail)/(len(norm_fail)+1e-5))

    arr_succ = np.zeros((len(l_cls), len(l_cls)))
    for i in range(len(l_cls)+2):
        line_str = ''
        for j in range(len(l_cls) + 2):
            if i == 0:
                if j == 0:
                    line_str += '   '
                elif j == len(l_cls) + 1:
                    line_str += ' |    sm'
                else:
                    line_str += ' |     ' + l_cls[j-1]
            elif i == len(l_cls) + 1:
                if j == 0:
                    line_str += ' sm'
                elif j == len(l_cls) + 1:
                    continue
                else:
                    line_str += ' | ' + '%.4f'%(arr_succ[:, j-1].sum())
            else:
                if j == 0:
                    line_str += ' ' + l_cls[i-1]
                elif j == len(l_cls) + 1:
                    line_str += ' | ' + '%.4f'%(arr_succ[i-1, :].sum())
                else:
                    arr_succ[i-1, j-1] = ct_succ[l_cls[i-1]][l_cls[j-1]]/ct_sample[l_cls[i-1]][l_cls[j-1]]
                    line_str += ' | ' + '%.4f'%(arr_succ[i-1, j-1])
        print(line_str)

    print('Continue to save npz file!! Continue???.....c' )
    pdb.set_trace()
    print('Continue to save npz file!!' )

    all_ori_pc = np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.array(all_real_lbl)  # [num_data]
    all_target_lbl = np.array(all_target_lbl)  # [num_data]
    all_succ = np.array(all_succ)  # [num_data]

    np.savez(os.path.join(args.img_dir, 'pntnet_ind_6.npz'),
             test_pc=all_adv_pc.astype(np.float32),
             test_ori_pc=all_ori_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8),
             is_succ=all_succ.astype(np.bool))


def add_blank(str_mod, length=2):
    for i in range(2 - len(str_mod)):
        str_mod = ' '+str_mod
    return str_mod

if __name__=='__main__':
    evaluate()
