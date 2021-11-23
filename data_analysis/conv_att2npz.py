import os

import pdb

import argparse
import numpy as np

PEND_ORIG = 'orig'
# PEND_NEW = 'splat'
PEND_ATT = 'adv'
PEND_ATT_FAIL = 'adv_f'
PEND_ATT2D = '2dimg'
PEND_PRED = 'pred'

parser = argparse.ArgumentParser(
    description='test shape net show image')
parser.add_argument('--img_dir', default='../log', type=str)
# parser.add_argument('--check_success', action='store_true', default=False, help='Whether to check success attack [default: False]')
# parser.add_argument('--plot', action='store_true', default=False, help='Whether to show image [default: False]')

args = parser.parse_args()

def conv_att2npz():

    # pend_attack = PEND_ATT if args.check_success else PEND_ATT_FAIL
    # norm_succ = []
    # norm_fail = []

    # ct_sample = {}
    # ct_succ = {}
    # l_cls = []

    all_ori_pc = []
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    all_succ = []

    for file in os.listdir(args.img_dir):
        if file.endswith(PEND_ORIG + '.npy'):
            pt_cld_ori = np.load( os.path.join(args.img_dir, file) )
            pt_cld_atts= np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_ATT+'.npy') )
            pt_cld_attf= np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_ATT_FAIL+'.npy') )
            # pt_cld_a2d = np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_ATT2D+'.npy') )
            pt_cld_prd = np.load( os.path.join(args.img_dir, file.split(PEND_ORIG)[0]+PEND_PRED+'.npy') )

            print(file)

            tgt = file.split('_')
            vic = int(tgt[0])
            tgt = int(tgt[1])


            for i in range(pt_cld_ori.shape[0]):

                if tgt == pt_cld_prd[i] \
                and (pt_cld_atts[i].max() - pt_cld_atts[i].min() > 0) :
                    flg = True
                    all_adv_pc.append(pt_cld_atts[None, i])
                elif (not tgt == pt_cld_prd[i]) \
                and (pt_cld_attf[i].max() - pt_cld_attf[i].min() > 0) \
                and (pt_cld_atts[i].max()==1 and pt_cld_atts[i].min()==1) :
                    flg = False
                    all_adv_pc.append(pt_cld_attf[None, i])
                else:
                    print('conflict!!!')
                    pdb.set_trace()

                all_ori_pc.append(pt_cld_ori[None, i])
                all_real_lbl.append(vic)
                all_target_lbl.append(tgt)
                all_succ.append(flg)


    # pdb.set_trace()
    all_ori_pc = np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.array(all_real_lbl)  # [num_data]
    all_target_lbl = np.array(all_target_lbl)  # [num_data]
    all_succ = np.array(all_succ)  # [num_data]
    pdb.set_trace()


    np.savez(os.path.join(args.img_dir, 'pntnet_pert_6.npz'),
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
    conv_att2npz()
