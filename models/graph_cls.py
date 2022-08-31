'''
Author: Kaidong Li
Purpose: Graph Drawing Classifier
Only use batch size 1
'''
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

import math
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance  import cdist,pdist,squareform

# from pointnet import STN3d #, PointNetEncoder #, feature_transform_reguliarzer
from resnet import resnet50



class get_model(nn.Module):

    # s is the final image scale
    def __init__(self, k=40, normal_channel=True, backbone=resnet50(40)):
        super(get_model, self).__init__()

        self.size2d = 256
        self.num_pc = 2048
        self.graph_drawing = GraphDraw(self.num_pc, self.size2d, normal_channel)
        self.network_2d = backbone

        self.normal_channel = normal_channel


    def forward(self, x):
        img_2d, _ = self.graph_drawing(x)

        # network takes [b, c, size, size]
        outputs = self.network_2d(img_2d)

        return outputs, [_[0]]



class get_adv_loss(torch.nn.Module):
    def __init__(self, num_class, mat_diff_loss_scale=0.001):
        super(get_adv_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.num_class = num_class

    def forward(self, pred, target, kappa=0):
        tlab = F.one_hot(target.squeeze(), self.num_class)

        # real: target class value
        real = torch.sum(tlab * pred, dim=1)
        # other: max value of the rest
        other = torch.max((1 - tlab) * pred - (tlab * 10000), dim=1)[0]

        loss1 = torch.maximum(torch.Tensor([0.]).cuda(), other - real + kappa)
        return torch.mean(loss1)


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss



class GraphDraw(nn.Module):
    def __init__(self, num_pts, s, normal_channel):
        super(GraphDraw, self).__init__()
        self.num_pts = num_pts
        self.num_cuts = 32
        self.size_sub = 16
        self.size_top = 16
        self.size2d = s
        # self.normal_channel = normal_channel
        
        self.kmeans = KMeans(n_clusters=self.num_cuts, n_init=1,max_iter=100)


    def fun_GPGL_layout_push(self, pos,size):
        dist_mat = pdist(pos)
        scale1 = 1/dist_mat.min()
        scale2 = (size-2)/(pos.max()-pos.min())
        scale = np.min([scale1,scale2])
        pos = pos*scale

        pos_quat = np.round(pos).astype(np.int)
        pos_quat = pos_quat-np.min(pos_quat,axis=0)+[1,1]
        pos_unique, count = np.unique(pos_quat,axis=0,return_counts=True)
        # node_loss = np.sum(count)-len(count)
        # print('node_loss',node_loss)

        mask = np.zeros([size,size]).astype(np.int)
        for pt in pos_quat:
            mask[pt[0],pt[1]]+=1

        for i_loop in range(50):
            if(mask.max()<=1):
                # print("early stop")
                break
            idxs = np.where(count>1)[0]
            for idx in idxs:
                pos_overlap = pos_unique[idx]
                dist = cdist(pos_quat,[pos_overlap])
                idy = np.argmin(dist)

                b_down = np.maximum(pos_overlap[0]-1,0)
                b_up   = np.minimum(pos_overlap[0]+2,size)
                b_left = np.maximum(pos_overlap[1]-1,0)
                b_right= np.minimum(pos_overlap[1]+2,size)

                mask_target = mask[b_down:b_up,b_left:b_right]
                if(mask_target.min()==0):
                    pos_target = np.unravel_index(np.argmin(mask_target),mask_target.shape)
                    pos_mask = pos_target+np.array([b_down,b_left])

                else:
                    pos_empty = np.array(np.where(mask==0)).T
                    dist = cdist(pos_empty,[pos_overlap])
                    pos_target  = pos_empty[np.argmin(dist)]
                    direction = (pos_target-pos_overlap)
                    direction1 = np.round(direction/np.linalg.norm(direction))
                    pos_mask = pos_overlap+direction1.astype(np.int)


                pos_quat[idy]=pos_mask
                mask[pos_overlap[0],pos_overlap[1]] -=1
                mask[pos_mask[0],pos_mask[1]] +=1

                pos_unique, count = np.unique(pos_quat,axis=0,return_counts=True)
        return pos_quat


    def graph_cut(self, data,dist_mat,NUM_POINTS):
        NUM_CUTPOINTS = int(NUM_POINTS/self.num_cuts)
        CUTPOINTS_THRESHOLD = np.ceil(NUM_CUTPOINTS*1.2)
        clsuter = np.argmin(dist_mat,axis=-1)
        mask = np.zeros([NUM_POINTS,self.num_cuts])
        for m, c in zip(mask,clsuter):
            m[c]=1
        loss_mask = mask.sum(0)

        flow_mat = np.zeros([self.num_cuts,self.num_cuts])

        ## %% separate point cloud into self.num_cuts clusters
        for i_loop in range(500):
            loss_mask = mask.sum(0)
            order_list = np.argsort(loss_mask)
            if(loss_mask.max()<=CUTPOINTS_THRESHOLD+1):
                break
            for i_order,order in zip(range(len(order_list)),order_list):
                if(loss_mask[order]>CUTPOINTS_THRESHOLD):
                    idxs = np.where(mask[:,order])[0]
                    idys_ori = order_list[:i_order]
                    idys = []
                    for idy in idys_ori:
                        if(flow_mat[order,idy]>=0):
                            idys.append(idy)

                    mat_new = dist_mat[idxs,:]
                    mat_new = mat_new[:,idys]
                    cost_list_row = mat_new.argmin(-1)
                    cost_list_col= mat_new.min(-1)

                    row = cost_list_col.argmin(-1)
                    col = cost_list_row[row]

                    target_idx = [idxs[row],idys[col]]
                    mask[target_idx[0],order]=0
                    mask[target_idx[0],target_idx[1]]=1
                    flow_mat[order,target_idx[1]]=1
                    flow_mat[target_idx[1],order]=-1
        center_pos = []
        for i_cut in range(self.num_cuts):
            if mask[:,i_cut].sum()>0:
                center_pos.append(data[mask[:,i_cut].astype(np.bool),:].mean(0))
            else:
                center_pos.append([0,0])
        labels = mask.argmax(-1)
        return np.array(center_pos),labels



    #%% 3D to 2D projection function
    def GPGL2_seg(self, data): # ,current_sample_seg):
        data = data+np.random.rand(len(data),len(data[0]))*1e-6
        dist_mat = self.kmeans.fit_transform(data)

        node_top,labels = self.graph_cut(data,dist_mat,self.num_pts)

        aij_mat = squareform(pdist(node_top),checks=False)
        H = nx.from_numpy_matrix(aij_mat)
        pos_spring = nx.spring_layout(H)
        pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])

        pos = self.fun_GPGL_layout_push(pos_spring,self.size_sub)
        pos_top = self.fun_GPGL_layout_push(pos_spring,self.size_top)

        ##%%
        pos_cuts = []
        for i_cut in range(self.num_cuts):
            pos_cut_3D = data[labels==i_cut,:]

            if(len(pos_cut_3D)<5):
                pos_raw = [[0,0],[0,1],[1,1],[1,0]]
                pos = pos_raw[:len(pos_cut_3D)]
                pos_cuts.append(pos)
                continue

            aij_mat = squareform(pdist(pos_cut_3D),checks=False)
            H = nx.from_numpy_matrix(aij_mat)
            pos_spring = nx.spring_layout(H)
            pos_spring = np.array([pos for idx,pos in sorted(pos_spring.items())])
            pos = self.fun_GPGL_layout_push(pos_spring,self.size_sub)

            pos_cuts.append(pos)

        ##%% combine all layout positions
        cuts_count = np.zeros(self.num_cuts).astype(np.int64)
        pos_all = []
        for idx in range(self.num_pts):
            label = labels[idx]
            pos_all.append(pos_cuts[label][cuts_count[label]]+pos_top[label]*self.size_sub)
            cuts_count[label] +=1
        pos_all=np.array(pos_all)

        num_nodes_m = len(np.unique(pos_all,axis=0))
        node_loss_rate=(1- num_nodes_m/self.num_pts)
        return pos_all, node_loss_rate


    def forward(self, pc1):
        assert pc1.shape[0] == 1, 'Batch size must be one'
        points = pc1.permute(0, 2, 1)
        one_np_pc = points.cpu().data.numpy()[0]
        pos, node_loss_rate = self.GPGL2_seg(one_np_pc)

        img_2d = torch.zeros((1, self.size2d, self.size2d, 3), dtype=torch.float32).cuda()
        img_2d[0, pos[:,0], pos[:,1]] = points[0]
        # np_im = np.zeros([SIZE_IMG, SIZE_IMG, 3], dtype=np.float32)
        # np_im[pos[:,0], pos[:,1]] = one_np_pc
        # np_im = np_im[None]

        img_2d = img_2d.permute(0, 3, 1, 2).contiguous()
        
        return img_2d, [pos]