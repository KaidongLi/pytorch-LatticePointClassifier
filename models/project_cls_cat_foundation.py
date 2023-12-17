import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
# from pointnet import STN3d #, PointNetEncoder #, feature_transform_reguliarzer

import math

from resnet_foundation_backbone import ModifiedResNet
from ViT import VisionTransformer

from resnet import resnet50

# class myRound(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         out = torch.round(input).long()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         import pdb; pdb.set_trace()
#         print('back', grad_output)
#         grad_input = grad_output.clone()
#         return grad_input

# my_round = myRound.apply


class get_model(nn.Module):

    # s is the final image scale
    def __init__(self, k=40, normal_channel=True, backbone=resnet50(40), foundation_backbone='resnet-50-64', s=448):
        super(get_model, self).__init__()


        self.proj_trans = ProjGen(s, normal_channel)
        self.size2d = s
        self.network_2d = backbone
        ori_feat = 2048

        dim_feat = 768
        if foundation_backbone == 'vit-l14':
            # input image pixel: 224
            dim_feat = 768
            self.backbone_2d_foundation = VisionTransformer(
                                    input_resolution=s,
                                    patch_size=14,
                                    width=1024,
                                    layers=24,
                                    heads=16,
                                    output_dim=dim_feat)
            self.backbone_2d_foundation.load_state_dict(
                        torch.load('checkpoints/foundation_clip_ViT-L14.pth')#, map_location ='cpu')
                    )
        elif foundation_backbone == 'resnet-50-64':
            # input image pixel: 448
            dim_feat = 1024
            self.backbone_2d_foundation = ModifiedResNet(
                                    layers=[3, 15, 36, 10],
                                    output_dim=dim_feat,
                                    heads=64,
                                    input_resolution=s,
                                    width=128
                                )
            # import pdb; pdb.set_trace()
            self.backbone_2d_foundation.load_state_dict(
                        torch.load('checkpoints/foundation_clip_resnet50_64.pth')#, map_location ='cpu')
                    )

        for param in self.backbone_2d_foundation.parameters():
            param.requires_grad = False


        # self.fc = nn.Linear(dim_feat, k)
        self.fc =  nn.Sequential(
                        nn.Linear(ori_feat + dim_feat, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, 4096),
                        nn.ReLU(inplace=True),
                        nn.Dropout(),
                        nn.Linear(4096, k)
                    )


        self.normal_channel = normal_channel


    def forward(self, x):
        if self.normal_channel:
            vv = x[:, 3:]
            # vv = x[:, :3]
        else:
            vv = torch.ones((x.size(0), 1, x.size(2))).cuda()

        # returned splatted has a shape of [b, size, size, c]
        splatted_2d = self.proj_trans(x[:, :3] * (self.size2d//2 - 2), vv)
        # splatted_2d = torch.cat((splatted_2d, splatted_2d_2), 3).permute(0, 3, 1, 2).contiguous()
        splatted_2d = splatted_2d.permute(0, 3, 1, 2).contiguous()

        # splatted_2d = x
        import datetime
        st = datetime.datetime.now().timestamp()
        
        # import pdb; pdb.set_trace()

        _, outputs = self.network_2d(splatted_2d)
        # foudnation network takes [b, 3, size, size]
        if not self.normal_channel:
            splatted_2d = splatted_2d.repeat(1, 3, 1, 1)
        outputs_f = self.backbone_2d_foundation(splatted_2d)

        outputs = self.fc(torch.cat((outputs, outputs_f), 1))

        return outputs, [] #[_[0], _[1]]#[_[0].permute(0, 3, 1, 2), splatted_2d, _[1], st]



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
        # import pdb; pdb.set_trace()
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



class ProjGen(nn.Module):
    def __init__(self, s, normal_channel):
        super(ProjGen, self).__init__()

        d = 3
        self.d = d
        # self.d1 = self.d + 1


        self.d1 = self.d
        self.size2d = s
        self.normal_channel = normal_channel

    def convert2Dcoord(self, coord, batch_size, num_pts):
        offset = coord.min(dim=2)[0]
        coord -= offset.view(batch_size, -1, 1).expand(batch_size, -1, num_pts)
        return coord

    def get2D(self, coord, tmp, batch_size):
        # remove points that are out of range in 2d image
        idx_out_range = coord >= self.size2d
        c = 1

        coord[idx_out_range] = 0
        idx_out_range = idx_out_range.sum(1).nonzero()
        tmp[idx_out_range[:, 0], idx_out_range[:, 1]] = 0


        splatted_2d = torch.zeros((batch_size, self.size2d, self.size2d, c), dtype=torch.float32).cuda()
        # filter_2d = torch.zeros((batch_size, self.size2d//self.d1, self.size2d//self.d1, c), dtype=torch.float32).cuda()

        # if self.normal_channel:
        #     for i in range(batch_size):
        #         splatted_2d[i] = torch.cuda.sparse.FloatTensor(coord[i], tmp[i],
        #                               torch.Size([self.size2d, self.size2d, c])).to_dense()
        #         # import pdb; pdb.set_trace()
        #         # filter_2d[i] = splatted_2d[i, pts_pick[i, 0]::self.d1, pts_pick[i, 1]::self.d1][:self.size2d//self.d1, :self.size2d//self.d1]
        # else:
        #     splatted_2d[coord] = 1.0

        for i in range(batch_size):
            splatted_2d[i] = torch.cuda.sparse.FloatTensor(coord[i], tmp[i],
                                  torch.Size([self.size2d, self.size2d, c])).to_dense()
            # filter_2d[i] = splatted_2d[i, pts_pick[i, 0]::self.d1, pts_pick[i, 1]::self.d1][:self.size2d//self.d1, :self.size2d//self.d1]
        
        # if not self.normal_channel:
        #     # cutoff = filter_2d[filter_2d>0].mean() * 2
        #     # filter_2d[filter_2d>cutoff] = cutoff
        #     splatted_2d[splatted_2d>0] = 1.0

        return splatted_2d



    def forward(self, pc1, features):
        d = 2
        batch_size = features.size(0)
        num_pts = features.size(-1)
        # import pdb; pdb.set_trace()

        pcoord_long = pc1[:, :d].long()
        feat = 2. - (pcoord_long-pc1[:, :d]).abs().sum(dim=1, keepdim=True)

        coord = self.convert2Dcoord(pc1[:, :d].long(), batch_size, num_pts)
        # coord = self.convert2Dcoord(my_round(pc1[:, :d]), batch_size, num_pts)
        # pc1_int = my_round(pc1[:, :d])
        # coord = self.convert2Dcoord(pc1_int, batch_size, num_pts)
        # coord = self.convert2Dcoord(torch.floor(pc1[:, :d]), batch_size, num_pts)
        
        filter_2d = self.get2D(coord, feat.permute(0, 2, 1), batch_size)

        return filter_2d