import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet import STN3d #, PointNetEncoder #, feature_transform_reguliarzer

import math

from resnet import resnet50


class SparseSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, size, cuda):
        """

        :param ctx:
        :param indices: (1, B*d1*N)
        :param values: (B*d1*N, feat_size)
        :param size: (B*(H+1), feat_size)
        :param cuda: bool
        :return: (B*(H+1), feat_size)
        """

        ctx.save_for_backward(indices)

        if cuda:
            output = torch.cuda.sparse.FloatTensor(indices, values, size)
        else:
            output = torch.sparse.FloatTensor(indices, values, size)

        output = output.to_dense()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors

        grad_values = None
        if ctx.needs_input_grad[1]:
            grad_values = grad_output[indices[0, :], indices[1,:], :]
            # grad_values = grad_output[indices.squeeze(0), :]

        return None, grad_values, None, None


sparse_sum = SparseSum.apply

class get_model(nn.Module):

    # s is the final image scale
    def __init__(self, k=40, normal_channel=True, backbone=resnet50(40), s=128*3):
        super(get_model, self).__init__()


        self.lat_transform = LatticeGen(s, normal_channel)
        self.size2d = s
        self.network_2d = backbone

        self.normal_channel = normal_channel


    def forward(self, x):
        if self.normal_channel:
            vv = x[:, 3:]
            # vv = x[:, :3]
        else:
            vv = torch.ones((x.size(0), 1, x.size(2))).cuda()

        # returned splatted has a shape of [b, size, size, c]
        splatted_2d, splatted_2d_2, _ = self.lat_transform(x[:, :3] * (self.size2d//2 - 2), vv)
        # splatted_2d = torch.cat((splatted_2d, splatted_2d_2), 3).permute(0, 3, 1, 2).contiguous()
        splatted_2d = splatted_2d.permute(0, 3, 1, 2).contiguous()

        # splatted_2d = x
        import datetime
        st = datetime.datetime.now().timestamp()

        # network takes [b, c, size, size]
        outputs = self.network_2d(splatted_2d)





        return outputs, [_[0], _[1]]#[_[0].permute(0, 3, 1, 2), splatted_2d, _[1], st]



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



class LatticeGen(nn.Module):
    def __init__(self, s, normal_channel):
        super(LatticeGen, self).__init__()

        d = 3
        self.d = d
        # self.d1 = self.d + 1


        self.d1 = self.d
        self.size2d = s
        self.normal_channel = normal_channel


        self.elevate_mat = (torch.FloatTensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]) / torch.tensor(6.).sqrt() ).cuda()

        # canonical
        canonical = torch.arange(self.d1, dtype=torch.long)[None, :].repeat(self.d1, 1)
        # (d+1, d+1)
        for i in range(1, self.d1):
            canonical[-i:, i] = i - self.d1
        self.canonical = canonical.cuda()

        self.dim_indices = torch.arange(self.d1, dtype=torch.long)[:, None].cuda()


    def get_keys_and_barycentric(self, pc, is_2nd_trans=False):
        """
        :param pc: (self.d, N -- undefined)
        :return:
        """
        batch_size = pc.size(0)
        num_points = pc.size(-1)
        point_indices = torch.arange(num_points, dtype=torch.long)[None, None, :]
        batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None, None]

        if is_2nd_trans:
        	elevated = torch.matmul(self.elevate_mat2, pc)
        else:
        	# elevated = torch.matmul(self.elevate_mat, pc) * self.expected_std  # (d+1, N)
        	elevated = torch.matmul(self.elevate_mat, pc) # * self.expected_std  # (d+1, N)

        greedy = (torch.round(elevated / self.d1) * self.d1)  # (d+1, N)


        el_minus_gr = elevated - greedy


        rank = torch.sort(el_minus_gr, dim=1, descending=True)[1]
        # the following advanced indexing is different in PyTorch 0.4.0 and 1.0.0
        #rank[rank, point_indices] = self.dim_indices  # works in PyTorch 0.4.0 but fail in PyTorch 1.x
        index = rank.clone()

        rank[batch_indices, index, point_indices] = self.dim_indices  # works both in PyTorch 1.x(has tested in PyTorch 1.2) and PyTorch 0.4.0
        del index


        remainder_sum = greedy.sum(dim=1, keepdim=True) / self.d1

        rank_float = rank.type(torch.float32)
        cond_mask = ((rank_float >= self.d1 - remainder_sum) * (remainder_sum > 0) + \
                     (rank_float < -remainder_sum) * (remainder_sum < 0)) \
            .type(torch.float32)
        sum_gt_zero_mask = (remainder_sum > 0).type(torch.float32)
        sum_lt_zero_mask = (remainder_sum < 0).type(torch.float32)
        sign_mask = -1 * sum_gt_zero_mask + sum_lt_zero_mask

        greedy += self.d1 * sign_mask * cond_mask
        rank += (self.d1 * sign_mask * cond_mask).type_as(rank)
        rank += remainder_sum.type(torch.long)

        # barycentric
        el_minus_gr = elevated - greedy
        greedy = greedy.type(torch.long)

        barycentric = torch.zeros((batch_size, self.d1 + 1, num_points), dtype=torch.float32).cuda()

        barycentric[batch_indices, self.d - rank, point_indices] += el_minus_gr
        barycentric[batch_indices, self.d1 - rank, point_indices] -= el_minus_gr
        barycentric /= self.d1
        barycentric[batch_indices, 0, point_indices] += 1. + barycentric[batch_indices, self.d1, point_indices]
        barycentric = barycentric[:, :-1, :]

        # import pdb; pdb.set_trace()

        # canonical[rank, :]: [d1, num_pts, d1]
        #                     (d1 dim coordinates) then (d1 vertices of a simplex)
        keys = greedy[:, :, :, None] + self.canonical[rank, :]  # (d1, num_points, d1)
        # rank: rearrange the coordinates of the canonical



        del elevated, greedy, rank, remainder_sum, rank_float, \
            cond_mask, sum_gt_zero_mask, sum_lt_zero_mask, sign_mask
        return keys, barycentric, el_minus_gr

    # def get_filter_size(self, radius):
    #     return (radius + 1) ** self.d1 - radius ** self.d1

    def convert2Dcoord(self, coord, batch_size, num_pts):
        offset = coord.min(dim=2)[0]

        coord -= offset.view(batch_size, -1, 1).expand(batch_size, -1, self.d1*num_pts)
        pts_pick = (-offset) % 3
        return coord, pts_pick

    def get2D(self, coord, tmp, pts_pick, batch_size):
        # remove points that are out of range in 2d image
        idx_out_range = coord >= self.size2d
        c = tmp.size(-1)

        coord[idx_out_range] = 0
        idx_out_range = idx_out_range.sum(1).nonzero()
        tmp[idx_out_range[:, 0], idx_out_range[:, 1]] = 0


        splatted_2d = torch.zeros((batch_size, self.size2d, self.size2d, c), dtype=torch.float32).cuda()
        filter_2d = torch.zeros((batch_size, self.size2d//self.d1, self.size2d//self.d1, c), dtype=torch.float32).cuda()

        for i in range(batch_size):
            splatted_2d[i] = torch.cuda.sparse.FloatTensor(coord[i], tmp[i],
                                  torch.Size([self.size2d, self.size2d, c])).to_dense()
            # import pdb; pdb.set_trace()
            filter_2d[i] = splatted_2d[i, pts_pick[i, 0]::self.d1, pts_pick[i, 1]::self.d1][:self.size2d//self.d1, :self.size2d//self.d1]

        # if not self.normal_channel:
        #     # cutoff = filter_2d[filter_2d>0].mean() * 2
        #     # filter_2d[filter_2d>cutoff] = cutoff
        #     filter_2d[filter_2d>0] = 1.0

        return filter_2d



    def forward(self, pc1, features):
        # with torch.no_grad():
            # keys, bary, el_minus_gr = self.get_single(pc1[0])
            keys, in_barycentric, _ = self.get_keys_and_barycentric(pc1)
            # keys2, in_barycentric2, _2 = self.get_keys_and_barycentric(pc1, True)

            d = 2


            batch_size = features.size(0)
            num_pts = features.size(-1)
            # batch_indices = torch.arange(batch_size, dtype=torch.long)

            # batch_indices = batch_indices.pin_memory()

            # convert to 2d image
            # coord [3, d * num_pts]: [d] + [d] + ... + [d]
            coord = keys[:, :d].view(batch_size, d, -1)
            # import pdb; pdb.set_trace()

            coord, pts_pick = self.convert2Dcoord(coord, batch_size, num_pts)

            # tmp: [batch, d, d, num]
            # d: coordinates of points; then d: vertices of each simplex
            tmp = in_barycentric[:, None, :, :] * features[:, :, None, :]
            # tmp: [d, d * num]
            # d * num_pts: [d] + [d] + ... + [d]
            cc = tmp.size(1)
            tmp = tmp.permute(0, 1, 3, 2).contiguous().view(batch_size, cc, -1).permute(0, 2, 1)

            filter_2d = self.get2D(coord, tmp, pts_pick, batch_size)



            return filter_2d, None, [filter_2d, keys.view(batch_size, 3, -1)]
        # return filter_2d, filter_2d_2, [filter_2d]#[splatted_2d, keys.view(batch_size, 3, -1)]



class RotatePCA(nn.Module):
    def __init__(self):
        super(RotatePCA, self).__init__()

    def forward(self, pc1, features=None):
        # with torch.no_grad():

            y = pc1.permute(0, 2, 1)

            # get the least important direction
            aa, bb, cc = torch.pca_lowrank(y)
            vec_low = cc[:, :, -1]

            trans_norm = torch.cuda.FloatTensor([1, 1, 1]) / torch.tensor(3.).sqrt()
            trans_norm = trans_norm.expand(pc1.size(0), -1)

            # cross product of least important vector and lattice transform plane vector
            # the cross product is also the rotation axis
            # the value is the sin value
            rot_axis = torch.cross(vec_low, trans_norm, dim=1)
            sin_rot = rot_axis.norm(dim=1)
            rot_axis = rot_axis / sin_rot[:, None]

            # dot product
            # also cos value
            cos_rot = (vec_low * trans_norm).sum(dim=1)

            # Rodrigues' rotation formula
            rot_pts = pc1 * cos_rot[:, None, None]
            rot_pts += torch.cross(rot_axis[:, :, None].expand(pc1.shape), pc1) * sin_rot[:, None, None]
            rot_pts += rot_axis[:, :, None] * (rot_axis[:, :, None] * pc1).sum(dim=1, keepdim=True) * (1 - cos_rot)[:, None, None]

            # y = rot_pts.permute(0, 2, 1)
            # aaa, bbb, ccc = torch.pca_lowrank(y)

            return rot_pts#.permute(0, 2, 1)

