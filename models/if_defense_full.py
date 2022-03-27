import os
import numpy as np

import torch
import torch.nn as nn

from .DUP_Net import SORDefense #, DUPNet
import conv_onet
from conv_onet import repulsion_loss 
# from .pointnet_cls import get_model
# import importlib

class IFDefense(nn.Module):

    def __init__(self, pnet_cls, num_cls=40, sor_k=2, sor_alpha=1.1,
                 npoint=1024, padding_scale=0.9):
        super(IFDefense, self).__init__()

        self.npoint = npoint
        self.padding_scale = padding_scale
        self.sor = SORDefense(k=sor_k, alpha=sor_alpha)

        dev = torch.device("cuda")

        with open('conv_onet/convonet_3plane_mn40.yaml', 'r') as f:
            cfg = yaml.load(f)
        self.convonet = conv_onet.config.get_model(cfg, device=dev, dataset=None)
        self.generator = conv_onet.config.get_generator(self.convonet, cfg, device=dev)

        # self.dupnet = defend_point_cloud(npoint=self.npoint, up_ratio=4)
        self.pnet_cls = pnet_cls

        # MODEL = importlib.import_module('pointnet_cls')
        # self.pnet_cls = MODEL.get_model(num_cls,normal_channel=False)
        # self.pnet_cls = get_model(num_cls, normal_channel=False)

    def process_data(self, pc):
        batch_proc_pc = [
                    normalize_cube(one_pc, 600, self.padding_scale) for one_pc in pc
                ]
        batch_proc_sel_pc = torch.cat([
                    one_pc[1] for one_pc in batch_proc_pc
                ], dim=0).float()
        try:
            batch_proc_pc = torch.cat([
                one_pc[0] for one_pc in batch_proc_pc
            ], dim=0).float()
        except RuntimeError:
            batch_proc_pc = [
                one_pc[0][0] for one_pc in batch_proc_pc
            ]

        c = self.generator.model.encode_inputs(batch_proc_sel_pc)
        # z is of no use
        z = None

        # init points and optimize
        points = self.init_points(batch_proc_pc)
        points.requires_grad_()
        points = self.optimize_points(points, z, c,
                                 rep_weight=500,
                                 iterations=200,
                                 printing=True)

        return points


    def optimize_points(self, opt_points, z, c,
                        rep_weight=1.,
                        iterations=1000,
                        printing=False):
        """Optimization process on point coordinates.

        Args:
            opt_points (tensor): input init points to be optimized
            z (tensor): latent code
            c (tensor): feature vector
            iterations (int, optional): opt iter. Defaults to 1000.
            printing (bool, optional): print info. Defaults to False.
        """
        # 2 losses in total
        # Geo-aware loss enforces occ_value = occ_threshold by BCE
        # Dist-aware loss pushes points uniform by repulsion loss
        opt_points = opt_points.float().cuda()
        opt_points.requires_grad_()
        B, K = opt_points.shape[:2]

        # GT occ for surface
        with torch.no_grad():
            occ_threshold = torch.ones(
                (B, K)).float().cuda() * args.threshold

        opt = torch.optim.Adam([opt_points], lr=args.lr)

        # start optimization
        for i in range(iterations + 1):
            # 1. occ = threshold
            occ_value = self.generator.model.decode(opt_points, c).logits
            occ_loss = F.binary_cross_entropy_with_logits(
                occ_value, occ_threshold, reduction='none')  # [B, K]
            occ_loss = torch.mean(occ_loss)
            occ_loss = occ_loss * K

            # 2. repulsion loss
            rep_loss = torch.tensor(0.).float().cuda()
            if rep_weight > 0.:
                rep_loss = repulsion_loss(opt_points)  # [B]
                rep_loss = torch.mean(rep_loss)
                rep_loss = rep_loss * rep_weight

            loss = occ_loss + rep_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if printing and i % 100 == 0:
                print('iter {}, loss {:.4f}'.format(i, loss.item()))
                print('occ loss: {:.4f}, '
                      'rep loss: {:.4f}\n'
                      'occ value mean: {:.4f}'.
                      format(occ_loss.item(),
                             rep_loss.item(),
                             torch.sigmoid(occ_value).mean().item()))
        opt_points.detach_()
        opt_points = self.normalize_batch_pc(opt_points)
        return opt_points


    def init_points(self, pc, init_sigma=0.01):
        """Initialize points to be optimized.

        Args:
            pc (tensor): input (adv) pc, [B, N, 3]
        """
        # with torch.no_grad():
        B = len(pc)
        # init points from ori_points with noise
        # if not enough points in input pc, randomly duplicate
        # else select a subset
        if isinstance(pc, list):  # after SOR
            idx = [
                torch.randint(
                    0, len(one_pc),
                    (self.npoint,)).long().cuda() for one_pc in pc
            ]
        else:
            idx = torch.randint(
                0, pc.shape[1], (B, self.npoint)).long().cuda()
        points = torch.stack([
            pc[i][idx[i]] for i in range(B)
        ], dim=0).float().cuda()

        # add noise
        noise = torch.randn_like(points) * init_sigma
        points = torch.clamp(
            points + noise,
            min=-0.5 * self.padding_scale,
            max=0.5 * self.padding_scale)
        return points

    def normalize_batch_pc(self, points):
        """points: [batch, K, 3]"""
        centroid = torch.mean(points, dim=1)  # [batch, 3]
        points -= centroid[:, None, :]  # center, [batch, K, 3]
        dist = torch.sum(points ** 2, dim=2) ** 0.5  # [batch, K]
        max_dist = torch.max(dist, dim=1)[0]  # [batch]
        points /= max_dist[:, None, None]
        return points

    def normalize_cube(self, pc, num_points=None, padding_scale=1.):
        """Center and scale to be within unit cube.
        Inputs:
            pc: np.array of [K, 3]
            num_points: pick a subset of points as OccNet input.
            padding_scale: padding ratio in unit cube.
        """
        # normalize into unit cube
        import pdb; pdb.set_trace()
        center = pc.mean(dim=0)  # [3]
        centered_pc = pc - center
        max_dim = centered_pc.max(dim=0)[0]  # [3]
        min_dim = centered_pc.min(dim=0)[0]  # [3]
        scale = (max_dim - min_dim).max()
        scaled_centered_pc = centered_pc / scale * padding_scale

        # select a subset as ONet input
        if num_points is not None and scaled_centered_pc.shape[0] > num_points:
            idx = np.random.choice(
                scaled_centered_pc.shape[0], num_points,
                replace=False)
            pc = scaled_centered_pc[idx]
        else:
            pc = scaled_centered_pc

        return scaled_centered_pc.unsqueeze(0), pc.unsqueeze(0)



    def forward(self, x):
        # with torch.no_grad():
        #     x = self.sor(x)  # a list of pc
        #     x = self.process_data(x)  # to batch input
        #     x = self.pu_net(x)  # [B, N * r, 3]
        # return x
        x = x[:self.npoint].transpose(2, 1)
        x = self.sor(x)

        # x = self.dupnet(x)
        x = self.process_data(x)

        x = x[:self.npoint].transpose(2, 1)
        pred, _ = self.pnet_cls(x)
        return pred, _

