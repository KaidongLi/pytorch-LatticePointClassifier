import os
import numpy as np

import torch
import torch.nn as nn

from .DUP_Net import DUPNet
# from .pointnet_cls import get_model
# import importlib

class DUPNetFull(nn.Module):

    def __init__(self, pnet_cls, num_cls=40, sor_k=2, sor_alpha=1.1,
                 npoint=1024, up_ratio=4):
        super(DUPNetFull, self).__init__()

        self.npoint = npoint
        self.dupnet = DUPNet(npoint=self.npoint, up_ratio=4)
        self.pnet_cls = pnet_cls

        # MODEL = importlib.import_module('pointnet_cls')
        # self.pnet_cls = MODEL.get_model(num_cls,normal_channel=False)
        # self.pnet_cls = get_model(num_cls, normal_channel=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # with torch.no_grad():
        #     x = self.sor(x)  # a list of pc
        #     x = self.process_data(x)  # to batch input
        #     x = self.pu_net(x)  # [B, N * r, 3]
        # return x
        x = x[:self.npoint].transpose(2, 1)
        x = self.dupnet(x)
        x = x[:self.npoint].transpose(2, 1)
        pred, _ = self.pnet_cls(x)
        return pred, _
