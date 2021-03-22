import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STN3dMod(nn.Module):
    def __init__(self, channel):
        super(STN3dMod, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([2, -1, -1, -1, 2, -1, -1, -1, 2]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, 3, 3)

        iden = iden.view(-1, 3, 3)
        x = x.view(-1, 2, 3)

        iden[:, :-1, :] +=  x
        iden[:, -1, :] -= x.sum(dim=1)
        return iden


class TransMatNet(nn.Module):

    def __init__(self, channel, num_pts, in_dim, out_dim):
        super().__init__()

        # self.in_channels = channel
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.stn = STN3dMod(channel)

        self.conv1 = nn.Sequential(
            nn.Conv1d(channel, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        # self.fc = nn.Linear(64 * num_pts, in_dim * (out_dim-1))
        self.fc1 = nn.Linear(64 * num_pts, 1024)
        self.fc2 = nn.Linear(1024, in_dim * (out_dim-1))

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        # self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        # self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        # self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        # self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    # def _make_layer(self, block, out_channels, num_blocks, stride):
    #     """make resnet layers(by layer i didnt mean this 'layer' was the 
    #     same as a neuron netowork layer, ex. conv layer), one layer may 
    #     contain more than one residual block 

    #     Args:
    #         block: block type, basic block or bottle neck block
    #         out_channels: output depth channel number of this layer
    #         num_blocks: how many blocks per layer
    #         stride: the stride of the first block of this layer
        
    #     Return:
    #         return a resnet layer
    #     """

    #     # we have num_block blocks per layer, the first block 
    #     # could be 1 or 2, other blocks would always be 1
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels * block.expansion
        
    #     return nn.Sequential(*layers)

    def forward(self, x):

        # import pdb; pdb.set_trace()
        b = x.size(0)


        trans = self.stn(x)

        return trans


        # output = self.conv1(x)
        # output = self.conv2(output)


        # # output = self.conv1(x)
        # # output = self.conv2(output)
        # # output = self.conv3_x(output)
        # # output = self.conv4_x(output)
        # # output = self.conv5_x(output)
        # # output = self.avg_pool(output)
        # output = output.view(b, -1)
        # # output = self.fc(output)


        # output = self.fc1(output)
        # output = self.fc2(output)

        # output = output.view(b, -1, self.in_dim)
        
        # # output_mat = torch.zeros((b, self.out_dim, self.in_dim), dtype=torch.float32).cuda()
        # output_mat = torch.FloatTensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]).cuda().repeat(b, 1, 1)

        # output_mat[:, :-1, :] += output
        # output_mat[:, -1, :] -= output.sum(dim=1)

        # return output_mat 

# def resnet18():
#     """ return a ResNet 18 object
#     """
#     return ResNet(BasicBlock, [2, 2, 2, 2])

# def resnet34():
#     """ return a ResNet 34 object
#     """
#     return ResNet(BasicBlock, [3, 4, 6, 3])

# def resnet50(num_cls=100):
#     """ return a ResNet 50 object
#     """
#     return ResNet(BottleNeck, [3, 4, 6, 3], num_cls)

# def resnet101():
#     """ return a ResNet 101 object
#     """
#     return ResNet(BottleNeck, [3, 4, 23, 3])

# def resnet152():
#     """ return a ResNet 152 object
#     """
#     return ResNet(BottleNeck, [3, 8, 36, 3])



