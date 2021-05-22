import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


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
