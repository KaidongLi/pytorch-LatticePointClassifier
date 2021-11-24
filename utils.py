import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#from dataset import CIFAR100Train, CIFAR100Test


CIFAR100_LABELS_LIST = [
'apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
'bicycle', 'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
'cans', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cups', 'dinosaur',
'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
'house', 'kangaroo', 'keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion',
'lizard', 'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse',
'mushrooms', 'oak', 'oranges', 'orchids', 'otter', 'palm', 'pears',
'pickup truck', 'pine', 'plain', 'plates', 'poppies', 'porcupine',
'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses',
'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
'spider', 'squirrel', 'streetcar', 'sunflowers', 'sweet peppers', 'table',
'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman',
'worm'
]

CIFAR100_LABELS = [
'beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
'bottles', 'bowls', 'cans', 'cups', 'plates',
'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
'clock', 'keyboard', 'lamp', 'telephone', 'television',
'bed', 'chair', 'couch', 'table', 'wardrobe',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper',
'cloud', 'forest', 'mountain', 'plain', 'sea',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
'maple', 'oak', 'palm', 'pine', 'willow',
'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
]

BIT_PER_MAJOR = 20
BIT_PER_MINOR = 16
BIT = BIT_PER_MAJOR + BIT_PER_MINOR * 5
POS_VAL = 1.0


def oriToMajorMinorList(ori_list):
    new_list = [CIFAR100_LABELS.index(CIFAR100_LABELS_LIST[x]) for x in ori_list]
    new_list_major = [x // 5 for x in new_list]
    return new_list_major, new_list



def oriToNewIdx(ori_label):
    return CIFAR100_LABELS.index(CIFAR100_LABELS_LIST[ori_label])

def newToOriIdx(new_label):
    return CIFAR100_LABELS_LIST.index(CIFAR100_LABELS[new_label])


def encodeConverter(idx):
    encode_label = [0.0] * 100
    encode_label[idx] = POS_VAL
    return encode_label

def decodeClassifier(raw_output):
    target_code = []

    for i in range(100):
        target_code.append(encodeConverter(i))

    target_code = torch.Tensor(target_code)
    target_code = target_code.cuda()

    res_cls = []
    for i in range(raw_output.size()[0]):
        pred = raw_output[i]
        pred = pred.expand(100, -1)
        #import pdb; pdb.set_trace()
        pred = pred - target_code
        dists = pred.norm(dim=1)
        #dists = [torch.dist(raw_output[i], tgt) for tgt in target_code]
        dist, idx = dists.min(dim=0)
        res_cls.append( newToOriIdx(idx) )

    res_cls = torch.LongTensor(res_cls)
    res_cls = res_cls.cuda()

    return res_cls


def get_backbone(backbone, num_cls=100, c=1, use_gpu=True):
    """ return given network
    """

    if backbone == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_cls, c)
    elif backbone == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_cls, c)
    elif backbone == 'efficientnet-b3':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b3', advprop=True, num_classes=num_cls, in_channels=c)
    elif backbone == 'efficientnet-b5':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b5', advprop=True, num_classes=num_cls, in_channels=c)
        # net = EfficientNet.from_name('efficientnet-b5', num_classes=num_cls, in_channels=1) #advprop=True,
    elif backbone == 'efficientnet-b7':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b7', advprop=True, num_classes=num_cls, in_channels=c)
    elif backbone == 'efficientnet-b8':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b8', advprop=True, num_classes=num_cls, in_channels=c)
    elif backbone == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_cls, c)

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]




import os
import time
import csv

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    # Reset for new bar
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    #L.append('  Step: %s' % format_time(step_time))
    L.append(' Time: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def log_row(logname, row):
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(row)

        
def adjust_lr_steep(lr_0, param_groups, epoch, adj_params):
    steps = adj_params['steps']
    decay_rates = adj_params ['decay_rates']
    for param_group in param_groups:
        lr = lr_0
        for j in range(len(steps)):
            if epoch >= steps[j]:
                lr = lr * decay_rates[j]
        param_group['lr'] = lr
    return param_groups