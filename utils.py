""" helper function

author baiyu
"""

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


'''
def encodeConverter(idx):
    idx_fine = idx % 5
    idx = idx // 5
    encode_label = [0.0] * (20*BIT)
    for i in range(idx * BIT, idx * BIT + BIT_PER_MAJOR):
        encode_label[i] = POS_VAL
    for i in range(idx * BIT + BIT_PER_MAJOR + idx_fine * BIT_PER_MINOR,
                   idx * BIT + BIT_PER_MAJOR + (idx_fine+1) * BIT_PER_MINOR):
        encode_label[i] = POS_VAL
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
'''



def get_backbone(backbone, num_cls=100, c=1, use_gpu=True):
    """ return given network
    """

    if backbone == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_cls, c)
    # elif backbone == 'vgg_mm16':
    #     from models.vgg_mm import vgg16_bn
    #     net = vgg16_bn()
    # elif backbone == 'vgg13':
    #     from models.vgg import vgg13_bn
    #     net = vgg13_bn()
    # elif backbone == 'vgg11':
    #     from models.vgg import vgg11_bn
    #     net = vgg11_bn()
    # elif backbone == 'vgg19':
    #     from models.vgg import vgg19_bn
    #     net = vgg19_bn()
    # elif backbone == 'densenet121':
    #     from models.densenet import densenet121
    #     net = densenet121()
    # elif backbone == 'densenet161':
    #     from models.densenet import densenet161
    #     net = densenet161()
    # elif backbone == 'densenet169':
    #     from models.densenet import densenet169
    #     net = densenet169()
    # elif backbone == 'densenet201':
    #     from models.densenet import densenet201
    #     net = densenet201()
    # elif backbone == 'googlenet':
    #     from models.googlenet import googlenet
    #     net = googlenet()
    # elif backbone == 'inceptionv3':
    #     from models.inceptionv3 import inceptionv3
    #     net = inceptionv3()
    # elif backbone == 'inceptionv4':
    #     from models.inceptionv4 import inceptionv4
    #     net = inceptionv4()
    # elif backbone == 'inceptionresnetv2':
    #     from models.inceptionv4 import inception_resnet_v2
    #     net = inception_resnet_v2()
    # elif backbone == 'xception':
    #     from models.xception import xception
    #     net = xception()
    # elif backbone == 'resnet18':
    #     from models.resnet import resnet18
    #     net = resnet18()
    # elif backbone == 'resnet34':
    #     from models.resnet import resnet34
    #     net = resnet34()
    elif backbone == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_cls, c)
    elif backbone == 'efficientnet-b3':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b3', advprop=True, num_classes=num_cls, in_channels=1)   
    elif backbone == 'efficientnet-b5':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b5', advprop=True, num_classes=num_cls, in_channels=1)
        # net = EfficientNet.from_name('efficientnet-b5', num_classes=num_cls, in_channels=1) #advprop=True, 
    elif backbone == 'efficientnet-b8':
        from efficientnet_pytorch import EfficientNet
        net = EfficientNet.from_pretrained('efficientnet-b8', advprop=True, num_classes=num_cls, in_channels=1)
    # elif backbone == 'resnet101':
    #     from models.resnet import resnet101
    #     net = resnet101()
    # elif backbone == 'resnet152':
    #     from models.resnet import resnet152
    #     net = resnet152()
    # elif backbone == 'preactresnet18':
    #     from models.preactresnet import preactresnet18
    #     net = preactresnet18()
    # elif backbone == 'preactresnet34':
    #     from models.preactresnet import preactresnet34
    #     net = preactresnet34()
    # elif backbone == 'preactresnet50':
    #     from models.preactresnet import preactresnet50
    #     net = preactresnet50()
    # elif backbone == 'preactresnet101':
    #     from models.preactresnet import preactresnet101
    #     net = preactresnet101()
    # elif backbone == 'preactresnet152':
    #     from models.preactresnet import preactresnet152
    #     net = preactresnet152()
    elif backbone == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(num_cls, c)
    # elif backbone == 'resnext101':
    #     from models.resnext import resnext101
    #     net = resnext101()
    # elif backbone == 'resnext152':
    #     from models.resnext import resnext152
    #     net = resnext152()
    # elif backbone == 'shufflenet':
    #     from models.shufflenet import shufflenet
    #     net = shufflenet()
    # elif backbone == 'shufflenetv2':
    #     from models.shufflenetv2 import shufflenetv2
    #     net = shufflenetv2()
    # elif backbone == 'squeezenet':
    #     from models.squeezenet import squeezenet
    #     net = squeezenet()
    # elif backbone == 'mobilenet':
    #     from models.mobilenet import mobilenet
    #     net = mobilenet()
    # elif backbone == 'mobilenetv2':
    #     from models.mobilenetv2 import mobilenetv2
    #     net = mobilenetv2()
    # elif backbone == 'nasnet':
    #     from models.nasnet import nasnet
    #     net = nasnet()
    # elif backbone == 'attention56':
    #     from models.attention import attention56
    #     net = attention56()
    # elif backbone == 'attention92':
    #     from models.attention import attention92
    #     net = attention92()
    # elif backbone == 'seresnet18':
    #     from models.senet import seresnet18
    #     net = seresnet18()
    # elif backbone == 'seresnet34':
    #     from models.senet import seresnet34
    #     net = seresnet34()
    # elif backbone == 'seresnet50':
    #     from models.senet import seresnet50
    #     net = seresnet50()
    # elif backbone == 'seresnet101':
    #     from models.senet import seresnet101
    #     net = seresnet101()
    # elif backbone == 'seresnet152':
    #     from models.senet import seresnet152
    #     net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net


def get_cifar_training(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='/scratch/kaidong/pytorch-cifar100/data/', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_cifar_test(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='/scratch/kaidong/pytorch-cifar100/data/', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

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
