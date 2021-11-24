from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.AttackModelNetLoader import AttackModelNetLoader
from data_utils.AttackScanNetLoader import AttackScanNetLoader
from data_utils.ScanNetDataLoader import ScanNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from utils import get_backbone # get_cifar_training, get_cifar_test
from models.DUP_noD_Net import DUP_noD_Net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

clip_min = -1.0
clip_max = 1.0
TOP_K = 10
NUM_STD = 1.0


def log_string(str):
    logger.info(str)
    print(str)
def log_only_string(str):
    logger.info(str)
    # print(str)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    # parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    # parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--eps', default=0.1, type=float, help='learning rate in training [default: 0.1]')
    parser.add_argument('--eps_iter', default=0.01, type=float, help='learning rate in training [default: 0.01]')
    parser.add_argument('--eps_fgsm', default=0.1, type=float, help='learning rate in training [default: 0.1]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    # parser.add_argument('--num_sample', type=int, default=25, help='number of samples per class [default: 25]')
    parser.add_argument('--file_affix', type=str, default='', help='log file/save folder affix')
    parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    parser.add_argument('--num_cls', type=int, default=40, help='Number of classes [default: 40]')


    # parser.add_argument('--target', type=int, default=5, help='target class index')
    # parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
    # parser.add_argument('--upper_bound_weight', type=float, default=80, help='upper_bound value for the parameter lambda')
    parser.add_argument('--n', type=int, default=40, help='step')
    # parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')

    # parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    # parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
    parser.add_argument('--num_attack', type=int, default=-1, help='number of samples to attack [default: -1]')
    parser.add_argument('--attack', type=str, default='JGBA', help='[JGBA, FGSM]')


    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        # print(pred_choice)
        # import pdb; pdb.set_trace()
        for cat in np.unique(target.cpu()):

            # resolve tensor cannot be (target==cat) eq() to a numpy bug
            cat = cat.item()

            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def FGSM(classifier_0, classifier, criterion, points, targets, args):
    assert points.shape[0] == 1, 'Batch size must be 1'
    NUM_POINT = args.num_point
    points = points[0]

    x_adv = np.copy(points)
    yvar = torch.LongTensor(targets).cuda()

    st = datetime.datetime.now().timestamp()
    indices_saved, x_sor = remove_outliers_defense(x_adv, top_k=TOP_K, num_std=NUM_STD)

    xvar = torch.tensor(x_sor[None,:,:]).cuda()
    xvar.requires_grad = True
    # xvar = pytorch_utils.to_var(torch.from_numpy(x_sor[None,:,:]), cuda=True, requires_grad=True)
    outputs = classifier_0(xvar)
    outputs = outputs[:, :NUM_POINT]
    outputs, _ = classifier(xvar.transpose(2, 1))
    loss = criterion(outputs, yvar)
    loss.backward()
    data_grad = xvar.grad.data

    # Call FGSM Attack
    perturbed_pts = fgsm_attack(xvar, args.eps_fgsm, data_grad)

    perturbed_pts = perturbed_pts.cpu().data.numpy()
    x_adv = x_adv[None]
    for idx, index_saved in enumerate(indices_saved):
        x_adv[:,index_saved,:] = perturbed_pts[:,idx,:]

    st = datetime.datetime.now().timestamp() - st
    # import pdb; pdb.set_trace()

    return x_adv[0], st


nbrs = NearestNeighbors(n_neighbors=TOP_K+1, algorithm='auto', metric='euclidean', n_jobs=-1)
def remove_outliers_defense(x, top_k=10, num_std=1.0):
    top_k = int(top_k)
    num_std = float(num_std)
    if len(x.shape) == 3:
        x = x[0]

    nbrs.fit(x)
    dists = nbrs.kneighbors(x, n_neighbors=top_k + 1)[0][:, 1:]
    dists = np.mean(dists, axis=1)

    avg = np.mean(dists)
    std = num_std * np.std(dists)

    remove_indices = np.where(dists > (avg + std))[0]

    save_indices = np.where(dists <= (avg + std))[0]
    x_remove = np.delete(np.copy(x), remove_indices, axis=0)
    return save_indices, x_remove

def JGBA(classifier_0, classifier, criterion, points, targets, args):
    eps = args.eps
    eps_iter = args.eps_iter
    n = args.n
    NUM_POINT = args.num_point

    assert points.shape[0] == 1, 'Batch size must be one'
    points = points[0]

    x_adv = np.copy(points)
    yvar = torch.LongTensor(targets).cuda()

    st = datetime.datetime.now().timestamp()

    for i in range(n):
        indices_saved, x_sor = remove_outliers_defense(x_adv, top_k=TOP_K, num_std=NUM_STD)

        xvar = torch.tensor(x_sor[None,:,:]).cuda()
        xvar.requires_grad = True
        # xvar = pytorch_utils.to_var(torch.from_numpy(x_sor[None,:,:]), cuda=True, requires_grad=True)
        # outputs = classifier_0(xvar)
        # outputs = outputs[:, :NUM_POINT]
        # outputs, _ = classifier(outputs.transpose(2, 1))
        outputs, _ = classifier(xvar.transpose(2, 1))
        loss = criterion(outputs, yvar)
        loss.backward()
        grad_np = xvar.grad.detach().cpu().numpy()[0]

        xvar_should = torch.tensor(x_adv[None,:,:]).cuda()
        xvar_should.requires_grad = True
        # xvar_should = pytorch_utils.to_var(torch.from_numpy(x_adv[None,:,:]), cuda=True, requires_grad=True)
        # outputs_should = classifier_0(xvar_should)
        # outputs_should = outputs_should[:, :NUM_POINT]
        # outputs_should, _ = classifier(outputs_should.transpose(2, 1))
        outputs_should, _ = classifier(xvar_should.transpose(2, 1))
        loss_should = criterion(outputs_should, yvar)
        loss_should.backward()
        grad_1024 = xvar_should.grad.detach().cpu().numpy()[0]

        grad_sor = np.zeros((1024, 3))

        for idx, index_saved in enumerate(indices_saved):
            grad_sor[index_saved,:] = grad_np[idx,:]

        grad_1024 += grad_sor
        grad_1024 = normalize(grad_1024, axis=1)

        perturb = eps_iter * grad_1024
        perturb = np.clip(x_adv + perturb, clip_min, clip_max) - x_adv
        norm = np.linalg.norm(perturb, axis=1)
        factor = np.minimum(eps / (norm + 1e-12), np.ones_like(norm))
        factor = np.tile(factor, (3,1)).transpose()
        perturb *= factor
        x_adv += perturb

    st = datetime.datetime.now().timestamp() - st
    x_perturb = np.copy(x_adv)

    return x_perturb, st



def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    # timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # experiment_dir = Path('./log/')
    # experiment_dir.mkdir(exist_ok=True)
    # experiment_dir = experiment_dir.joinpath('perturbation')
    # experiment_dir.mkdir(exist_ok=True)
    # if args.log_dir is None:
    #     experiment_dir = experiment_dir.joinpath(timestr)
    # else:
    #     experiment_dir = experiment_dir.joinpath(args.log_dir)
    # experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = experiment_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)
    atk_dir = experiment_dir.joinpath('attacked_dup_%s%s/' % (args.attack, args.file_affix))
    atk_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    # logger = logging.getLogger("Model")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # DATA_PATH = 'data/modelnet40_normal_resampled/'

    if args.dataset == 'ModelNet40':
        DATA_PATH = 'data/modelnet40_normal_resampled/'
        TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                        normal_channel=args.normal)
    elif args.dataset == 'ScanNetCls':
        # TEST_PATH  = 'dump/scannet_test_data8316.npz'
        TEST_PATH  = 'data/scannet/test_files.txt'
        TEST_DATASET = ScanNetDataLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = args.num_cls

    # num_class = 100
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    classifier_0 = DUP_noD_Net(npoint=args.num_point, up_ratio=4).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = MODEL.get_adv_loss(num_class).cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])

        checkpoint_0 = 'dump/pu-in_1024-up_4.pth'
        classifier_0.pu_net.load_state_dict( torch.load(checkpoint_0) )
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    classifier = classifier.eval()
    classifier_0.eval()

    attack_timer = []
    pert_list = []
    cloud_list = []
    lbl_list = []
    pred_list = []
    visit = 0
    correct_ori = 0
    correct = 0

    for batch_id, data in enumerate(testDataLoader, 0):
        if args.num_attack > 0 and args.num_attack <= batch_id:
            break

        input, targets = data
        assert input.size(0) == 1, 'Batch size must be one'
        point = input[0].data.numpy()
        targets = targets[:, 0].cuda()

        visit += 1
        with torch.no_grad():
            indices_saved, x = remove_outliers_defense(point, top_k=TOP_K, num_std=NUM_STD)
            x = torch.tensor(x[None,:,:]).cuda()
            x = classifier_0(x)
            x = x[:, :args.num_point].transpose(2, 1)
            pred, _ = classifier(x)
            pred = pred.data.max(1)[1]
            if not pred.eq(targets.long().data):
                continue
        correct_ori += 1

        # import pdb; pdb.set_trace()

        if args.attack == 'JGBA':
            pert_img, tm = JGBA(classifier_0, classifier, criterion,
                                    input.data.numpy(),
                                    targets.cpu().data.numpy(), args)
        elif args.attack == 'FGSM':
            pert_img, tm = FGSM(classifier_0, classifier, criterion,
                                    input.data.numpy(),
                                    targets.cpu().data.numpy(), args)
        else:
            assert False, 'invalid attack method'

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            indices_saved, x = remove_outliers_defense(pert_img, top_k=TOP_K, num_std=NUM_STD)
            x = torch.tensor(x[None,:,:]).cuda()
            x = classifier_0(x)
            x = x[:, :args.num_point].transpose(2, 1)
            pred_adv, _ = classifier(x)
            pred_adv = pred_adv.data.max(1)[1]
            if pred_adv.eq(targets.long().data):
                correct += 1

        attack_timer.append(tm)
        # mean_correct.append(correct/args.batch_size)
        cloud_list.append(input.data.numpy())
        lbl_list.append(targets.cpu().data.numpy())

        pert_list.append(pert_img[None])
        pred_list.append(pred_adv.cpu().data.numpy())
        # dist, img = attack_one_batch(classifier, criterion, images, targets, args, optimizer)
        # dist_list.append(dist)

        if batch_id % 10 == 0:
            log_string('%d done, success rate: %f, test accu.: %f' % (visit, (correct_ori-correct)/correct_ori, correct/visit))

    log_string('success rate: %f, test accu.: %f' % ((correct_ori-correct)/correct_ori, correct/visit))
    log_string('Attack Mean Time: %fs on %d point clouds, %d orig correct, %d successfully attacked'% (sum(attack_timer)/len(attack_timer), visit, correct_ori, (correct_ori-correct)))

    cloud_list = np.concatenate(cloud_list, axis=0)
    pert_list = np.concatenate(pert_list, axis=0)
    lbl_list = np.concatenate(lbl_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)

    np.save(os.path.join(atk_dir, 'orig.npy'), cloud_list)
    np.save(os.path.join(atk_dir, 'fgsm_pert.npy'), pert_list)
    np.save(os.path.join(atk_dir, 'lable.npy'), lbl_list)
    np.save(os.path.join(atk_dir, 'pred.npy'), pred_list)




if __name__ == '__main__':
    args = parse_args()

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('perturbation')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_pertdup_%s%s.txt' % (log_dir, args.model, args.attack, args.file_affix))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
