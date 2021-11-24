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
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--eps', default=0.1, type=float, help='learning rate in training [default: 0.1]')
    parser.add_argument('--eps_iter', default=0.01, type=float, help='learning rate in training [default: 0.01]')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training [default: 0.1]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--file_affix', type=str, default='', help='log file/save folder affix')
    parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    parser.add_argument('--num_cls', type=int, default=40, help='Number of classes [default: 40]')

    parser.add_argument('--n', type=int, default=40, help='step')

    parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
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

def FGSM(classifier, criterion, points, targets, args):
    points = torch.tensor(points).transpose(2,1).cuda()
    targets = torch.LongTensor(targets).cuda()
    points.requires_grad = True


    st = datetime.datetime.now().timestamp()

    pred, _ = classifier(points)
    loss = criterion(pred, targets)
    loss.backward()

    data_grad = points.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(points, args.learning_rate, data_grad)

    st = datetime.datetime.now().timestamp() - st
    return perturbed_data.transpose(2, 1)[0].cpu().data.numpy(), st


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

def JGBA(classifier, criterion, points, targets, args):
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
        # outputs = model_0(xvar)
        # outputs = outputs[:, :NUM_POINT]
        outputs, _ = classifier(xvar.transpose(2, 1))
        loss = criterion(outputs, yvar)
        loss.backward()
        grad_np = xvar.grad.detach().cpu().numpy()[0]

        xvar_should = torch.tensor(x_adv[None,:,:]).cuda()
        xvar_should.requires_grad = True
        # xvar_should = pytorch_utils.to_var(torch.from_numpy(x_adv[None,:,:]), cuda=True, requires_grad=True)
        # outputs_should = model_0(xvar_should)
        # outputs_should = outputs_should[:, :NUM_POINT]
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

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = experiment_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)
    atk_dir = experiment_dir.joinpath('attacked_%s%s/' % (args.attack, args.file_affix))
    atk_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
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

    if args.model == 'lattice_cls':
        classifier = MODEL.get_model(num_class,
            normal_channel=args.normal,
            backbone=get_backbone(args.backbone, num_class, 1), s=args.dim*3).cuda()
    elif args.model == 'pointnet_ddn':
        print('using ddn')
        dnn_conf = {
            'input_transform': False,
            'feature_transform': False,
            'robust_type': 'W',
            'alpha': 1.0
        }
        classifier = MODEL.get_model(
                        num_class, dnn_conf['input_transform'], 
                        dnn_conf['feature_transform'], 
                        dnn_conf['robust_type'], 
                        dnn_conf['alpha']
                    ).cuda()
    else:
        classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = MODEL.get_adv_loss(num_class).cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    classifier = classifier.eval()
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

        images, targets = data
        targets = targets[:, 0]
        images = images.transpose(2, 1)
        images, targets = images.cuda(), targets.cuda()

        visit += 1
        with torch.no_grad():
            pred, _ = classifier(images)
            pred = pred.data.max(1)[1]
            if not pred.eq(targets.long().data):
                continue
        correct_ori += 1

        # import pdb; pdb.set_trace()

        if args.attack == 'JGBA':
            pert_img, tm = JGBA(classifier, criterion,
                                    images.transpose(2, 1).cpu().data.numpy(),
                                    targets.cpu().data.numpy(), args)
        elif args.attack == 'FGSM':
            pert_img, tm = FGSM(classifier, criterion,
                                    images.transpose(2, 1).cpu().data.numpy(),
                                    targets.cpu().data.numpy(), args)

        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            pred_adv, _ = classifier(torch.tensor(pert_img[None, :]).transpose(2, 1).cuda())
            pred_adv = pred_adv.data.max(1)[1]
            if pred_adv.eq(targets.long().data):        # untargeted attack success.
                correct += 1

        attack_timer.append(tm)
        # mean_correct.append(correct/args.batch_size)
        cloud_list.append(images.transpose(2, 1).cpu().data.numpy())
        lbl_list.append(targets.cpu().data.numpy())

        pert_list.append(pert_img[None])
        pred_list.append(pred_adv.cpu().data.numpy())
        # dist, img = attack_one_batch(classifier, criterion, images, targets, args, optimizer)
        # dist_list.append(dist)

        if batch_id % 10 == 0:
            log_string('%d done, success rate: %f, test accu.: %f' % (visit, (correct_ori-correct)/correct_ori, correct/visit))

    log_string('success rate: %f, test accu.: %f' % ((correct_ori-correct)/correct_ori, correct/visit))
    log_string('Attack Mean Time: %fs on %d point clouds, %d orig correct, %d successfully attacked'% (sum(attack_timer)/len(attack_timer), visit, correct_ori, (correct_ori-correct)))

    # import pdb; pdb.set_trace()

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
    file_handler = logging.FileHandler('%s/%s_pert_%s%s.txt' % (log_dir, args.model, args.attack, args.file_affix))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
