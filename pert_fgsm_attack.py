from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.AttackModelNetLoader import AttackModelNetLoader
from data_utils.AttackScanNetLoader import AttackScanNetLoader
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


from utils import get_backbone # get_cifar_training, get_cifar_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# SCALE_LOW = 2
# SCALE_UP  = 2

SCALE_LOW = 30
SCALE_UP  = 32

# SCALE_LOW = 150
# SCALE_UP  = 192

# CLASS_ATTACK = [0, 2, 4, 8, 22, 25, 30, 33, 35, 37]
# CLASS_ATTACK = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]

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
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training [default: 0.001]')
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
    # parser.add_argument('--step', type=int, default=10, help='binary search step')
    # parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')

    parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
    parser.add_argument('--num_attack', type=int, default=-1, help='number of samples to attack [default: -1]')


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

            # kaidong mod: resolve tensor cannot be (target==cat) eq() to a numpy bug
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

def attack_one_batch(classifier, criterion, points, targets, args):

    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'points_ori' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################
    targets = targets[:, 0]
    points = points.transpose(2, 1)
    points, targets = points.cuda(), targets.cuda()
    points.requires_grad = True


    st = datetime.datetime.now().timestamp()

    pred, _ = classifier(points)
    loss = criterion(pred, targets.long())
    loss.backward()

    data_grad = points.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(points, args.learning_rate, data_grad)

    st = datetime.datetime.now().timestamp() - st

    output, _ = classifier(perturbed_data)

    pred_choice = output.data.max(1)[1]

    return perturbed_data.cpu().data.numpy(), pred_choice.cpu(), st



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
    atk_dir = experiment_dir.joinpath('attacked_fgsm%s/' % (args.file_affix))
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
        # TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
        #                                                  normal_channel=args.normal)
        TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                        normal_channel=args.normal)
    elif args.dataset == 'ScanNetCls':
        # TRAIN_PATH = '/scratch/kaidong/tf-point-cnn/data/test_scan_in/train_files.txt'
        TEST_PATH  = '/scratch/kaidong/tf-point-cnn/data/test_scan_in/test_files.txt'
        # TRAIN_DATASET = ScanNetDataLoader(TRAIN_PATH, npoint=args.num_point, split='train',
        #                                                  normal_channel=args.normal)
        TEST_DATASET = ScanNetDataLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = args.num_cls

    # num_class = 100
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    # classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    # classifier = MODEL.get_model(num_class,normal_channel=args.normal,s=128*3).cuda()


    if args.model == 'lattice_cls':
        classifier = MODEL.get_model(num_class,
            normal_channel=args.normal,
            backbone=get_backbone(args.backbone, num_class, 1), s=args.dim*3).cuda()
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

    # # kaidong debug test
    # with torch.no_grad():
    #     instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)
    #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
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
        visit += 1

        pert_img, pred, tm = attack_one_batch(classifier, criterion, images, targets, args)

        pred = pred[:, None]
        c1 = pred.eq(targets.long().data).sum().item()
        correct += c1


        attack_timer.append(tm)
        cloud_list.append(images.data.numpy())
        lbl_list.append(targets.data.numpy())

        pert_list.append(pert_img)
        pred_list.append(pred.data.numpy())
        # dist, img = attack_one_batch(classifier, criterion, images, targets, args, optimizer)
        # dist_list.append(dist)

        if batch_id % 10 == 0:
            log_string('%d clouds, accu.: %f' % (visit, correct/visit))

        # import pdb; pdb.set_trace()
        # log_string("{}_{}_{} attacked.".format(victim,args.target,j))
        # np.save(os.path.join(atk_dir, '{}_{}_{}_adv.npy' .format(victim,args.target,j)), img)
        # np.save(os.path.join(atk_dir, '{}_{}_{}_adv_f.npy' .format(victim,args.target,j)), img_f[0])
        # np.save(os.path.join(atk_dir, '{}_{}_{}_orig.npy' .format(victim,args.target,j)),images)#dump originial example for comparison
        # np.save(os.path.join(atk_dir, '{}_{}_{}_pred.npy' .format(victim,args.target,j)),preds)
        # if args.model == 'lattice_cls':
        #     np.save(os.path.join(atk_dir, '{}_{}_{}_2dimg.npy' .format(victim,args.target,j)), img_2d)

    log_string('Accu.: %f, succ rate: %f' % (correct/visit, 1-correct/visit))
    log_string('Attack Mean Time: %fs on %d clouds, %d correct'% (sum(attack_timer)/len(attack_timer), visit, correct))

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
    file_handler = logging.FileHandler('%s/%s_pert_fgsm%s.txt' % (log_dir, args.model, args.file_affix))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
