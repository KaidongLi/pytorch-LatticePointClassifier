"""
Author: Benny
Date: Nov 2019
"""
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
CLASS_ATTACK = [0, 2, 4, 5, 8, 22, 30, 33, 35, 37]

def log_string(str):
    logger.info(str)
    print(str)
def log_only_string(str):
    logger.info(str)
    # print(str)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_sample', type=int, default=25, help='number of samples per class [default: 25]')
    parser.add_argument('--file_affix', type=str, default='', help='log file/save folder affix')
    parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    parser.add_argument('--num_cls', type=int, default=40, help='Number of classes [default: 40]')


    parser.add_argument('--target', type=int, default=5, help='target class index')
    parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
    parser.add_argument('--upper_bound_weight', type=float, default=80, help='upper_bound value for the parameter lambda')
    parser.add_argument('--step', type=int, default=10, help='binary search step')
    parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')

    parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')


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


def attack_one_batch(classifier, criterion, points_ori, attacked_label, args, optimizer=None):

    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'points_ori' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################

    is_training = False

    BATCH_SIZE = args.batch_size
    INITIAL_WEIGHT = args.initial_weight
    UPPER_BOUND_WEIGHT = args.upper_bound_weight
    NUM_POINT = args.num_point
    BINARY_SEARCH_STEP = args.step
    NUM_ITERATIONS = args.num_iter

    #the bound for the binary search
    lower_bound=np.zeros(BATCH_SIZE)
    WEIGHT = torch.Tensor(np.ones(BATCH_SIZE) * INITIAL_WEIGHT).cuda()
    upper_bound=np.ones(BATCH_SIZE) * UPPER_BOUND_WEIGHT


    # o_bestdist:   starting with norm 1e10,
    #               recording lowest norm of successful perturbation
    # o_bestscore:  starting with -1,
    #               recording the successful attacked label
    # o_bestattack: starting with 1s,
    #
    o_bestdist = [1e10] * BATCH_SIZE
    o_bestscore = [-1] * BATCH_SIZE
    if args.normal:
        o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT,6))

        o_leastFailAttack = np.ones(shape=(BATCH_SIZE,NUM_POINT,6))
        o_record2D = np.ones(shape=(BATCH_SIZE, args.dim, args.dim, 3))
    else:
        o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT,3))

        o_leastFailAttack = np.ones(shape=(BATCH_SIZE,NUM_POINT,3))
        o_record2D = np.ones(shape=(BATCH_SIZE, args.dim, args.dim, 1))


    o_failPred = [-1] * BATCH_SIZE
    o_failDist = [0] * BATCH_SIZE

    train_timer = []
    b_step = [-1] * BATCH_SIZE
    b_iter = [-1] * BATCH_SIZE

    for out_step in range(BINARY_SEARCH_STEP):
        log_string((" Step {} of {}")
                              .format(out_step, BINARY_SEARCH_STEP))

        # feed_dict[ops['dist_weight']]=WEIGHT

        # pert = torch.normal(0, 0.0000001, size=(BATCH_SIZE,NUM_POINT,3)).cuda()
        # pert = torch.normal(0, 0.05, size=(BATCH_SIZE,NUM_POINT,3)).requires_grad_(True).cuda()
        # pert = torch.normal(0, 0.05, size=(BATCH_SIZE,NUM_POINT,3)).cuda()

        # pert_list = torch.normal(0, 0.05, size=(BATCH_SIZE,NUM_POINT,3)).tolist()
        # pert = torch.tensor(pert_list, requires_grad=True).cuda()

        # parameter_vector = torch.tensor(range(10), dtype=torch.float, requires_grad=True)
        # i = torch.ones(parameter_vector.size(0))
        # sigma = 0.1
        # m = torch.distributions.Normal(parameter_vector, sigma*i)
        # pert = m.rsample()

        # pert = (torch.randn((BATCH_SIZE,NUM_POINT,3), requires_grad=True, device='cuda'))
        INIT_STD = 1e-7
        # INIT_STD = 0.1

        if args.normal:
            pert = torch.normal(0, INIT_STD, size=(BATCH_SIZE,NUM_POINT,6), requires_grad=True, device='cuda')
        else:
            pert = torch.normal(0, INIT_STD, size=(BATCH_SIZE,NUM_POINT,3), requires_grad=True, device='cuda')

        # pert = torch.normal(0, 0.1, size=(BATCH_SIZE,NUM_POINT,3), requires_grad=True, device='cuda')

        # import pdb; pdb.set_trace()


        # pert = torch.empty((BATCH_SIZE,NUM_POINT,3)).normal_(0, 0.01)
        # pert = pert.requires_grad_(True).cuda()


        optimizer = torch.optim.Adam(
            # classifier.parameters(),
            [pert], # + list(classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        # sess.run(tf.assign(ops['pert'],tf.truncated_normal([BATCH_SIZE,NUM_POINT,3], mean=0, stddev=0.0000001)))

        # bestdist: starting with norm 1e10,
        # 			recording lowest norm of successful perturbation
        # bestscore: starting with -1
        #            recording the successful attacked label
        bestdist = [1e10] * BATCH_SIZE
        bestscore = [-1] * BATCH_SIZE

        prev = 1e6

        for iteration in range(NUM_ITERATIONS):
            # pre-process point cloud
            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3], 0.8, 1.)
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            # points = torch.Tensor(points)


            # pert = torch.normal(0, 0.0000001, size=(BATCH_SIZE,NUM_POINT,3))


            # add perturbation
            points = points_ori
            points = points.cuda()
            points[:,:, 0:3] = points[:,:, 0:3] + pert
            # points = points + pert



            points_cls = points.transpose(2, 1)
            points_cls, attacked_label = points_cls.cuda(), attacked_label.cuda()


            optimizer.zero_grad()

            # classifier = classifier.train()

            st = datetime.datetime.now().timestamp()

            pred, _ = classifier(points_cls)
            adv_loss = criterion(pred, attacked_label.long())

            #perturbation l2 constraint
            pert_norm = torch.square(pert).sum(dim=2).sum(dim=1).sqrt().cuda()

            norm_loss = (WEIGHT * pert_norm).mean()

            loss = adv_loss + norm_loss
            loss.backward()
            optimizer.step()

            st = datetime.datetime.now().timestamp() - st
            train_timer.append(st)

            # import pdb; pdb.set_trace()


            pred_cls_np = pred.max(dim=1)[1].cpu().data.numpy()
            pert_norm_np = pert_norm.cpu().data.numpy()
            points_np = points.cpu().data.numpy()

            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                # print(WEIGHT)
                log_string((" Iteration {} of {}: loss={} adv_loss:{} " +
                               "distance={}")
                              .format(iteration, NUM_ITERATIONS,
                                loss, adv_loss, pert_norm.mean()))

                # import pdb; pdb.set_trace()

            for e, (dist, prd, ii) in enumerate(zip(pert_norm_np, pred_cls_np, points_np)):
                if dist < bestdist[e] and prd == attacked_label[e]:
                    bestdist[e] = dist
                    bestscore[e] = prd
                if dist < o_bestdist[e] and prd == attacked_label[e]:
                    o_bestdist[e] = dist
                    o_bestscore[e] = prd
                    o_bestattack[e] = ii
                    if args.model == 'lattice_cls':
                        o_record2D[e] = _[0][e].cpu().data.numpy()


                    if b_step[e] == -1:
                        b_step[e] = out_step
                        b_iter[e] = iteration
                # kaidong mods: no success yet, prepare to record least failure
                # only start record at the last binary step
                if out_step == BINARY_SEARCH_STEP-1 and o_bestscore[e] != attacked_label[e] and dist > o_failDist[e]:
                    o_failDist[e] = dist
                    o_failPred[e] = prd
                    o_leastFailAttack[e] = ii
                    if args.model == 'lattice_cls':
                        o_record2D[e] = _[0][e].cpu().data.numpy()

        # adjust the constant as needed
        for e in range(BATCH_SIZE):
            if bestscore[e]==attacked_label[e] and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e] :
                # success
                lower_bound[e] = max(lower_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
                #print('new result found!')
            else:
                # failure
                upper_bound[e] = min(upper_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2

                # kaidong
        #bestdist_prev=deepcopy(bestdist)

    log_string(" Successfully generated adversarial exampleson {} of {} instances." .format(sum(lower_bound > 0), BATCH_SIZE))
    log_string(' Best res on step %s iter %s. Train Mean Time: %fms, batch size: %d'% (str(b_step), str(b_iter), sum(train_timer)/len(train_timer), BATCH_SIZE))

    # for e in range(BATCH_SIZE):
    #     if o_bestscore[e] != attacked_label[e]:
    #         o_bestscore[e] = o_failPred[e]
    #         o_bestdist[e] = o_failDist[e]
    #         o_bestattack[e] = o_leastFailAttack[e]

    return o_bestdist, o_bestattack, o_bestscore, o_record2D, [o_leastFailAttack]




def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global CLASS_ATTACK

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
    atk_dir = experiment_dir.joinpath('attacked%s/' % (args.file_affix))
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

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = MODEL.get_adv_loss(num_class).cuda()

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
    mean_correct = []

    dist_list=[]

    if args.dataset == 'ScanNetCls':
        CLASS_ATTACK = [0, 4, 5, 6, 8, 12, 16]

    # lg_class = 0
    # for victim in range(1, num_class):
    # for victim in range(num_class):
    for victim in CLASS_ATTACK:
        if victim == args.target:
            continue

        if args.dataset == 'ModelNet40':
            DATA_PATH = '/dev/shm/data/modelnet40/'
            # DATA_PATH = 'data/modelnet40_normal_resampled/'
            # TRAIN_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='train',
            #                                              normal_channel=args.normal, victim=victim, target=args.target)
            TEST_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal, victim=victim, target=args.target)
        elif args.dataset == 'ScanNetCls':
            # TRAIN_PATH = '/scratch/kaidong/tf-point-cnn/data/test_scan_in/train_files.txt'
            TEST_PATH  = 'dump/scannet_test_data8316.npz'
            # TEST_PATH  = '/dev/shm/data/scannet/test_files.txt'
            # TEST_PATH  = '/scratch/kaidong/tf-point-cnn/data/test_scan_in/test_files.txt'
            # TRAIN_DATASET = ScanNetDataLoader(TRAIN_PATH, npoint=args.num_point, split='train',
            #                                                  normal_channel=args.normal)
            TEST_DATASET = AttackScanNetLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal, victim=victim, target=args.target)

        # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # test(classifier, trainDataLoader)

        batch_iterator = iter(testDataLoader)

        for j in range(args.num_sample//args.batch_size):
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            # images: b * num_pts * c
            dist, img, preds, img_2d, img_f = attack_one_batch(classifier, criterion, images, targets, args)
            # dist, img = attack_one_batch(classifier, criterion, images, targets, args, optimizer)
            dist_list.append(dist)

            # import pdb; pdb.set_trace()
            log_string("{}_{}_{} attacked.".format(victim,args.target,j))
            np.save(os.path.join(atk_dir, '{}_{}_{}_adv.npy' .format(victim,args.target,j)), img)
            np.save(os.path.join(atk_dir, '{}_{}_{}_adv_f.npy' .format(victim,args.target,j)), img_f[0])
            np.save(os.path.join(atk_dir, '{}_{}_{}_orig.npy' .format(victim,args.target,j)),images)#dump originial example for comparison
            np.save(os.path.join(atk_dir, '{}_{}_{}_pred.npy' .format(victim,args.target,j)),preds)
            if args.model == 'lattice_cls':
                np.save(os.path.join(atk_dir, '{}_{}_{}_2dimg.npy' .format(victim,args.target,j)), img_2d)

    # print('class num: ', num_class, ', class with enough images: ', lg_class)


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
    file_handler = logging.FileHandler('%s/%s_pert%s.txt' % (log_dir, args.model, args.file_affix))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
