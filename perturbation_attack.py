"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.AttackModelNetLoader import AttackModelNetLoader
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

# from utils import get_cifar_training, get_cifar_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# SCALE_LOW = 2
# SCALE_UP  = 2

SCALE_LOW = 30
SCALE_UP  = 32

# SCALE_LOW = 150
# SCALE_UP  = 192

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
    
    parser.add_argument('--target', type=int, default=5, help='target class index')
    parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
    parser.add_argument('--upper_bound_weight', type=float, default=80, help='upper_bound value for the parameter lambda')
    parser.add_argument('--step', type=int, default=10, help='binary search step')
    parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')


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
    o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT,6))

    for out_step in range(BINARY_SEARCH_STEP):

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
        pert = torch.normal(0, 0.01, size=(BATCH_SIZE,NUM_POINT,3), requires_grad=True, device='cuda')

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

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



            points_cls = points.transpose(2, 1)
            points_cls, attacked_label = points_cls.cuda(), attacked_label.cuda()

            
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, _ = classifier(points_cls)
            adv_loss = criterion(pred, attacked_label.long())

            #perturbation l2 constraint
            pert_norm = torch.square(pert).sum(dim=2).sum(dim=1).sqrt().cuda()

            norm_loss = (WEIGHT * pert_norm).mean()

            loss = adv_loss + norm_loss
            loss.backward()
            optimizer.step()

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
        #bestdist_prev=deepcopy(bestdist)

    log_string(" Successfully generated adversarial exampleson {} of {} instances." .format(sum(lower_bound > 0), BATCH_SIZE))
    return o_bestdist, o_bestattack




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
    atk_dir = experiment_dir.joinpath('attacked/')
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
    DATA_PATH = 'data/modelnet40_normal_resampled/'

    # TRAIN_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='train',
    #                                                  normal_channel=args.normal)
    # TEST_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='test',
    #                                                 normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)


    # trainDataLoader = get_cifar_training(
    #     (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
    #     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    #     num_workers=4, batch_size=args.batch_size, shuffle=True
    # )

    # testDataLoader = get_cifar_test(
    #     (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
    #     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    #     num_workers=4, batch_size=args.batch_size, shuffle=False
    # )

    '''MODEL LOADING'''
    num_class = 40
    # num_class = 100
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    # classifier = MODEL.get_model(num_class,normal_channel=args.normal,s=128*3).cuda()
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


    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []





    # if args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(
    #         classifier.parameters(),
    #         lr=args.learning_rate,
    #         betas=(0.9, 0.999),
    #         eps=1e-08,
    #         weight_decay=args.decay_rate
    #     )
    # else:
    #     optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    dist_list=[]
    # for victim in range(1, num_class):
    for victim in range(num_class):
        if victim == args.target:
            continue

        # TRAIN_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='train',
        #                                              normal_channel=args.normal, victim=victim, target=args.target)
        TEST_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                        normal_channel=args.normal, victim=victim, target=args.target)

        # skip classes with small amount of examples
        if len(TEST_DATASET) < 50:
            continue

        # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # test(classifier, trainDataLoader)

        batch_iterator = iter(testDataLoader)

        for j in range(25//args.batch_size):
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            # images: b * num_pts * c
            dist, img = attack_one_batch(classifier, criterion, images, targets, args)
            # dist, img = attack_one_batch(classifier, criterion, images, targets, args, optimizer)
            dist_list.append(dist)
            np.save(os.path.join(atk_dir, '{}_{}_{}_adv.npy' .format(victim,args.target,j)), img)
            np.save(os.path.join(atk_dir, '{}_{}_{}_orig.npy' .format(victim,args.target,j)),images)#dump originial example for comparison
            





    '''TRANING'''
    # logger.info('Start training...')
    # for epoch in range(start_epoch,args.epoch):
    #     log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        # kaidong debug test
        # with torch.no_grad():
        #     instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)

        #     if (instance_acc >= best_instance_acc):
        #         best_instance_acc = instance_acc
        #         best_epoch = epoch + 1

        #     if (class_acc >= best_class_acc):
        #         best_class_acc = class_acc
        #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
        #     log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

        # scheduler.step()
        # pbar = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        # # for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        # for batch_id, data in pbar:
        #     points, target = data
        #     points = points.data.numpy()
        #     points = provider.random_point_dropout(points)
        #     points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3], 0.8, 1.)
        #     # if points.max() >= SCALE_UP or points.min() <= -SCALE_UP:
        #     #     print('...')

        #     #     # kaidong test
        #     #     import pdb; pdb.set_trace()

        #     points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
        #     points = torch.Tensor(points)
        #     target = target[:, 0]

        #     points = points.transpose(2, 1)


        #     points, target = points.cuda(), target.cuda()
        #     optimizer.zero_grad()

        #     classifier = classifier.train()

        #     pred, _ = classifier(points)
        #     loss = criterion(pred, target.long())

        #     if global_step % 10 == 0:
        #         log_only_string("Loss: %f" % loss)
        #         pbar.set_description("Loss: %f" % loss)

            # np.save(os.path.join(log_dir, 'l_'+str(global_step)+'_ori'),
            #     # points.permute(0, 2, 3, 1).cpu().data.numpy()
            #     points.cpu().data.numpy()
            # )
            # np.save(os.path.join(log_dir, 'l_'+str(global_step)+'_3dlat'),
            #     _[2].cpu().data.numpy()
            # )
            # np.save(os.path.join(log_dir, 'l_'+str(global_step)+'_lat'),
            #     _[1].permute(0, 2, 3, 1).cpu().data.numpy()
            # )
            # np.save(os.path.join(log_dir, 'l_'+str(global_step)+'_sparse_lat'),
            #     _[0].permute(0, 2, 3, 1).cpu().data.numpy()
            # )



            # # kaidong test
            # import pdb; pdb.set_trace()
            


            # pred, trans_feat = classifier(points)
            # loss = criterion(pred, target.long(), trans_feat)
    #         pred_choice = pred.data.max(1)[1]
    #         correct = pred_choice.eq(target.long().data).cpu().sum()
    #         mean_correct.append(correct.item() / float(points.size()[0]))
    #         loss.backward()
    #         optimizer.step()
    #         global_step += 1

    #         # kaidong test
    #         # if global_step > 5:
    #         #     global_step = 0
    #         #     break

    #     train_instance_acc = np.mean(mean_correct)
    #     log_string('Train Instance Accuracy: %f' % train_instance_acc)


    #     with torch.no_grad():
    #         instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)

    #         if (instance_acc >= best_instance_acc):
    #             best_instance_acc = instance_acc
    #             best_epoch = epoch + 1

    #         if (class_acc >= best_class_acc):
    #             best_class_acc = class_acc
    #         log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
    #         log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

    #         if (instance_acc >= best_instance_acc):
    #             logger.info('Save model...')
    #             savepath = str(checkpoints_dir) + '/best_model.pth'
    #             log_string('Saving at %s'% savepath)
    #             state = {
    #                 'epoch': best_epoch,
    #                 'instance_acc': instance_acc,
    #                 'class_acc': class_acc,
    #                 'model_state_dict': classifier.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #             }
    #             torch.save(state, savepath)
    #         global_epoch += 1

    # logger.info('End of training...')

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
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
