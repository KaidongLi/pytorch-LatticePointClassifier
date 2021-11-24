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
from sklearn.cluster import DBSCAN

from pytorch3d.loss import chamfer
from utils import get_backbone

# from utils import get_cifar_training, get_cifar_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

DATA_PATH = 'data/modelnet40_normal_resampled/'

# SCALE_LOW = 2
# SCALE_UP  = 2

SCALE_LOW = 30
SCALE_UP  = 32

# SCALE_LOW = 150
# SCALE_UP  = 192

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
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
    parser.add_argument('--num_sample', type=int, default=5, help='number of samples per class [default: 5]')
    parser.add_argument('--file_affix', type=str, default='', help='log file/save folder affix')
    parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    parser.add_argument('--num_cls', type=int, default=40, help='Number of classes [default: 40]')

    parser.add_argument('--target', type=int, default=5, help='target class index')
    parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
    parser.add_argument('--upper_bound_weight', type=float, default=80, help='upper_bound value for the parameter lambda')
    parser.add_argument('--step', type=int, default=10, help='binary search step')
    parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')

    # for init points
    parser.add_argument('--init_pt_batch', type=int, default=8, help='batch size in initial point generation [default: 8]')
    parser.add_argument('--max_num', type=int,help='max number of points selected from the critical point set for clustering',default=16)
    parser.add_argument('--eps', type=float,default=0.2)
    parser.add_argument('--min_num', type=int,help='the min number for each cluster',default=3)
    # for clusters
    parser.add_argument('--mu', type=float, default=0.05, help='preset value for parameter mu')
    parser.add_argument('--add_num', type=int, default=512, help='number of added points [default: 512]')
    parser.add_argument('--num_cluster', type=int, default=3, help='cluster number')


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


def get_initial_cluster(args, images):

    att_critical = get_critical_points_simple(images, args)

    att_critical=[x[-args.max_num:,:] for x in att_critical]#get the points for DBSCAN clustering
    cri_all=np.concatenate(att_critical,axis=0)
    db = DBSCAN(eps=args.eps, min_samples=args.min_num)
    result=db.fit_predict(cri_all)  # the cluster/class label of each point
    filter_idx=result > -0.5 #get the index of non-outlier point
    # result with value > -0.5 (non-noise labels)
    result=result[filter_idx]
    # cri_all with DBSCAN value > -0.5
    cri_all=cri_all[filter_idx]
    l,c=np.unique(result,return_counts=True)
    i_idx=np.argsort(c)[-args.num_cluster:]
    l=l[i_idx]#get the label number for the largest NUM_CLUSTER clusters


    clustered_cri_list=[]

    for label in l:
        tmp=cri_all[result==label]#the point set belong to cluster "label"
        clustered_cri_list.append(tmp)

    return clustered_cri_list



def get_critical_points_simple(data, args):
    ####################################################
    ### get the critical point of the given point clouds
    ### data shape: BATCH_SIZE*NUM_POINT*3
    ### return : BATCH_SIZE*NUM_ADD*3
    #####################################################
    b = data.size(0)
    NUM_POINT = data.size(1)
    NUM_ADD = args.add_num

    critical_points=[]
    for i in range(b):
        #get the most important critical points if NUM_ADD < number of critical points
        #the importance is demtermined by counting how many elements in the global featrue vector is
        #contributed by one specific point

        rdn_idx = np.random.randint(0, NUM_POINT, 1024)
        idx,counts=np.unique(rdn_idx, return_counts=True)
        idx_idx=np.argsort(counts)
        # idx:     unique feature channel with max value
        # counts:  how many times the max value happened
        # idx_idx: arg in ascending order

        # idx is the idx occurance in ascending order
        idx=idx[idx_idx]
        points = data[i][idx]
        critical_points.append(points)
    # critical_points=np.stack(critical_points)
    return critical_points

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
    NUM_ADD = args.add_num

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
    o_best_dists = np.ones(shape=(BATCH_SIZE,3))

    init_points_list = []

    if args.dataset == 'ModelNet40':
        # DATA_PATH = '/dev/shm/data/modelnet40/'
        DATA_PATH = 'data/modelnet40_normal_resampled/'
        TARGET_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                        normal_channel=args.normal, victim=args.target, target=args.target)
    elif args.dataset == 'ScanNetCls':
        # TEST_PATH  = 'dump/scannet_test_data8316.npz'
        TEST_PATH  = 'data/scannet/test_files.txt'
        TARGET_DATASET = AttackScanNetLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                        normal_channel=args.normal, victim=args.target, target=args.target)

    targetDataLoader = torch.utils.data.DataLoader(TARGET_DATASET, batch_size=args.init_pt_batch, shuffle=False, num_workers=4)
    b_iter = iter(targetDataLoader)
    try:
        images, targets = next(b_iter)
    except StopIteration:
        b_iter = iter(data_loader)
        images, targets = next(b_iter)

    while len(init_points_list) < args.num_cluster:
        init_points_list.extend( get_initial_cluster(args, images)[:args.num_cluster-len(init_points_list)] )
    NUM_CLUSTER=len(init_points_list)
                                 #we force it to be always number of clusters
                                 #sometimes, there is only a limited number of cluster formed
                                 #so that DBSCAN may only get a NUM_CLUSTER smaller than the specified parameter
                                 #considering that, NUM_CLUSTER in this script is not a given parameter but obtained from the init point data


    if args.normal:
        o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT+NUM_ADD*NUM_CLUSTER,6))
        o_bestadd = np.ones(shape=(BATCH_SIZE, NUM_CLUSTER, NUM_ADD,6))
        o_failadd = np.ones(shape=(BATCH_SIZE, NUM_CLUSTER, NUM_ADD,6))

        o_leastFailAttack = np.ones(shape=(BATCH_SIZE,NUM_POINT+NUM_ADD*NUM_CLUSTER,6))
        o_record2D = np.ones(shape=(BATCH_SIZE, args.dim, args.dim, 3))
    else:
        o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT+NUM_ADD*NUM_CLUSTER,3))
        o_bestadd = np.ones(shape=(BATCH_SIZE, NUM_CLUSTER, NUM_ADD,3))
        o_failadd = np.ones(shape=(BATCH_SIZE, NUM_CLUSTER, NUM_ADD,3))

        o_leastFailAttack = np.ones(shape=(BATCH_SIZE,NUM_POINT+NUM_ADD*NUM_CLUSTER,3))
        o_record2D = np.ones(shape=(BATCH_SIZE, args.dim, args.dim, 1))



    o_failPred = [-1] * BATCH_SIZE
    o_failDist = [0] * BATCH_SIZE

    train_timer = []

    #make sure each element(cluster) in init_point_list is in shape of BATCH_SIZE*NUM_ADD*3
    for i in range(NUM_CLUSTER):
        tmp=init_points_list[i]
        if len(tmp) >= NUM_ADD:
            tmp=tmp[-NUM_ADD:]
        else:
            tmp=np.tile(tmp,[NUM_ADD // len(tmp),1])
            if NUM_ADD % len(tmp) != 0:
                tmp=np.concatenate([tmp,tmp[- (NUM_ADD % len(tmp)) : ]],axis=0)
        tmp=np.expand_dims(tmp,axis=0)
        init_points_list[i]=np.tile(tmp,[BATCH_SIZE,1,1])
        #print(init_points_list[i].shape)

    b_step = [-1] * BATCH_SIZE
    b_iter = [-1] * BATCH_SIZE
    for out_step in range(BINARY_SEARCH_STEP):
        log_string((" Step {} of {}")
                              .format(out_step, BINARY_SEARCH_STEP))

        pert_list = []
        optim_list = []

        # INIT_STD = 0.01
        INIT_STD = 1e-7
        for i in range(NUM_CLUSTER):
            if args.normal:
                pert = torch.normal(0, INIT_STD, size=(BATCH_SIZE,NUM_ADD,6), requires_grad=True, device='cuda')
            else:
                pert = torch.normal(0, INIT_STD, size=(BATCH_SIZE,NUM_ADD,3), requires_grad=True, device='cuda')

            optimizer = torch.optim.Adam(
                # classifier.parameters(),
                [pert], # + list(classifier.parameters()),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
            pert_list.append(pert)
            optim_list.append(optimizer)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        # bestdist: starting with norm 1e10,
        #           recording lowest norm of successful perturbation
        # bestscore: starting with -1
        #            recording the successful attacked label
        bestdist = [1e10] * BATCH_SIZE
        bestscore = [-1] * BATCH_SIZE


        prev = 1e6

        for iteration in range(NUM_ITERATIONS):


            # add perturbation
            points = points_ori
            points = points.cuda()

            farthest_loss_list=[]
            nndistance_loss_list=[]
            point_added_list=[]

            for i in range(NUM_CLUSTER):
                optim_list[i].zero_grad()
                pert = pert_list[i]
                point_added = torch.Tensor(init_points_list[i]).cuda()
                point_added[:, :, 0:3] = point_added[:, :, 0:3] + pert
                point_added_list.append(point_added)
                # initial_point_pl_list.append(initial_point_pl)
                # initial_point_pl_list: initial points from target class
                # point_added_list: list of final points

                #farthest distance loss
                # import pdb; pdb.set_trace()
                point_expand=point_added.unsqueeze(dim=1)
                point_tranpose=point_expand.permute(0,2,1,3)

                # delta_matrix: below/above diagnal line, representing deltas between each other
                delta_matrix = point_expand - point_tranpose + 1e-5 #avoid division by zero
                norm_matrix = torch.linalg.norm(delta_matrix, dim=3)
                farthest_loss = norm_matrix.amax(dim=(1, 2))
                farthest_loss_list.append(farthest_loss)

                #Chamfer/Hausdorff
                dists_forward = chamfer.chamfer_distance(point_added, points, batch_reduction=None)[0]
                # dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(point_added,pointclouds_pl)
                # dists_forward = tf.reduce_mean(dists_forward,axis=1)#Chamfer
                nndistance_loss_list.append(dists_forward)

            points_all = torch.cat(([points]+point_added_list), dim=1)
            points_cls = points_all.transpose(2, 1)
            points_cls, attacked_label = points_cls.cuda(), attacked_label.cuda()



            # classifier = classifier.train()

            st = datetime.datetime.now().timestamp()
            pred, _ = classifier(points_cls)
            adv_loss = criterion(pred, attacked_label.long())



            for i in range(NUM_CLUSTER):
                l_cluster = (WEIGHT* (farthest_loss_list[i] + args.mu * nndistance_loss_list[i])).mean()
                loss = adv_loss + l_cluster

                if i == NUM_CLUSTER-1:
                	loss.backward()
                else:
                	loss.backward(retain_graph=True)
                optim_list[i].step()

            st = datetime.datetime.now().timestamp() - st
            train_timer.append(st)

            pred_cls_np = pred.max(dim=1)[1].cpu().data.numpy()

            farthest_loss_np = (sum(farthest_loss_list)/len(farthest_loss_list)).cpu().data.numpy()
            nndistance_loss_np = (sum(nndistance_loss_list)/len(nndistance_loss_list)).cpu().data.numpy()

            # import pdb; pdb.set_trace()
            points_np = points_all.cpu().data.numpy()
            points_add_np = np.stack([x.cpu().data for x in point_added_list]).transpose((1, 0, 2, 3))
            init_add_pts_np = np.stack([x for x in init_points_list]).transpose((1, 0, 2, 3))

            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                # print(WEIGHT)
                log_string((" Iteration {} of {}: loss={} adv_loss:{} " +
                               "distance={},{}")
                              .format(iteration, NUM_ITERATIONS,
                                loss, adv_loss,
                                farthest_loss_np.mean(), nndistance_loss_np.mean() ))


            for e, (dist_f, dist_h, prd, ii, ii_add) in enumerate(zip(farthest_loss_np, nndistance_loss_np, pred_cls_np, points_np, points_add_np)):
                dist = dist_h*args.mu + dist_f
                if dist < bestdist[e] and prd == attacked_label[e]:
                    bestdist[e] = dist
                    bestscore[e] = prd
                if dist < o_bestdist[e] and prd == attacked_label[e]:
                    o_best_dists[e] = [dist_f, dist_h, dist]
                    o_bestdist[e] = dist
                    o_bestscore[e] = prd
                    o_bestattack[e] = ii
                    o_bestadd[e] = ii_add
                    if args.model == 'lattice_cls':
                        o_record2D[e] = _[0][e].cpu().data.numpy()

                    if b_step[e] == -1:
                        b_step[e] = out_step
                        b_iter[e] = iteration
                # no success yet, prepare to record least failure
                # only start record at the last binary step
                if out_step == BINARY_SEARCH_STEP-1 and o_bestscore[e] != attacked_label[e] and dist > o_failDist[e]:
                    o_best_dists[e] = [dist_f, dist_h, dist]
                    o_failDist[e] = dist
                    o_failPred[e] = prd
                    o_leastFailAttack[e] = ii
                    o_failadd[e] = ii_add
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
        #bestdist_prev=deepcopy(bestdist)

    log_string(" Successfully generated adversarial exampleson {} of {} instances." .format(sum(lower_bound > 0), BATCH_SIZE))
    log_string('Best res on step %s iter %s. Train Mean Time: %fms, batch size: %d'% (str(b_step), str(b_iter), sum(train_timer)/len(train_timer), BATCH_SIZE))

    return o_bestdist, o_bestattack, o_bestscore, o_bestadd, [init_add_pts_np, o_failadd, o_record2D, o_best_dists]




def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    global CLASS_ATTACK

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = experiment_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)
    atk_dir = experiment_dir.joinpath('attacked_cluster%s/' % (args.file_affix))
    atk_dir.mkdir(exist_ok=True)

    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    '''MODEL LOADING'''
    num_class = args.num_cls
    # num_class = 100
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))


    if args.model == 'lattice_cls' or args.model == 'lattice_cls_test':
        classifier = MODEL.get_model(num_class,
            normal_channel=args.normal,
            backbone=get_backbone(args.backbone, num_class, 1), s=args.dim*3).cuda()
    elif args.model == 'pointnet_ddn':
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
    classifier.eval()

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    dist_list=[]

    if args.dataset == 'ScanNetCls':
        CLASS_ATTACK = [0, 4, 5, 6, 8, 12, 16]

    # for victim in range(1, num_class):
    # for victim in range(num_class):
    for victim in CLASS_ATTACK:
        if victim == args.target:
            continue

        if args.dataset == 'ModelNet40':
            # DATA_PATH = '/dev/shm/data/modelnet40/'
            DATA_PATH = 'data/modelnet40_normal_resampled/'
            TEST_DATASET = AttackModelNetLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal, victim=victim, target=args.target)
        elif args.dataset == 'ScanNetCls':
            # TEST_PATH  = 'dump/scannet_test_data8316.npz'
            TEST_PATH  = 'data/scannet/test_files.txt'
            TEST_DATASET = AttackScanNetLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal, victim=victim, target=args.target)

        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

        batch_iterator = iter(testDataLoader)

        for j in range(args.num_sample//args.batch_size):
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)

            # images: b * num_pts * c
            dist, img, preds, adds, info_add = attack_one_batch(classifier, criterion, images, targets, args)
            # dist, img = attack_one_batch(classifier, criterion, images, targets, args, optimizer)
            dist_list.append(dist)

            log_string("{}_{}_{} attacked.".format(victim,args.target,j))
            np.save(os.path.join(atk_dir, '{}_{}_{}_cluster.npy' .format(victim,args.target,j)), adds)
            np.save(os.path.join(atk_dir, '{}_{}_{}_cluster_f.npy' .format(victim,args.target,j)), info_add[1])
            np.save(os.path.join(atk_dir, '{}_{}_{}_orig.npy' .format(victim,args.target,j)),images)#dump originial example for comparison
            np.save(os.path.join(atk_dir, '{}_{}_{}_cluster_orig.npy' .format(victim,args.target,j)), info_add[0])

            np.save(os.path.join(atk_dir, '{}_{}_{}_pred.npy' .format(victim,args.target,j)),preds)
            np.save(os.path.join(atk_dir, '{}_{}_{}_dists.npy' .format(victim,args.target,j)), info_add[3])
            if args.model == 'lattice_cls':
                np.save(os.path.join(atk_dir, '{}_{}_{}_2dimg.npy' .format(victim,args.target,j)),info_add[2])


if __name__ == '__main__':
    args = parse_args()

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('attacks')
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
    file_handler = logging.FileHandler('%s/%s_cluster%s.txt' % (log_dir, args.model, args.file_affix))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    main(args)
