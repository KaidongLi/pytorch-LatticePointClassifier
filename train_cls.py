from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.PCAModelNetDataLoader import PCAModelNetDataLoader
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

from utils import get_backbone 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

SCALE_LOW = 30
SCALE_UP  = 32


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--backbone', default='resnet50', help='backbone network name [default: resnet50]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--pca', action='store_true', default=False, help='Whether to use PCA to rotate data [default: False]')
    parser.add_argument('--dim', type=int, default=128, help='size of final 2d image [default: 128]')
    parser.add_argument('--dataset', default='ModelNet40', help='dataset name [default: ModelNet40]')
    parser.add_argument('--num_cls', type=int, default=40, help='Number of classes [default: 40]')
    parser.add_argument('--pre_rotation', action='store_true', default=False, help='Whether to use random rotation [default: False]')
    return parser.parse_args()

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,5))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        # timer
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        # timer ends
        for cat in np.unique(target.cpu()):

            # resolve tensor cannot be (target==cat) eq() to a numpy bug
            cat = cat.item()

            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
            class_acc[cat,3]+=classacc.item()
            class_acc[cat,4]+=[target==cat][0].cpu().sum().item()

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] = class_acc[:, 3]/(class_acc[:,4]+1e-08)
    # class_acc[:,2] =  class_acc[:,0]/ (class_acc[:,1]+1e-08)
    # import pdb; pdb.set_trace()
    class_acc = np.mean(class_acc[class_acc[:,4]!=0,2])
    # class_acc = np.mean(class_acc[class_acc[:,1]!=0,2])
    # class_acc = np.mean(class_acc[:,2].nonzero())
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    def log_only_string(str):
        logger.info(str)
        # print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    vis_dir = experiment_dir.joinpath('visual/')
    vis_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    if args.dataset == 'ModelNet40':
        DATA_PATH = 'data/modelnet40_normal_resampled/'

        if args.pca:
            TRAIN_DATASET = PCAModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                                        normal_channel=args.normal)
            TEST_DATASET = PCAModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)
        else:
            TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                                             normal_channel=args.normal)
            TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)
    elif args.dataset == 'ScanNetCls':
        assert (not args.pca), 'ScanNetCls with PCA is not supported yet'

        # TEST_PATH  = 'dump/scannet_test_data8316.npz'
        TEST_PATH  = 'data/scannet/test_files.txt'
        TRAIN_DATASET = ScanNetDataLoader(TRAIN_PATH, npoint=args.num_point, split='train',
                                                         normal_channel=args.normal)
        TEST_DATASET = ScanNetDataLoader(TEST_PATH, npoint=args.num_point, split='test',
                                                            normal_channel=args.normal)





    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)


    '''MODEL LOADING'''
    num_class = args.num_cls
    # num_class = 100
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))


    criterion = torch.nn.CrossEntropyLoss()
    # criterion = MODEL.get_loss().cuda()

    if args.model == 'lattice_cls':
        classifier = MODEL.get_model(num_class,
            normal_channel=args.normal,
            backbone=get_backbone(args.backbone, num_class, 1), s=args.dim*3).cuda()
    elif args.model == 'pointnet_cls':
        classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    elif args.model == 'lattice_cls_2ch':
        classifier = MODEL.get_model(num_class,
            normal_channel=args.normal,
            backbone=get_backbone(args.backbone, num_class, 2), s=args.dim*3).cuda()
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
        criterion = torch.nn.NLLLoss()
    # classifier = MODEL.get_model(num_class,
    #     normal_channel=args.normal,
    #     backbone=get_backbone(args.backbone, num_class, 1)).cuda()



    best_instance_acc = 0.0
    best_class_acc = 0.0


    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use saved model')

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)

            best_instance_acc = instance_acc
            best_class_acc = class_acc

            log_string('Load Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))

    except:
        log_string('No saved model, checking pretrain...')
        start_epoch = 0

        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/pretrain.pth')
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')

            with torch.no_grad():
                instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)

                best_instance_acc = instance_acc
                best_class_acc = class_acc

                log_string('Load Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))

        except:
            log_string('No existing model, start from scratch')


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        pbar = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        # for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        for batch_id, data in pbar:
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3], 0.8, 1.)
            # if points.max() >= SCALE_UP or points.min() <= -SCALE_UP:
            #     print('...')
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])

            points = torch.Tensor(points)

            if args.pre_rotation:
                points = points.cuda()
                points[:,:,:3] = provider.random_rotate_pc_3axis(points[:,:,:3])
                points = points.cpu()


            target = target[:, 0]

            points = points.transpose(2, 1)


            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            torch.autograd.set_detect_anomaly(True)

            pred, _ = classifier(points)
            loss = criterion(pred, target.long())

            if global_step % 10 == 0:
                log_only_string("Loss: %f" % loss)
                pbar.set_description("Loss: %f" % loss)


            if loss.isnan():
                aaa = 10
                import pdb; pdb.set_trace()
                for name, param in classifier.named_parameters():
                    if param.requires_grad and (param.data>aaa).sum()>0:
                        print(name, param.data)


            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
