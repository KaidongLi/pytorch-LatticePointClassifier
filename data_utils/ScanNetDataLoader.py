import numpy as np
import h5py
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


# size of pc: B, N, 3
def pc_normalize(pc):
    centroid = np.mean(pc, axis=1)
    pc = pc - centroid[:, None, :]
    m = np.max(np.sqrt(np.sum(pc**2, axis=2)), axis=1)
    pc = pc / m[:, None, None]
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ScanNetDataLoader(Dataset):
    def __init__(self, data_file,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        # self.root = data_file
        self.npoints = npoint
        self.normal_channel = normal_channel

        assert (split == 'train' or split == 'test')


        if data_file.endswith('.txt'):
            points = []
            labels = []
            folder = os.path.dirname(data_file)
            for line in open(data_file):
                filename = os.path.basename(line.rstrip())
                data = h5py.File(os.path.join(folder, filename))
                if 'normal' in data:
                    pts = np.concatenate([data['data'][...], data['normal'][...]], axis=-1).astype(np.float32)
                else:
                    pts = data['data'][...].astype(np.float32)
                
                idx = []
                for i in range(pts.shape[0]):
                    ix = np.random.choice(pts.shape[1], self.npoints, replace=False)
                    idx.append(ix[None])
                idx = np.concatenate(idx, axis=0)

                if not self.normal_channel:
                    pts = pts[:, :, :3]
                # delete color channels if desired
                pts = np.take_along_axis(pts, idx[:, :, None], axis=1)

                points.append(pts)
                labels.append(np.squeeze(data['label'][:]).astype(np.int64))


            points, labels = np.concatenate(points, axis=0), np.concatenate(labels, axis=0)
            points[:, :, :3] = pc_normalize(points[:, :, :3])
        elif data_file.endswith('.npz'):
            npz = np.load(data_file, allow_pickle=True)
            points = npz['test_pc']
            labels = np.squeeze(npz['test_label'])
        else:
            assert False, 'wrong data file'

        if split == 'train':
            (points, labels) = grouped_shuffle((points, labels))

        print('The size of %s data is %d'%(split, labels.shape[0]))
        self.pts_cld = points
        self.label = labels[:, None]




    def __len__(self):
        return self.label.shape[0]

    def _get_item(self, index):

        return self.pts_cld[index], self.label[index]

    def __getitem__(self, index):
        return self._get_item(index)



def grouped_shuffle(inputs):
    for idx in range(len(inputs) - 1):
        assert (len(inputs[idx]) == len(inputs[idx + 1]))

    shuffle_indices = np.arange(inputs[0].shape[0])
    np.random.shuffle(shuffle_indices)
    outputs = []
    for idx in range(len(inputs)):
        outputs.append(inputs[idx][shuffle_indices, ...])
    return outputs


if __name__ == '__main__':
    import torch

    import argparse
    import pdb

    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--train_file', type=str, default='', help='train data file list')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    args = parser.parse_args()

    data = ScanNetDataLoader(args.train_file, split='train', uniform=False, normal_channel=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)
        pdb.set_trace()
        break

