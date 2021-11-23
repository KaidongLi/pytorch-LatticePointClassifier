import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
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

class AttackModelNetLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, victim=5, target=0):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel
        self.cls_idx = victim
        self.target = target

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() 
                                for line in open(os.path.join(self.root, 'modelnet40_train.txt')) 
                                if '_'.join(line.rstrip().split('_')[0:-1]) == self.cat[victim]
                             ]
        shape_ids['test'] = [line.rstrip() 
                                for line in open(os.path.join(self.root, 'modelnet40_test.txt')) 
                                if '_'.join(line.rstrip().split('_')[0:-1]) == self.cat[victim]
                            ]

        assert (split == 'train' or split == 'test')
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [os.path.join(self.root, self.cat[victim], shape_ids[split][i]) + '.txt' for i
                         in range(len(shape_ids[split]))]
        print('The size of %s [%d]%s data is %d'%(split, victim, self.cat[victim], len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.target
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn, delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

    def get_class(self):
        return self.cat[self.cls_idx]




if __name__ == '__main__':
    import torch

    data = AttackModelNetLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)