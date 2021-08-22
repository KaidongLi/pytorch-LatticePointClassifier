import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import torch
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








class RotatePCA(object):
    # def __init__(self):
    #     super(RotatePCA, self).__init__()

    def __call__(self, pc, features=None):
        with torch.no_grad():
            y = torch.from_numpy(pc).unsqueeze(dim=0)
            pc1 = y.transpose(2, 1)

            # kaidong: get the least important direction
            aa, bb, cc = torch.pca_lowrank(y)
            vec_low = cc[:, :, -1]

            trans_norm = torch.FloatTensor([1, 1, 1]) / torch.tensor(3.).sqrt()
            trans_norm = trans_norm.expand(pc1.size(0), -1)

            # print('vec_low size: ', vec_low.size())
            # print('trans_norm size: ', trans_norm.size())

            # cross product of least important vector and lattice transform plane vector
            # the cross product is also the rotation axis
            # the value is the sin value
            rot_axis = torch.cross(vec_low, trans_norm, dim=1)
            sin_rot = rot_axis.norm(dim=1)
            rot_axis = rot_axis / sin_rot[:, None]

            # dot product
            # also cos value
            cos_rot = (vec_low * trans_norm).sum(dim=1)

            # Rodrigues' rotation formula
            rot_pts = pc1 * cos_rot[:, None, None] 
            rot_pts += torch.cross(rot_axis[:, :, None].expand(pc1.shape), pc1) * sin_rot[:, None, None] 
            rot_pts += rot_axis[:, :, None] * (rot_axis[:, :, None] * pc1).sum(dim=1, keepdim=True) * (1 - cos_rot)[:, None, None]

            rot_pts = rot_pts.transpose(2, 1).squeeze(dim=0)
            rot_pts_np = rot_pts.numpy()

            return rot_pts_np #, [bb.squeeze(dim=0), cc.squeeze(dim=0)]


    def __repr__(self):
        format_string = self.__class__.__name__ + '\n(Rotate least important axis to normal\n'
        format_string += ')'
        return format_string






class PCAModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        self.rotate_pca = RotatePCA()

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            point_set = self.rotate_pca(point_set)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)




if __name__ == '__main__':
    import torch

    data = PCAModelNetDataLoader('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)