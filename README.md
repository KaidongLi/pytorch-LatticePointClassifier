# Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients

Created by [Kaidong Li](https://www.linkedin.com/in/KaidongLi/), [Ziming Zhang](https://zhang-vislab.github.io/), [Cuncong Zhong](https://cbb.ittc.ku.edu/index.html), [Guanghui Wang](https://www.cs.ryerson.ca/~wangcs/)

This repo is the official implementation of Lattice Point Classifier (LPC) for our CVPR 2022 paper [**Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients**](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Robust_Structured_Declarative_Classifiers_for_3D_Point_Clouds_Defending_Adversarial_CVPR_2022_paper.pdf). 

We study its classification and defense performances against some state-of-the-art adversarial attacks and compare the results with state-of-the-art defenses. 

We also collect the implementations of these 3D point cloud attacks ([FGSM](https://arxiv.org/pdf/1412.6572.pdf), [JGBA](https://dl.acm.org/doi/pdf/10.1145/3394171.3413875?casa_token=fk6eajSNqSwAAAAA:rqBCH1XnUfVdUrIFOL7nzMQ_gaEbLYFvQqs8IU9BABW7ge28AsVCtTILnancYZXKM_Z3EpOUVN1nAg), [Perturbation Add Cluster and Object Attacks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiang_Generating_3D_Adversarial_Point_Clouds_CVPR_2019_paper.pdf), [CTRI](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_On_Isometry_Robustness_of_Deep_3D_Point_Cloud_Models_Under_CVPR_2020_paper.pdf)) and defenses ([DUP-Net](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_DUP-Net_Denoiser_and_Upsampler_Network_for_3D_Adversarial_Point_Clouds_ICCV_2019_paper.pdf), [IF-Defense](https://arxiv.org/pdf/2010.05272.pdf) and [RPL](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9355027&casa_token=9kl6Y0dQ330AAAAA:6iZ47UV7veR7HvRXlkALeenfsiwe7qoqX3euYp-etvUj4Kw7xBAmUgD5p04bTVJeiZ2APXYSyXA&tag=1)) in this repository, in the hope of serving as a reference point for future research.

Code for *Robust 3D Point Clouds Classification based on Declarative Defenders* is currently in `lp_seg` branch.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{li2022robust,
  title={Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients},
  author={Li, Kaidong and Zhang, Ziming and Zhong, Cuncong and Wang, Guanghui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15294--15304},
  year={2022}
}
```

## Environments
We run our experiments under:

Python 3.6.9 <br>
GCC 6.3.0 <br>
CUDA 10.1.105 <br>
torch 1.7.1 <br>
pytorch3d 0.4.0 <br>
open3d 0.9.0.0

And other packages might be needed.

## Classification
### Data Preparation
  *  ModelNet40
     
     Download alignment **ModelNet40** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

  *  ScanNet
     
     Please refer to [here](http://www.scan-net.org/) for downloading ScanNet data.
  
     Refer to [here](https://github.com/yangyanli/PointCNN) for converting the data to classification task and save in `data/scannet/`.
     
      * Once receiving ScanNet download script from [official site](http://www.scan-net.org/), download the v2 "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json" and label map files
      * Once download completes, run `extract_scannet_objs.py` and `prepare_scannet_cls_data.py` in [this repo](https://github.com/yangyanli/PointCNN) to get classification data as suggested.
      * During extract, there are 5 mismatched class name pairs ("trash", "trash can"), ("bathtub", "tub"), ("shelf", "bookshelf"), ("keyboard", "computer keyboard") and ("tv", "monitor or tv"). We modify `extract_scannet_objs.py` to link these pairs and change to the former name.

### Run Classification
```
# train LPC with EfficientNet-B5 as backbone and scaling factor 456 on ScanNet dataset
python train_cls.py --dataset ScanNetCls --num_cls 17 --model lattice_cls --backbone efficientnet-b5 --dim 456 --learning_rate 0.0001 --log_dir lpc_eff_456 --batch_size 4

# train PointNet on ModelNet40 dataset
python train_cls.py --model pointnet_cls --dataset ModelNet40 --num_cls 40 --log_dir pointnet_cls --batch_size 24
```
Refer to `train_cls.py` for detailed options. Best model from training will be saved in `log/classification/***/checkpoints/best_model.pth`.

To switch between binarized weights and barycentric weights, it is currently done manually. Find the following lines in `models/lattice_cls.py`. It is binarized weights. By commenting out all the lines, the LPC models use barycentric weights. 
```
if not self.normal_channel:
    # next two lines are for cutoff weights
    # cutoff = filter_2d[filter_2d>0].mean() * 2
    # filter_2d[filter_2d>cutoff] = cutoff
    # next line is for binarized weights
    filter_2d[filter_2d>0] = 1.0
```

DUP-Net and IF-Defense have similar architecture of their specific module plus PointNet. We can use the previously trained PointNet model. Download DUP-Net specific pretrained model `pu-in_1024-up_4.pth` from [here](https://github.com/Wuziyi616/IF-Defense/blob/main/baselines/defense/DUP_Net/pu-in_1024-up_4.pth) and save in `dump/`.

PointNet with RPL can be trained using our code. But we also recommend using [this implementation](https://github.com/anucvml/ddn/tree/master/apps/classification/pointcloud) for faster result and moving the trained model here.

### Accuracy
| Model | ModelNet40 | ScanNet|
|--|--|--|
| PointNet              | **90.15** | **84.61** |
| DUP-Net               | 89.30 | 83.62 |
| IF-Defense            | 87.60 | 80.19 |
| PointNet RPL          | 84.76 | 76.02 |
| LPC w/ vgg16          | 88.65 | --    |
| LPC w/ resnet50       | 88.90 | --    |
| LPC w/ efficientnet-b5| 89.51 | 83.16 |

## Attack
### Model Preparation
* #### FGSM, JGBA, Perturbation
  Copy model folder from classifcation result `log/classification/` to `log/perturbation/`
* #### Add, Cluster, Object Attacks
  Copy model folder from classifcation result `log/classification/` to `log/attacks/`
* #### CTRI
  Copy saved `.pth` model from classifcation result `log/classification/***/checkpoints/best_model.pth` to `log_ctri_attack/`
* #### DUP-Net
  Download pretrained model `pu-in_1024-up_4.pth` from [here](https://github.com/Wuziyi616/IF-Defense/blob/main/baselines/defense/DUP_Net/pu-in_1024-up_4.pth) to `dump/` and `log_ctri_attack/`
### Run
Be careful and match model parameters with the setups during training process
* #### FGSM/JGBA attack (on PointNet, PointNet w/ RPL and LPC, except DUP-Net and IF-Defense): 
```
# FGSM/JGBA attack on PointNet model on ModelNet40
python pert_JGBA_attack.py --attack [FGSM/JGBA] --model pointnet_cls --dataset ModelNet40 --num_cls 40 --log_dir [folder_name_under_perturbation]
```
* #### FGSM/JGBA attack (for both DUP-Net and IF-Defense): 
```
# FGSM/JGBA attack on ScanNet
python pert_JGBA_attack_SOR.py --attack [FGSM/JGBA] --dataset ScanNetCls --num_cls 17 --log_dir [folder_name_under_perturbation]
```
* #### Perturbation, add, cluster and object attacks (on PointNet, PointNet w/ RPL and LPC, except DUP-Net and IF-Defense): 
```
# Perturbation attack on PointNet on ModelNet40
python perturbation_attack.py --batch_size 6 --learning_rate 0.01 --target 0 --model pointnet_cls --log_dir [folder_name_under_perturbation]
```
```
# Add attack on RPL on ScanNet, adding 60 independent points
python independent_attack.py --batch_size 6 --learning_rate 0.01 --target 0 --model pointnet_ddn --add_num 60 --log_dir [folder_name_under_attacks] --dataset ScanNetCls --num_cls 17
```
```
# Cluster attack on LPC with vgg16 and scaling factor 512 on ModelNet40, adding 3 clusters of 32 points
python cluster_attack.py --batch_size 6 --learning_rate 0.01 --target 0 --model lattice_cls --backbone vgg16 --dim 512 --add_num 32 --num_cluster 3 --log_dir [folder_name_under_attacks] --eps 0.11 --mu 0.1 --init_pt_batch 8 --max_num 32
```
Download the object  `airplane.npy` to be attached from [here](https://github.com/xiangchong1/3d-adv-pc/blob/master/data/airplane.npy), and copy it to `data/`.
```
# Object attack on PointNet on ModelNet40, adding 3 objects of 64 points
python object_attack.py --batch_size 6 --learning_rate 0.01 --target 0 --model pointnet_cls --add_num 64 --num_cluster 3 --log_dir [folder_name_under_attacks] --eps 0.11 --mu 0.1 --init_pt_batch 8 --max_num 32
```
The above attack results (e.g. adversarial samples) will be saved in a folder `attacked_[attack_name]` under `$log_dir`. For DUP-Net and IF-Defense, perturbation, add, cluster and object attacks are done on clean PointNet. Then package the attack results under the `attacked_[attack_name]` folder using `data_analysis/sum-[attack_name].py` and move to [here](https://github.com/Wuziyi616/IF-Defense) for defense performance analysis. 

* #### CTRI attack (on PointNet, DUP-Net, PointNet w/ RPL and LPC, except IF-Defense): 
```
# CTRI attack on DUP-Net on ModelNet40
python ctri_attack.py --data modelnet40 --model dupnet --model_path [pretrain_name_under_log_ctri_attack] --num_points 1024
```

## Reference Implementations
[laoreja/HPLFlowNet](https://github.com/laoreja/HPLFlowNet)<br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
[Wuziyi616/IF-Defense](https://github.com/Wuziyi616/IF-Defense) <br>
[xiangchong1/3d-adv-pc](https://github.com/xiangchong1/3d-adv-pc) <br>
[yangyanli/PointCNN](https://github.com/yangyanli/PointCNN) <br>
[pytorch/fgsm](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html) <br>
[machengcheng2016/JGBA-pointcloud-attack](https://github.com/machengcheng2016/JGBA-pointcloud-attack) <br>
[anucvml/ddn](https://github.com/anucvml/ddn) <br>
[skywalker6174/3d-isometry-robust](https://github.com/skywalker6174/3d-isometry-robust)
