# GKGNet
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gkgnet-group-k-nearest-neighbor-based-graph/multi-label-classification-on-pascal-voc-2007)](https://paperswithcode.com/sota/multi-label-classification-on-pascal-voc-2007?p=gkgnet-group-k-nearest-neighbor-based-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gkgnet-group-k-nearest-neighbor-based-graph/multi-label-classification-on-ms-coco)](https://paperswithcode.com/sota/multi-label-classification-on-ms-coco?p=gkgnet-group-k-nearest-neighbor-based-graph)
## Introduction

This repo contains the official PyTorch implementation of our ECCV'2024 paper
[GKGNet: Group K-Nearest Neighbor based Graph Convolutional Network for Multi-Label Image Recognition](https://arxiv.org/abs/2308.14378).

<div align="center"><img src="assets/arch.png" width="800"></div>


## Quick Start Guide

### 1. Clone the Repository
To get started, clone the repository using the following commands:
```sh
git clone https://github.com/jin-s13/GKGNet.git
cd GKGNet
```

### 2. Environment Setup
Set up the required environment with the following commands:
```sh
conda create -n mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmlab
pip install openmim
mim install mmcv-full==1.5.0
pip install -e .
```

### 3. Data Preparation
Prepare the required data by downloading the [MS-COCO 2014](https://cocodataset.org/#download) dataset. The file structure should look like this:
```sh
-0data
    -coco
        -train.data
        -val_test.data
        -annotations
            -instances_train2014.json
            -instances_val2014.json
        -train2014
            -COCO_train2014_000000000009.jpg
            -COCO_train2014_000000000025.jpg
            ...
        -val2014
            -COCO_val2014_000000000042.jpg
            -COCO_val2014_000000000073.jpg
            ...
-GKGNet
    -configs
    -checkpoint
      -pvig_s_82.1.pth.tar
    -tools
    ...
```
You can obtain `train.data` and `val_test.data` from the `coco_multi_label_annos` directory. The pretrained backbones on ImageNet can be downloaded from [Vig](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_s_82.1.pth.tar):
```sh
mkdir checkpoint
wget https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_s_82.1.pth.tar
mv pvig_s_82.1.pth.tar checkpoint/
```

### 4. Training
To train the model, use one of the following commands:

#### Single Process
```sh
python tools/train.py configs/gkgnet/gkgnet_coco_576.py
```

#### Multi Process
```sh
bash tools/dist_train.sh configs/gkgnet/gkgnet_coco_576.py 8
```

### 5. Pretrained Models

#### 5.1 Download Pretrained Models
You can download the pretrained models from the following link:

| Model Name   | mAP  | Link (Google Drive)  | 
| -------------| ---- | ---------------------| 
| GKGNet-576   | 87.65| [Download Link](https://drive.google.com/file/d/1TB_UqqFvpQ2bvy_qau0aKP6GoK9Xlix_/view?usp=share_link) |

#### 5.2 Test Pretrained Models
To test the pretrained models, run the following command:
```sh
python tools/test.py configs/gkgnet/gkgnet_coco_576.py *.pth --metrics mAP
or
bash tools/dist_test.sh configs/gkgnet/gkgnet_coco_576.py *.pth 8 --metrics mAP

```



## Acknowledgement
This repo is developed based on [MMPreTrain](https://github.com/open-mmlab/mmpretrain). 



## License
GKGNet is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please contact Mr. Sheng Jin (jinsheng13[at]foxmail[dot]com). We will send the detail agreement to you.
