# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from .builder import AUGMENT
from .utils import one_hot_encoding

import numpy as np

class BaseMixupLayer(object, metaclass=ABCMeta):
    """Base class for MixupLayer.

    Args:
        alpha (float): Parameters for Beta distribution.
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
    """

    def __init__(self, alpha, num_classes, prob=1.0):
        super(BaseMixupLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob

    @abstractmethod
    def mixup(self, imgs, gt_label):
        pass


@AUGMENT.register_module(name='BatchGenderMixupLayer')
class BatchGenderMixupLayer(BaseMixupLayer):
    """Mixup layer for batch mixup."""

    def __init__(self,*args, **kwargs):
        super(BatchGenderMixupLayer, self).__init__(alpha=0.3, num_classes=2,*args, **kwargs)

    def mixup_single(self,img1,img2):
        h, w = img1.shape[-2:]
        #图片下2/3
        h1=(np.random.rand()*1/3+1/3)*h
        h1=int(h1)
        w1=np.sign(np.random.rand()-0.5)*(np.random.rand()*0.5+0.5)*w
        w1=int(w1)
        if w1>0:
            img1[:,-h1:,:w1]=img2[:,:h1,-w1:]
        else:
            img1[:,-h1:,-w1:]=img2[:,:h1,:w1]
        return img1



    def mixup(self, img, gt_label,p=0.3):
        h, w = img.shape[-2:]
        #图片下2/3
        h1=(np.random.rand()*2/3+0.01)*h
        h1=int(h1)
        w1=(np.random.rand()*0.5+0.5)*w
        w1=int(w1)

        h2=(np.random.rand()*2/3+0.01)*h
        h2=int(h2)
        w2=(np.random.rand()*0.5+0.5)*w
        w2=int(w2)


        indexs=np.random.choice(len(img),int(len(img)*p),replace=False)
        indexs_double=indexs[np.random.choice(int(len(img)*p),int(len(img)*p*p),replace=False)]

        indexs_rest=np.array([i for i in range(len(img)) if i not in indexs_double])
        indexs_mix=indexs_rest[np.random.choice(len(indexs_rest),len(indexs),replace=False)]
        indexs_mix_double=indexs_rest[np.random.choice(len(indexs_rest),len(indexs_double),replace=False)]

        img_mix=img.clone()

        if np.random.rand()-0.5>0:
            img_mix[indexs,:,-h1:,:w1]=img[indexs_mix,:,:h1,-w1:]*0.5+img_mix[indexs,:,-h1:,:w1]*0.5
            img_mix[indexs_double, :, -h2:, -w2:] = img[indexs_mix_double, :, :h2, :w2]*0.5+img_mix[indexs_double, :, -h2:, -w2:]*0.5
        else:
            img_mix[indexs, :, -h1:, -w1:] = img[indexs_mix, :, :h1, :w1]*0.5+img_mix[indexs, :, -h1:, -w1:]*0.5
            img_mix[indexs_double, :, -h2:, :w2] = img[indexs_mix_double, :, :h2, -w2:]*0.5+img_mix[indexs_double, :, -h2:, :w2]*0.5


        import matplotlib.pyplot as plt
        img1= img_mix.permute(0,2,3,1)
        imgs=img1.cpu().numpy()
        plt.imshow(imgs[indexs[0]]);
        plt.show();


        return img_mix, gt_label


    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)

@AUGMENT.register_module(name='BatchMixupLayer')
class BatchMixupLayer(BaseMixupLayer):
    """Mixup layer for batch mixup."""

    def __init__(self, *args, **kwargs):
        super(BatchMixupLayer, self).__init__(*args, **kwargs)

    def mixup(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]

        return mixed_img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)
