# Implementation of CropMix: Sampling a Rich Input Distribution via Multi-Scale Cropping
import random
import numpy as np
import torchvision.transforms as transforms


def mixup(view1, view2, lam, inter_aug):
    if inter_aug:
        permute = [0, 1, 2]
        random.shuffle(permute)
        if lam > 0.5:
            view1 = lam * view1 + (1 - lam) * view2[permute]
            return view1
        else:
            view1 = lam * view1[permute] + (1 - lam) * view2
            return view1
    else:
        return lam * view1 + (1 - lam) * view2


def cutmix(view1, view2, inter_aug):
    lam = np.random.uniform(low=0.0, high=1.0)
    lam2 = np.random.uniform(low=0.0, high=1.0)

    def random_bbox(lam, H, W):
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    _, h, w = view1.shape
    bbx1, bby1, bbx2, bby2 = random_bbox(lam, h, w)

    if inter_aug:
        permute = [0, 1, 2]
        random.shuffle(permute)
        if lam2 > 0.5:
            view2 = view2[permute]
            view1[:, bbx1:bbx2, bby1:bby2] = view2[:, bbx1:bbx2, bby1:bby2]
            return view1
        else:
            view1 = view1[permute]
            view2[:, bbx1:bbx2, bby1:bby2] = view1[:, bbx1:bbx2, bby1:bby2]
            return view2
    else:
        if lam2 > 0.5:
            view1[:, bbx1:bbx2, bby1:bby2] = view2[:, bbx1:bbx2, bby1:bby2]
            return view1
        else:
            view2[:, bbx1:bbx2, bby1:bby2] = view1[:, bbx1:bbx2, bby1:bby2]
            return view2


class CropMix:

    def __init__(self, size,scale, mix_ratio, number, operation, inter_aug):
        self.size = size
        self.scale = scale
        self.mix_ratio = mix_ratio
        self.number = number
        self.operation = operation
        self.inter_aug = inter_aug


    def __call__(self, x):
        if self.number == 234:
            self.number = random.choice([2, 3, 4])

        if self.number == 2:
            t1 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(self.scale, 1)),
                transforms.ToTensor(),
            ])
            t2 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(0.99, 1)),
                transforms.ToTensor(),
            ])
            view1 = t1(x)
            view2 = t2(x)
            if self.operation == 0:
                lam = np.random.beta(self.mix_ratio / self.number, self.mix_ratio / self.number)
                mixed = mixup(view1, view2, lam, self.inter_aug)
            else:
                mixed = cutmix(view1, view2, self.inter_aug)

        elif self.number == 3:
            t1 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(self.scale, self.scale + (1 - self.scale) / self.number)),
                transforms.ToTensor(),
            ])
            t2 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(
                (self.scale + (1 - self.scale) / self.number), (self.scale + 2 * (1 - self.scale) / self.number))),
                transforms.ToTensor(),
            ])
            t3 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(0.99, 1)),
                transforms.ToTensor(),
            ])

            view1 = t1(x)
            view2 = t2(x)
            view3 = t3(x)
            views = [view1, view2, view3]
            random.shuffle(views)

            if self.operation == 0:
                lam = np.random.beta(self.mix_ratio / self.number, self.mix_ratio / self.number)
                mixed = mixup(views[0], views[1], lam, self.inter_aug)
                lam = np.random.beta(self.mix_ratio / self.number, self.mix_ratio / self.number)
                mixed = mixup(mixed, views[2], lam, self.inter_aug)
            else:
                mixed = cutmix(views[0], views[1], self.inter_aug)
                mixed = cutmix(mixed, views[2], self.inter_aug)

        elif self.number == 4:
            t1 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(self.scale, self.scale + (1 - self.scale) / self.number)),
                transforms.ToTensor(),
            ])
            t2 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(
                (self.scale + (1 - self.scale) / self.number), (self.scale + 2 * (1 - self.scale) / self.number))),
                transforms.ToTensor(),
            ])
            t3 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(
                (self.scale + 2 * (1 - self.scale) / self.number), (self.scale + 3 * (1 - self.scale) / self.number))),
                transforms.ToTensor(),
            ])
            t4 = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(0.99, 1)),
                transforms.ToTensor(),
            ])

            view1 = t1(x)
            view2 = t2(x)
            view3 = t3(x)
            view4 = t4(x)
            views = [view1, view2, view3, view4]
            random.shuffle(views)

            if self.operation == 0:
                lam = np.random.beta(self.mix_ratio / self.number, self.mix_ratio / self.number)
                mixed = mixup(views[0], views[1], lam, self.inter_aug)
                lam = np.random.beta(self.mix_ratio / self.number, self.mix_ratio / self.number)
                mixed = mixup(mixed, views[2], lam, self.inter_aug)
                lam = np.random.beta(self.mix_ratio / self.number, self.mix_ratio / self.number)
                mixed = mixup(mixed, views[3], lam, self.inter_aug)
            else:
                mixed = cutmix(views[0], views[1], self.inter_aug)
                mixed = cutmix(mixed, views[2], self.inter_aug)
                mixed = cutmix(mixed, views[3], self.inter_aug)

        return mixed