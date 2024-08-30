# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmcls.datasets import SAMPLERS


@SAMPLERS.register_module()
class IdInorder(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        ids = [id['gt_label'] for id in self.dataset.datasets[0].data_infos]
        ids_idc={}
        for i,id in enumerate(ids):
            if id not in ids_idc:
                ids_idc[id]=[]
            ids_idc[id].append(i)
        ids_key=list(ids_idc.keys())
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices_id = torch.randperm(len(ids_idc), generator=g).tolist()
        indices=[i for j in indices_id for i in ids_idc[ids_key[j]]]

        return iter(indices)
