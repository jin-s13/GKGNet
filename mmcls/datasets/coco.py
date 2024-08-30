import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
import os
from scipy.io import loadmat
from .builder import DATASETS
from .multi_label import MultiLabelDataset

import copy
from abc import ABCMeta, abstractmethod
from sklearn import metrics
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from .pipelines import Compose
import pickle
import numpy as np
from easydict import EasyDict
import torch
def average_precision_area(pred, target,area,area_thr='s'):
    r"""Calculate the average precision for a single class.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        pred (np.ndarray): The model prediction with shape (N, ).
        target (np.ndarray): The target of each prediction with shape (N, ).

    Returns:
        float: a single float as average precision value.
    """
    eps = np.finfo(np.float32).eps

    # sort examples
    sort_inds = np.argsort(-pred)
    sort_target = target[sort_inds]
    area=area[sort_inds]

    # count not difficult examples
    # pn_inds = sort_target != -1
    if area_thr=='s':
        pn_inds = area < 32*32
    elif area_thr=='m':
        pn_inds = ((area < 96*96 )* (area>=32*32)) + (area==0)
    else:
        pn_inds = (area >= 96 * 96) + (area==0)
    pn = np.cumsum(pn_inds)

    # count true positive examples
    pos_inds = (sort_target == 1)*pn_inds
    tp = np.cumsum(pos_inds)
    total_pos = tp[-1]


    tp[np.logical_not(pos_inds)] = 0
    precision = tp / np.maximum(pn, eps)
    ap = np.sum(precision) / np.maximum(total_pos, eps)
    return ap
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return  ap.mean(),ap

def mAP_area(targs, preds,area):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap_record=[]
    for thr in ['s','m','l']:
        ap = np.zeros((preds.shape[1]))
        # compute average precision for each class
        for k in range(preds.shape[1]):
            # sort scores
            scores = preds[:, k]
            targets = targs[:, k]
            # compute average precision
            ap[k] = average_precision_area(scores, targets,area[:, k],thr)
        ap_record.append(ap.mean())
    return  ap_record
#/home/SENSETIME/yaoruijie/Downloads/coco/coco/annotations_pytorch/instances_val2014.json
def get_coco_metrics(all_targets, all_predictions, threshold=0.5):
    """
    index: evaluated label index
    """

    # meanAP = metrics.average_precision_score(all_targets, all_predictions, average='macro', pos_label=1)
    meanAP,ap = mAP(all_targets, all_predictions)
    optimal_threshold = 0.5

    top_3rd = np.sort(all_predictions)[:, -3].reshape(-1, 1)
    all_predictions_top3 = all_predictions.copy()
    all_predictions_top3[all_predictions_top3 < top_3rd] = 0
    all_predictions_top3[all_predictions_top3 < optimal_threshold] = 0
    all_predictions_top3[all_predictions_top3 >= optimal_threshold] = 1

    CP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='macro')
    CR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='macro')
    CF1_top3 = (2 * CP_top3 * CR_top3) / (CP_top3 + CR_top3)
    OP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='micro')
    OR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='micro')
    OF1_top3 = (2 * OP_top3 * OR_top3) / (OP_top3 + OR_top3)

    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2 * CP * CR) / (CP + CR)
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2 * OP * OR) / (OP + OR)


    metrics_dict = {}
    metrics_dict['mAP'] = meanAP

    metrics_dict['CP'] = CP
    metrics_dict['CR'] = CR
    metrics_dict['CF1'] = CF1
    metrics_dict['OP'] = OP
    metrics_dict['OR'] = OR
    metrics_dict['OF1'] = OF1

    metrics_dict['CP_top3'] = CP_top3
    metrics_dict['CR_top3'] = CR_top3
    metrics_dict['OP_top3'] = OP_top3
    metrics_dict['CF1_top3'] = CF1_top3
    metrics_dict['OR_top3'] = OR_top3
    metrics_dict['OF1_top3'] = OF1_top3

    return metrics_dict



@DATASETS.register_module()
class COCO(MultiLabelDataset):
    CLASSES = np.array([
        'person',
                'bicycle',
                'car',
                'motorcycle',
                'airplane',
                'bus',
                'train',
                'truck',
                'boat',
                'traffic light',
                'fire hydrant',
                'stop sign',
                'parking meter',
                'bench',
                'bird',
                'cat',
                'dog',
                'horse',
                'sheep',
                'cow',
                'elephant',
                'bear',
                'zebra',
                'giraffe',
                'backpack',
                'umbrella',
                'handbag',
                'tie',
                'suitcase',
                'frisbee',
                'skis',
                'snowboard',
                'sports ball',
                'kite',
                'baseball bat',
                'baseball glove',
                'skateboard',
                'surfboard',
                'tennis racket',
                'bottle',
                'wine glass',
                'cup',
                'fork',
                'knife',
                'spoon',
                'bowl',
                'banana',
                'apple',
                'sandwich',
                'orange',
                'broccoli',
                'carrot',
                'hot dog',
                'pizza',
                'donut',
                'cake',
                'chair',
                'couch',
                'potted plant',
                'bed',
                'dining table',
                'toilet',
                'tv',
                'laptop',
                'mouse',
                'remote',
                'keyboard',
                'cell phone',
                'microwave',
                'oven',
                'toaster',
                'sink',
                'refrigerator',
                'book',
                'clock',
                'vase',
                'scissors',
                'teddy bear',
                'hair drier',
                'toothbrush',])

    def __init__(self,
                 **kwargs):
        super(COCO, self).__init__(**kwargs)


    def load_annotations(self):
        """Load annotations.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        self.coco_data = pickle.load(open(self.ann_file, 'rb'))

        coco_data = self.coco_data
        num=len(coco_data)
        labels = np.array([coco_data[idx]['objects'] for idx in range(num)])
        # area = np.array([coco_data[idx]['area'] for idx in range(num)])
        images_name= [coco_data[idx]['file_name'] for idx in range(num)]
        for i in range(num):
            info = dict(
                img_prefix=self.data_prefix,
                img_info=dict(filename=images_name[i]),
                gt_label=labels[i].astype(np.int8),
                # area=area
                )
            data_infos.append(info)
        return data_infos
    def get_gt_areas(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """

        gt_areas = np.array([data['area'] for data in self.data_infos])
        return gt_areas
    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                           'be of the same length as gt_labels.'
        evals_record=get_coco_metrics(gt_labels, results, threshold=0.5)
        evals_record={k:round(evals_record[k]*100,4) for k in evals_record}
        evals_record_round={k:round(evals_record[k],4) for k in evals_record}
        if logger is not None:
            logger.info(evals_record_round)
        return evals_record

