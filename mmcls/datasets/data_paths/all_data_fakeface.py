import collections
from genericpath import exists
import json, random, os
from pycocotools.coco import COCO
from tqdm import tqdm

all_data_fakeface = dict(
        # sh 40
        CUHK_fakeface = dict(
            ceph_path='sh1985:s3://fakeface/CUHK/CUHK_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/cuhk_mask/CUHK_v2_mask.json',
            sample_ratio=1,
        ),
        CUHK_fakeface_train = dict(
            ceph_path='sh1985:s3://fakeface/CUHK/CUHK_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/cuhk_mask/CUHK_v2_mask_train.json',
            sample_ratio=1,
        ),
        CUHK_fakeface_val = dict(
            ceph_path='sh1985:s3://fakeface/CUHK/CUHK_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/cuhk_mask/CUHK_v2_mask_val.json',
            sample_ratio=1,
        ),
        SenseAIC_fakeface = dict(
            ceph_path='sh1985:s3://fakeface/senseaic_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/sense_aic_mask/latest_annotations_mask.json',
            sample_ratio=1,
        ),
        COCO_train_fakeface = dict(
            ceph_path='sh1985:s3://fakeface/COCO/train2017_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/mscoco2017/train2017/coco_wholebody_train_v1.0_mask.json',
            sample_ratio=1,
        ),
        COCO_val_fakeface = dict(
            ceph_path='sh1985:s3://fakeface/COCO/val2017_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/mscoco2017/val2017/coco_wholebody_val_v1.0_mask.json',
            sample_ratio=1,
        ),

        HKJC_part1_fakeface_train = dict(
            ceph_path='sh1985:s3://fakeface/HKJC_part1_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/HKJC_data_part1/hkjc_data_part1_20210724_v1_mask_train.json',
            # anno_path='data/anno_paths/HKJC_data/hkjc_data_part1_20210724_v1_mask_train.json',
            sample_ratio=1,
        ),

        HKJC_part1_fakeface_val = dict(
            ceph_path='sh1985:s3://fakeface/HKJC_part1_mask/',
            lustre_path='',
            # anno_path='data/anno_paths/HKJC_data/hkjc_data_part1_20210724_v1_mask_val.json',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/HKJC_data_part1/hkjc_data_part1_20210724_v1_mask_val.json',
            sample_ratio=1,
        ),

        HKJC_part2_fakeface_train = dict(
            ceph_path='sh1985:s3://fakeface/HKJC_part2_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/HKJC_data_part2/hkjc_data_part2_20210802_v1_mask_train.json',
            sample_ratio=1,
        ),
        HKJC_part2_fakeface_val = dict(
            ceph_path='sh1985:s3://fakeface/HKJC_part2_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/HKJC_data_part2/hkjc_data_part2_20210802_v1_mask_val.json',
            sample_ratio=1,
        ),

        face_300w_train = dict(
            ceph_path='sh1985:s3://fakeface/300W_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/300W/annotations/face_landmarks_300w_train_mask.json',
            sample_ratio=1,
        ),
        face_300w_val = dict(
            ceph_path='sh1985:s3://fakeface/300W_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/300W/annotations/face_landmarks_300w_valid_mask.json',
            sample_ratio=1,
        ),
        face_300w_test = dict(
            ceph_path='sh1985:s3://fakeface/300W_mask/',
            lustre_path='',
            anno_path='/mnt/lustre/share/wangjiahang1/FakeFace/300W/annotations/face_landmarks_300w_test_mask.json',
            sample_ratio=1,
        ),
)