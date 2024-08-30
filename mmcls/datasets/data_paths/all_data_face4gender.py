import collections
from genericpath import exists
import json, random, os
from pycocotools.coco import COCO
from tqdm import tqdm


all_data_face4gender = {
    'gender_data_imdb_v1':
        {
        'name':'gender_data_imdb_v1',
        'type': 'attribute',
        'task': 'gender',
        'prefix': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustrenew/wangfei/attritbutes/data/gender/imdbface_v1/st_align',
        'json_path': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/data_mds1/dataset/gender_imdb_finance_tantan/gender_trn/gender_data_imdb_v1.json',
        'batch_size': 2 # for single gpu
        },
    'gender_data_imdb_v2':
        {
        'name':'gender_data_imdb_v2',
        "type": 'attribute',
        "task": 'gender',
        "prefix": '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustrenew/lixiaojie/imdbface/imdbface_v2/st_align',
        "json_path": '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/data_mds1/dataset/gender_imdb_finance_tantan/gender_trn/gender_data_imdb_v2.json',
        "batch_size": 2 # for single gpu
        },
    'gender_data_finance': 
        {
        'name':'gender_data_finance',
        "type": 'attribute',
        "task": 'gender',
        "prefix": '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustrenew/lixiaojie/imdbface/finance/',
        "json_path": '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/data_mds1/dataset/gender_imdb_finance_tantan/gender_trn/gender_data_finance.json',
        },
    'gender_data_tantan':
        {
        'name':'gender_data_tantan',
        "type": 'attribute',
        "task": 'gender',
        "prefix": '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustrenew/lixiaojie/imdbface/wild_chinese/tantan_v2/st_align_106',
        "json_path": '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/data_mds1/dataset/gender_imdb_finance_tantan/gender_trn/gender_data_tantan.json',
        'batch_size': 8 # for single gpu
        },
    'gender_data_senseeye':
        {
        'name':'gender_data_senseeye',
        'type': 'attribute',
        'task': 'gender',
        'prefix': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_align_result_2.2.0/senseeye_trn_maplist_gender_images',
        'json_path': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_json_new/LK_gender_trn.json',
        },
    'gender_data_CD':
        {
        'name':'gender_data_CD',
        'type': 'attribute',
        'task': 'gender',
        'prefix': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_align_result_2.2.0/CD_trn_maplist_gender_images',
        'json_path': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_json_new/CD_gender_trn.json',
        'batch_size': 4 # for single gpu
        },
    'gender_data_0144':
        {
        'name':'gender_data_0144',
        'type': 'attribute',
        'task': 'gender',
        'prefix': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_align_result_2.2.0/0144_trn_maplist_gender_images',
        'json_path': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_json_new/0144_gender_trn.json',
        'batch_size': 4 # for single gpu
        },
    'gender_data_ipc':
        {
        'name':'gender_data_ipc',
        'type': 'attribute',
        'task': 'gender',
        'prefix': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_align_result_2.2.0/ipc_trn_maplist_gender_images',
        'json_path': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_json_new/vehicle_RGB_gender_train.json',
        'batch_size': 4 # for single gpu
        },
    'gender_data_children':
        {
        'name':'gender_data_children',
        'type': 'attribute',
        'task': 'gender',
        'prefix': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_align_result_2.2.0/child_trn_maplist_gender_images',
        'json_path': '/mnt/lustrenew/zhangweiliang/data_bak/mnt/lustre/yangmingmin/dataset/attribute_2.2.0/trn_json_new/child_gender_train.json',
        'batch_size': 2 # for single gpu
        }
}