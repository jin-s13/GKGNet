import collections
from genericpath import exists
import json, random, os
from pycocotools.coco import COCO
from tqdm import tqdm


all_data_face4age = {
    'train':
        {
        'type': 'attribute',
        'task': 'age',
        # 
        'lustre_path': '',
        'ceph_path':'sh1985:s3://add_attribute/SenseInsight_age_v1',
        # 
        'anno_path': 'data/anno_paths/face_align/SenseInsight_shai_train_QR_shai_v20_test_four_suoni_2bei.json',
        'sample_ratio':1,
        'anno_key':'age_trn',
        },
    'val':
        {
        "type": 'attribute',
        "task": 'age',
        # 
        "lustre_path": 'data/attribute_age/test_all',
        "ceph_path": 'sh1985:s3://add_attribute/test_all',
        # 
        "anno_path": 'data/anno_paths/face_align/zhangyue_tst_0706_fhd.json', # ~7000
        # 
        'sample_ratio':1,
        'anno_key':'zhangyue_0706_fhd_tst',
        },
    'val_picked':
        {
        "type": 'attribute',
        "task": 'age',
        # 
        "lustre_path": 'data/attribute_age/test_picked',
        "ceph_path": 'sh1985:s3://add_attribute/test_picked',
        # 
        "anno_path": 'data/anno_paths/face_align/zhangyue_0706_fhd_all_jiaodu_qingxi.json',  # ~5000
        # 
        'sample_ratio':1,
        'anno_key':'zhangyue_0706_fhd_all',
        },
    'Hanshin':
        {
        "type": 'attribute',
        "task": 'age',
        "lustre_path": '',
        "anno_path": 'data/anno_paths/face_align/hanshin.json', 
        'sample_ratio':1,
        'anno_key':'hanshin',
        },
    'HancomTest_part1':
        {
        "type": 'attribute',
        "task": 'age',
        "lustre_path": '',
        "anno_path": 'data/anno_paths/face_align/part1.json', 
        'sample_ratio':1,
        'anno_key':'part1',
        },
    'HancomTest_part2':
        {
        "type": 'attribute',
        "task": 'age',
        "lustre_path": '',
        "anno_path": 'data/anno_paths/face_align/part2.json', 
        'sample_ratio':1,
        'anno_key':'part2',
        },
    #***************#
    'localdebug':
        {
        "type": 'attribute',
        "task": 'age',
        # 
        "lustre_path": 'data/attribute_age/',
        "ceph_path": 'sh1985:s3://add_attribute/test_all',
        # 
        "anno_path": 'data/anno_paths/face_align/zhangyue_tst_0706_fhd_update.json', 
        # 
        'sample_ratio':1,
        'anno_key':'zhangyue_0706_fhd_tst',
        },
}


def make_age_annotation():
    from pycocotools.coco import COCO
    from glob import glob
    from collections import defaultdict
    face_path = 'data/face_align'
    anno_paths={'hanshin':'data/anno_paths/hanshin/hanshin_20210813_v2.json',
                'part1':'data/anno_paths/hancom/hancom_part1_20210715_v1.json',
                'part2':'data/anno_paths/hancom/hancom_part2_20210715_v1.json'
    }
    imgs = []
    for dir, path, files in os.walk(face_path):
        for file in files:
            if file.endswith('bmp'):
                imgs.append(os.path.join(dir, file))

    print(len(imgs))

    annid2age = {}
    for data_key, value in anno_paths.items():
        coco = COCO(value)
        image_ids = coco.getImgIds()
        id2name = {}
        name2id = {}
        for id in coco.getImgIds():
            image_info = coco.loadImgs(id)[0]
            id2name[image_info['id']] = image_info['file_name']
            name2id[image_info['file_name']] = image_info['id']
        
        for img_id in image_ids:
            image_path = id2name[img_id]
            annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None) # iscrowd
            objs = coco.loadAnns(annIds)

            for obj in objs:
                # if obj['id'] == 34377:print(obj)
                age, face_bbox = None, None
                age_keys = [
                        'age_manual',
                        'manual_age',
                        'age_real',
                        'age', 
                        'age_pseudo',
                        'extra'
                ]
                for key in age_keys:
                    if key in obj:
                        age = obj[key]
                        break
                face_bbox_keys = [
                            'face_bbox_visible_manual',
                            'face_bbox', 
                            'face_bbox_visible_pseudo'
                ]
                for key in face_bbox_keys:
                    if key in obj:
                        face_bbox = obj[key]
                        break
                
                if age is not None and face_bbox is not None:
                    if data_key not in annid2age:
                        annid2age.update({data_key:{}})
                    annid2age[data_key][obj['id']] = {'face_bbox':face_bbox, 'age':age, 'image_path':image_path}
    
    print(annid2age.keys())
    save_anns = defaultdict(list)
    for img in imgs:
        dataname = img.split('/')[3]
        basenames = os.path.basename(img).split('_')
        annid = int(basenames[-1][:-4])
        bbox = [
            int(basenames[0].split('jpg')[1]),
            int(basenames[-4]),
            int(basenames[-3]),
            int(basenames[-2])
        ]
        # print(img)
        if annid not in annid2age[dataname]: continue
        assert [int(_) for _ in annid2age[dataname][annid]['face_bbox']][:2] == bbox[:2], [bbox, img]
        save_anns[dataname].append({
            'img_info':{'filename':img},
            'annos':{
                'attribute':{'age': annid2age[dataname][annid]['age']}
            }
        })
    
    for data_key in anno_paths.keys():
        with open('data/anno_paths/face_align/{}.json'.format(data_key), 'w') as f:
            json.dump({f'{data_key}':save_anns[data_key]}, f)

def rename_key(json_path):
    import re
    with open(json_path, 'r') as f:
        r_json = json.load(f)
    for key, values in r_json.items():
        for value in values:
            ori_filename = value['img_info']['filename']
            updated_filename = ori_filename.split('/zhouyang/')[1]
            updated_filename = '/mnt/lustre/share/wangjiahang1/face_attribute/' + updated_filename
            # pattern = re.compile(r'[\u4e00-\u9fa5]')
            # updated_filename = re.sub(pattern, "", updated_filename)
            value['img_info'].update({'filename':updated_filename})
    with open(json_path[:-5] + '_update.json', 'w') as f:
        json.dump(r_json, f)

def rename_dir(path):
    import re, shutil
    extensions = ['png', 'PNG', 'JPG', 'jpg', 'bmp', 'BMP']
    for dir, path, files in os.walk(path):
        for file in files:
            if dir != re.sub(re.compile(r'[\u4e00-\u9fa5]'), "", dir) and os.path.exists(dir):
                shutil.move(dir, re.sub(re.compile(r'[\u4e00-\u9fa5]'), "", dir))
            # if os.path.join(dir, file).split('.')[-1] in extensions:
            #     ori = os.path.join(dir, file)
            #     update = re.sub(re.compile(r'[\u4e00-\u9fa5]'), "", ori)
            #     shutil.move(ori, update)

if __name__ == '__main__':
    # make_age_annotation()
    # rename_dir('/mnt/lustre/wangjiahang1/mmclassification/data/attribute_age/test_data_from_zhangyue_0706')
    # rename_key('/mnt/lustre/wangjiahang1/mmclassification/data/anno_paths/face_align/zhangyue_tst_0706_fhd.json')
    pass




                
            










