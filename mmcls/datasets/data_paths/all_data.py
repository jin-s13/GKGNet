import collections
from genericpath import exists
import json, random, os
from pycocotools.coco import COCO
from tqdm import tqdm


all_data = dict(
        # sh 40
        HKJC_data_part1 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part1_20210724_v1.json',
            sample_ratio=1,
        ),
        HKJC_data_part1_train = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part1_20210724_v1_train.json',
            sample_ratio=1,
        ),
        HKJC_data_part1_val = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part1_20210724_v1_val.json',
            sample_ratio=1,
        ),
        HKJC_data_part2 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part2_20210802_v1.json',
            sample_ratio=1,
        ),
        HKJC_data_part2_train = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part2_20210802_v1_train.json',
            sample_ratio=1,
        ),
        HKJC_data_part2_val = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part2_20210802_v1_val.json',
            sample_ratio=1,
        ),
        AIO_HKJC_part1 = dict(
            ceph_path='sh1985:s3://insight2021/mot/HKJC/',
            lustre_path='',
            anno_path='data/anno_paths/AIO_HKJC/AIO_HKJC_part1.json',
            sample_ratio=1/5,
        ),
        AIO_HKJC_part2 = dict(
            ceph_path='sh1985:s3://insight2021/cam_155/mot/',
            lustre_path='',
            anno_path='data/anno_paths/AIO_HKJC/AIO_HKJC_part2.json',
            sample_ratio=1/5,
        ),
        HancomTest_part1 = dict(
            ceph_path='sh1986:s3://add_hancom/',
            lustre_path='/mnt/lustre/datatag/jinsheng/hancom/',
            anno_path='data/anno_paths/hancom/hancom_part1_20210715_v1.json',
            sample_ratio=1
        ),
        HancomTest_part2 = dict(
            ceph_path='sh1986:s3://add_hancom/',
            lustre_path='/mnt/lustre/datatag/jinsheng/hancom/',
            anno_path='data/anno_paths/hancom/hancom_part2_20210715_v1.json',
            sample_ratio=1
        ),
        Escalator_20210805 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng/',
            anno_path='data/anno_paths/escalator_20210805/escalator_20210805_v1.json',
            sample_ratio=1/2
        ),
        Escalator_20210807 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng/',
            anno_path='data/anno_paths/escalator_20210807/escalator_20210807_v1.json',
            sample_ratio=1/4
        ),
        Hanshin = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng/',
            anno_path='data/anno_paths/hanshin/hanshin_20210813_v2.json',
            # anno_path='data/anno_paths/hanshin/hanshin_20210813_v5.json',
            sample_ratio=1
        ),
        Hanshin_4440 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng/',
            anno_path='data/anno_paths/hanshin/hanshin4440_20211012_v1.json',
            sample_ratio=1
        ),
        Hanshin_pseudo_bbox = dict(
            ceph_path='sh1986:s3://add_imx500/hanshin',
            lustre_path='/mnt/lustre/datatag/jinsheng/',
            anno_path='/mnt/lustrenew/share_data/zhangweijia/autolink_work_dirs/hanshin/latest_annotations.json',
            sample_ratio=1
        ),
        SeniorCitizen = dict(
            ceph_path='sh1986:s3://add_senior_cam_data/',
            lustre_path='/mnt/lustrenew/share_data/zhangweijia/senior_cam_data/images/',
            anno_path='data/anno_paths/SeniorCitizen/annotations_all.json',
            sample_ratio=1/30,
        ),
        # sh 36
        CRP = dict(
            ceph_path='sh1986:s3://add_crp/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/CRP/',
            anno_path='data/anno_paths/CRP/crp_annotations_kp17.json',
            sample_ratio=1
        ),
        CRP_train = dict(
            ceph_path='sh1986:s3://add_crp/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/CRP/',
            anno_path='data/anno_paths/CRP/crp_annotations_kp17_train.json',
            sample_ratio=1
        ),
        CRP_val = dict(
            ceph_path='sh1986:s3://add_crp/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/CRP/',
            anno_path='data/anno_paths/CRP/crp_annotations_kp17_val.json',
            sample_ratio=1
        ),
        #sh36
        WIDER_trainval = dict(
            ceph_path='sh1986:s3://add_wider/Image/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/WIDER/Image',
            anno_path='data/anno_paths/WIDER/wider_attribute_trainval_kp17.json',
            sample_ratio=1

        ),
        WIDER_test = dict(
            ceph_path='sh1986:s3://add_wider/Image/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/WIDER/Image',
            anno_path='data/anno_paths/WIDER/wider_attribute_test_kp17.json',
            sample_ratio=1
        ),
        # sh36
        Parse27k_train = dict(
            ceph_path='sh1986:s3://add_parse27k/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/Parse27k/',
            anno_path='data/anno_paths/Parse27k/parse27k_train_kp17.json',
            sample_ratio=1
        ),
        Parse27k_val = dict(
            ceph_path='sh1986:s3://add_parse27k/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/Parse27k/',
            anno_path='data/anno_paths/Parse27k/parse27k_val_kp17.json',
            sample_ratio=1
        ),
        Parse27k_test = dict(
            ceph_path='sh1986:s3://add_parse27k/',
            lustre_path='/mnt/lustre/share/zhangweijia/datasets/HumanAttributes/Parse27k/',
            anno_path='data/anno_paths/Parse27k/parse27k_test_kp17.json',
            sample_ratio=1
        ),
        # DukeMTMC_reID bj15
        # DukeMTMC_reID_train = dict(
        #     ceph_path='sh1986:s3://add_cropbody',
        #     lustre_path='data/datatag',
        #     anno_path='data/anno_paths/DukeMTMC_reID/DukeMTMC_reid_train_20210811.json',
        #     sample_ratio=1
        # ),
        # DukeMTMC_reID_test = dict(
        #     ceph_path='sh1986:s3://add_cropbody',
        #     lustre_path='data/datatag',
        #     anno_path='data/anno_paths/DukeMTMC_reID/DukeMTMC_reid_test_20210811.json',
        #     sample_ratio=1
        # ),

        # DukeMTMC_reID sh40
        # 源于DukeMTMC-reID数据集，将具有人脸的person放在hanshin的背景上贴图，
        # 期望提高hanshin的效果
        DukeMTMC_reID_v1 = dict(
            ceph_path='sh1986:s3://add_cropbody',
            lustre_path='/mnt/lustre/share/wangcan/dataset/DukeMTMC/images',
            anno_path='data/anno_paths/DukeMTMC_reID_v1/dukev1_train_0811.json',
            sample_ratio=1
        ),
        DukeMTMC_reID_v2 = dict(
            ceph_path='sh1986:s3://add_cropbody',
            lustre_path='/mnt/lustre/share/wangcan/dataset/DukeMTMC/images',
            anno_path='data/anno_paths/DukeMTMC_reID_v2/dukev2_train_0811.json',
            sample_ratio=1
        ),
        # bj15
        DukeMTMC_reID_train = dict(
            ceph_path='sh1986:s3://add_cropbody',
            lustre_path='/mnt/lustre/share/jinsheng/DukeMTMC-reID',
            anno_path='data/anno_paths/DukeMTMC_reID/duke_reid_train_v1.json',
            sample_ratio=1
        ),
        DukeMTMC_reID_test = dict(
            ceph_path='sh1986:s3://add_cropbody',
            lustre_path='/mnt/lustre/share/jinsheng/DukeMTMC-reID',
            anno_path='data/anno_paths/DukeMTMC_reID/duke_reid_test_v1.json',
            sample_ratio=1
        ),
        # Blancev1 sh36
        Blancev1 = dict(
            ceph_path='sh1986:s3://add_divide2w/images_2w/',
            lustre_path='mnt/lustre/share/wangcan/dataset/aio/divide4_2w/',
            anno_path='data/anno_paths/aio/divide4_2w/train.json',
            sample_ratio=1
        ),
        # CUHK sh36
        CUHK = dict(
            ceph_path='sh1985_1:s3://imx500/images/',
            lustre_path='/mnt/lustre/share/wencheng/datasets/CUHKSYSU/images/',
            anno_path='data/anno_paths/CUHK/CUHK_v2.json',
            sample_ratio=1
        ),
        CUHK_train = dict(
            ceph_path='sh1985_1:s3://imx500/images/',
            lustre_path='/mnt/lustre/share/wencheng/datasets/CUHKSYSU/images/',
            anno_path='data/anno_paths/CUHK/CUHK_v2_train.json',
            sample_ratio=1
        ),
        CUHK_val = dict(
            ceph_path='sh1985_1:s3://imx500/images/',
            lustre_path='/mnt/lustre/share/wencheng/datasets/CUHKSYSU/images/',
            anno_path='data/anno_paths/CUHK/CUHK_v2_val.json',
            sample_ratio=1
        ),

        # FilterConverted sh36
        FilterConverted = dict(
            ceph_path='sh1985_1:s3://imx500/captured/',
            lustre_path='/mnt/lustre/share/zhouyang/hanzhixiong/captured/',
            anno_path='data/anno_paths/aio/new/homa_0611_final_filter_converted.json',
            sample_ratio=1/5
        ),
        # NCSV1 sh36
        NCSV1 = dict(
            ceph_path='sh1985_1:s3://imx500/ncs_images/',
            lustre_path='/mnt/lustre/liuxin1/data_bak/yangmingmin/NCS/',
            anno_path='data/anno_paths/ncsv1/ncsv1_v1.json',
            sample_ratio=1/4
        ),
        NCSV2 = dict(
            ceph_path='sh1985:s3://insight2021/',
            lustre_path='unknown',
            anno_path='data/anno_paths/ncsv2/ncsv2_all.json',
            sample_ratio=1/25,
        ),
        # Spuermarket2020 sh36
        Spuermarket2020 = dict(
            ceph_path='sh1985_1:s3://imx500/pack_supermarket/',
            lustre_path='/mnt/lustre/share/zhouyang/pack_supermarket/',
            anno_path='data/anno_paths/supermarket2020/supermarket2020_v2.json',
            sample_ratio=1/10
        ),
        Spuermarket2020_train = dict(
            ceph_path='sh1985_1:s3://imx500/pack_supermarket/',
            lustre_path='/mnt/lustre/share/zhouyang/pack_supermarket/',
            anno_path='data/anno_paths/supermarket2020/supermarket2020_v2_train.json',
            sample_ratio=1/10
        ),
        Spuermarket2020_val = dict(
            ceph_path='sh1985_1:s3://imx500/pack_supermarket/',
            lustre_path='/mnt/lustre/share/zhouyang/pack_supermarket/',
            anno_path='data/anno_paths/supermarket2020/supermarket2020_v2_val.json',
            sample_ratio=1
        ),
        # Spuermarket2021 sh36
        Spuermarket2021 = dict(
            ceph_path='sh1985_1:s3://imx500/supermarket_data_FHD/',
            lustre_path='/mnt/lustre/share/zhouyang/supermarket_data_FHD/',
            anno_path='data/anno_paths/supermarket2020/supermarket2021_v1.json',
            sample_ratio=1/5
        ),
        Spuermarket2021_train = dict(
            ceph_path='sh1985_1:s3://imx500/supermarket_data_FHD/',
            lustre_path='/mnt/lustre/share/zhouyang/supermarket_data_FHD/',
            anno_path='data/anno_paths/supermarket2020/supermarket2021_v1_train.json',
            sample_ratio=1/5
        ),
        Spuermarket2021_val = dict(
            ceph_path='sh1985_1:s3://imx500/supermarket_data_FHD/',
            lustre_path='/mnt/lustre/share/zhouyang/supermarket_data_FHD/',
            anno_path='data/anno_paths/supermarket2020/supermarket2021_v1_val.json',
            sample_ratio=1
        ),
        # WholeBodyAttribute sh40
        WholeBodyAttribute_train = dict(
            ceph_path='sh1986:s3://canwangmscoco/',
            lustre_path='/mnt/lustre/share/DSK/datasets/mscoco2017/',
            anno_path='data/anno_paths/coco_attribute/coco_wholebody_train_v1.0_attribute_20210801.json',
            sample_ratio=1
        ),
        WholeBodyAttribute_train_pseudo = dict(
            ceph_path='sh1986:s3://canwangmscoco/',
            lustre_path='/mnt/lustre/share/DSK/datasets/mscoco2017/',
            anno_path='data/anno_paths/coco_attribute/coco_wholebody_train_v1.0_attribute_20210801_masklabel.json',
            sample_ratio=1
        ),
        WholeBodyAttribute_val = dict(
            ceph_path='sh1986:s3://canwangmscoco/',
            lustre_path='/mnt/lustre/share/DSK/datasets/mscoco2017/',
            anno_path='data/anno_paths/coco_attribute/coco_wholebody_val_v1.0_attribute_20210801.json',
            sample_ratio=1
        ),
        coco_train = dict(
            ceph_path='none',
            lustre_path='data/coco/train2017',
            anno_path='data/coco/annotations/person_keypoints_train2017.json',
            sample_ratio=1
        ),

        # part of training data
        SeniorCitizen_tmp = dict(
            ceph_path='sh1986:s3://add_senior_cam_data/',
            lustre_path='/mnt/lustrenew/share_data/zhangweijia/senior_cam_data/images/',
            anno_path='data/anno_paths/SeniorCitizen/annotations_all_save_train.json',
            sample_ratio=1,
        ),

        NCSV1_tmp = dict(
            ceph_path='sh1985_1:s3://imx500/ncs_images/',
            lustre_path='/mnt/lustre/liuxin1/data_bak/yangmingmin/NCS/',
            anno_path='data/anno_paths/ncsv1/ncsv1_v1_save_train.json',
            sample_ratio=1
        ),
        NCSV2_tmp = dict(
            ceph_path='sh1985:s3://insight2021/',
            lustre_path='unknown',
            anno_path='data/anno_paths/ncsv2/ncsv2_all_save_train.json',
            sample_ratio=1,
        ),

        AIO_HKJC_part1_tmp = dict(
            ceph_path='sh1985:s3://insight2021/mot/HKJC/',
            lustre_path='',
            anno_path='data/anno_paths/AIO_HKJC/AIO_HKJC_part1_save_train.json',
            sample_ratio=1,
        ),

        AIO_HKJC_part2_tmp = dict(
            ceph_path='sh1985:s3://insight2021/cam_155/mot/',
            lustre_path='',
            anno_path='data/anno_paths/AIO_HKJC/AIO_HKJC_part2_save_train.json',
            sample_ratio=1,
        ),

        Spuermarket2020_train_tmp = dict(
            ceph_path='sh1985_1:s3://imx500/pack_supermarket/',
            lustre_path='/mnt/lustre/share/zhouyang/pack_supermarket/',
            anno_path='data/anno_paths/supermarket2020/supermarket2020_v2_train_save_train.json',
            sample_ratio=1
        ),

        Spuermarket2021_train_tmp = dict(
            ceph_path='sh1985_1:s3://imx500/supermarket_data_FHD/',
            lustre_path='/mnt/lustre/share/zhouyang/supermarket_data_FHD/',
            anno_path='data/anno_paths/supermarket2020/supermarket2021_v1_train_save_train.json',
            sample_ratio=1
        ),

        
        # allval
        val_all = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all.json',
            sample_ratio=1
        ),
        val_all_summer = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all_summer.json',
            sample_ratio=1
        ),

        val_all_accessory = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all_accessory.json',
            sample_ratio=1
        ),

        val_all_mask = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all_mask.json',
            sample_ratio=1
        ),
    
        val_all_mask_v1 = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all_mask_v1.json',
            sample_ratio=1
        ),

        val_all_kpts = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all_kpts.json',
            sample_ratio=1
        ),

        val_all_sample = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/val_all_val.json',
            sample_ratio=1
        ),

        Hanshin_pseudo = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='data/datatag/',
            anno_path='outputs/imx500/resnest101/Hanshin_coco_format.json',
            sample_ratio=1
        ),

        HancomTest_part1_pseudo = dict(
            ceph_path='sh1986:s3://add_hancom/',
            lustre_path='/mnt/lustre/datatag/jinsheng/hancom/',
            anno_path='outputs/imx500/resnest101/HancomTest_part1_coco_format.json', 
            sample_ratio=1
        ),

        HancomTest_part2_pseudo = dict(
            ceph_path='sh1986:s3://add_hancom/',
            lustre_path='/mnt/lustre/datatag/jinsheng/hancom/',
            anno_path='outputs/imx500/resnest101/HancomTest_part2_coco_format.json', 
            sample_ratio=1
        ),

        # debug
        debug_hkjc = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='/mnt/lustre/datatag/jinsheng',
            anno_path='data/anno_paths/HKJC_data/hkjc_data_part1_20210724_v1.json',
            sample_ratio=1
        ),
        # local
        localdebug = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='data/datatag/',
            anno_path='data/anno_paths/hanshin/hanshin_20210813_v2.json',
            sample_ratio=1
        ),
        localhancom_1 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='data/datatag/hancom',
            anno_path='data/anno_paths/hancom/hancom_part1_20210715.json',
            sample_ratio=1
        ),
        localhancom_2 = dict(
            ceph_path='sh1986:s3://add_imx500/',
            lustre_path='data/datatag/hancom',
            anno_path='data/anno_paths/hancom/hancom_part2_20210715.json',
            sample_ratio=1
        ),
        localdebug_duke = dict(
            ceph_path='sh1986:s3://add_cropbody',
            lustre_path='data/datatag',
            anno_path='data/anno_paths/DukeMTMC-reID/DukeMTMC_reid_test_20210811.json',
            sample_ratio=1
        ),
        total_val = dict(
            ceph_path='',
            lustre_path='',
            anno_path='data/anno_paths/total_val.json',
            sample_ratio=1
        ),
        
)

# attention
# 包含外国人数据集: WIDER_trainval, WIDER_test, COCO
# Parse27k_train, Parse27k_val, Parse27k_test, CRP 

# DukeMTMC_reID_train, DukeMTMC_reID_test
# DukeMTMC_reID_v1, DukeMTMC_reID_v2, 
# Market1501 
# 拼接图： Blancev1
# 贴图： DukeMTMC_reID_v1， DukeMTMC_reID_v2

# NCSV2 里面的id是什么意思 
# AIOHKJC


# NCSV2_ids = [0, 12, 18, 21, 24, 3, 6, 9, 10, 11, 13, 14, 16, 17, 19, 1, 20, 23, 25, 2, 4, 5, 7, 8, 4, 5, 7]
# for id in NCSV2_ids:
#     tmp_key = "NCSV2_" + str(id)
#     tmp_value = dict(
#             ceph_path='sh1985_1:s3://insight2021/',
#             lustre_path='unknown',
#             anno_path='data/anno_paths/MOT/new/homa_ncs_mot_v2_{}.json'.format(id))
#     tmp_dic = {tmp_key:tmp_value}
#     all_data.update(tmp_dic)



Balance = {'Spuermarket2020':1/8, 'Spuermarket2021':1/4}



def make_trainval(exception, save_path='anno_paths_new', ratios=1/5):
    random.seed(1)
    with open('data/anno_paths/hanshin/hanshin_20210813.json', 'r') as f:
        r = json.load(f)
    categories = r['categories']

    val_anno = {'images':[], 'annotations':[], 'categories': categories}
    img_id, anno_id = 0, 0
    for data_name, data_info in all_data.items():
        if data_name in exception:
            continue
        print(data_name)
        train_anno = {'images':[], 'annotations':[], 'categories':categories}
        lustre_path = data_info['lustre_path']
        ceph_path = data_info['ceph_path']
        anno_path = data_info['anno_path']

        data_load = COCO(anno_path)
        imgids = data_load.getImgIds()
        sample_ratio = data_info['sample_ratio']

        balance_ratio = Balance.get(data_name, 1)
        imgids = random.sample(imgids, int(len(imgids) * sample_ratio * balance_ratio))
        trainids = random.sample(imgids, int(len(imgids) * (1 - ratios)))
        
        for imgid in tqdm(imgids):
            image_info = data_load.loadImgs(imgid)[0]
            annsids = data_load.getAnnIds(imgid)
            anns = data_load.loadAnns(annsids)

            if imgid not in trainids and data_name not in ['Hanshin', 'HancomTest_part1', 'HancomTest_part2']:
                image_info['id'] = img_id
                image_path = os.path.join(lustre_path, image_info['file_name'])
                if not os.path.exists(image_path):
                    image_info['file_name'] = os.path.join(ceph_path, image_info['file_name'])
                else:
                    image_info['file_name'] = image_path
                for i in range(len(anns)):
                    anns[i]['image_id'] = img_id
                    anns[i]['id'] = anno_id
                    anno_id += 1
                img_id += 1
                val_anno['images'].append(image_info)
                val_anno['annotations'] += anns
            else:
                train_anno['images'].append(image_info)
                train_anno['annotations'] += anns
        
        save_dir = os.path.dirname(anno_path.replace('anno_paths', save_path))
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        with open(anno_path.replace('anno_paths', save_path), 'w') as f:
            json.dump(train_anno, f)
    
    with open('data/anno_paths_new/total_val.json', 'w') as f:
        json.dump(val_anno, f)
        



# attention
# 包含外国人数据集: CRP, WIDER_trainval, WIDER_test, Parse27k_train, Parse27k_val, Parse27k_test, DukeMTMC_reID_train, DukeMTMC_reID_test, 
#                DukeMTMC_reID_v1, DukeMTMC_reID_v2
# 拼接图： Blancev1
# 贴图： DukeMTMC_reID_v1， DukeMTMC_reID_v2

# NCSV2 里面的id是什么意思 
# AIOHKJC


if __name__ == '__main__':
    exception = ['CRP', 'WIDER_trainval', 'WIDER_test', 'Parse27k_train', 'Parse27k_val', 'Parse27k_test', 'DukeMTMC_reID_train', 'DukeMTMC_reID_test', 
               'DukeMTMC_reID_v1', 'DukeMTMC_reID_v2', 'Blancev1', 'WholeBodyAttribute_train', 'WholeBodyAttribute_val',
               'AIOHKJC_10001', 'AIOHKJC_10002', 'AIOHKJC_10003', 'AIOHKJC_10004', 'AIOHKJC_10005', 'AIOHKJC_10006', 'AIOHKJC_10007',
               'NCSV2']
    
    exception += ['localdebug', 'localdebug_1', 'localdebug_duke', 'total_val']
    exception += ['HKJC_data_part1_ceph']

    make_trainval(exception)
