import inspect
import math
import random
from numbers import Number
from typing import Sequence

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES
from .compose import Compose

try:
    import albumentations
except ImportError:
    albumentations = None

@PIPELINES.register_module()
class CropBBox(object):
    def __init__(self, body_size, face_size, input_key, scale, concat_mask=False):
        self.body_size = body_size
        self.face_size = face_size
        self.pixel_std = 200
        self.input_key = input_key
        self.scale = scale
        self.concat_mask=concat_mask
    
    def __call__(self, results):
        self.results = results
        image = results['img']
        if self.scale:
            body_c = results['body_c']
            body_s = results['body_s']
            body_s_ori = results['body_s_ori']
            r = results['rotation']

            if self.input_key == 'face_cropped':
                face_c = results['face_c']
                face_s = results['face_s']
                face_s_ori = results['face_s_ori']
                face_trans = self.get_affine_transform(face_c, face_s, r, self.face_size)
                face_cropped = cv2.warpAffine(image, face_trans, (int(self.face_size[0]), int(self.face_size[1])), flags=cv2.INTER_LINEAR)
            
                results['face_cropped'] = face_cropped
                results['ori_face_cropped_shape'] = face_cropped.shape

            body_trans = self.get_affine_transform(body_c, body_s, r, self.body_size)
            body_cropped = cv2.warpAffine(image, body_trans, (int(self.body_size[0]), int(self.body_size[1])), flags=cv2.INTER_LINEAR)
        
        else:
            if 'abs_body_bbox' in results:
                [bx1, by1, bx2, by2] = results['abs_body_bbox']
                body_cropped = image[int(by1):int(by2),int(bx1):int(bx2),:]
                body_bbox = [int(_) for _ in results['abs_body_bbox']]
                results['body_cropped'] = body_cropped
                results['ori_body_cropped_shape'] = body_cropped.shape
            
            if 'abs_face_bbox' in results:
                if 'abs_body_bbox' not in results:
                    body_bbox = [0, 0, image.shape[1] - 1, image.shape[0] - 1]
                [x1, y1, x2, y2] = results['abs_face_bbox']
                face_bbox = [int(_) for _ in results['abs_face_bbox']]

                w_padding = min(body_bbox[2] - face_bbox[2], face_bbox[0] - body_bbox[0])
                w_padding = min(w_padding, (face_bbox[2] - face_bbox[0]) // 2)
                h_padding =  min(body_bbox[3] - face_bbox[3], face_bbox[1] - body_bbox[1])
                h_padding = min(h_padding, (face_bbox[3] - face_bbox[1]) // 2)
                w_padding, h_padding = max(w_padding, 0), max(h_padding, 0)

                face_cropped = image[(face_bbox[1] - h_padding): (face_bbox[3] + h_padding), (face_bbox[0] - w_padding) : (face_bbox[2] + w_padding), :]          
                
                results['face_cropped'] = face_cropped
                results['ori_face_cropped_shape'] = face_cropped.shape

            if 'body_cropped' in results and 'abs_face_bbox' in results:
                # ---------------------------- # 
                face_mask = np.zeros((body_cropped.shape[0], body_cropped.shape[1]))

                # one hot
                face_mask[(face_bbox[1] - h_padding - body_bbox[1]): (face_bbox[3] + h_padding - body_bbox[1]), \
                                (face_bbox[0] - w_padding - body_bbox[0]) : (face_bbox[2] + w_padding - body_bbox[0])] = 255

                # gaussian
                # face_center = [(face_bbox[0] + face_bbox[2]) // 2, (face_bbox[1] + face_bbox[3]) // 2]
                # w, h = face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]
                # bw, bh = body_bbox[2] - body_bbox[0], body_bbox[3] - body_bbox[1]
                # iw, ih = image.shape[1], image.shape[0]


                # sigma = 10
                # tmp_size = sigma * 3
                # mu_x = face_center[0]
                # mu_y = face_center[1]

                # # Check that any part of the gaussian is in-bounds
                # ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                # br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # if ul[0] >= iw or ul[1] >= ih \
                #         or br[0] < 0 or br[1] < 0:
                #     pass
                
                # else:
                #     # Generate gaussian
                #     size = 2 * tmp_size + 1
                #     x = np.arange(0, size, 1, np.float32)
                #     y = x[:, np.newaxis]
                #     x0 = y0 = size // 2
                #     # The gaussian is not normalized, we want the center value to equal 1
                #     g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                #     g_x = max(0, -ul[0]), min(br[0], iw) - ul[0]
                #     g_y = max(0, -ul[1]), min(br[1], ih) - ul[1]
                #     img_x = max(0, ul[0]) - int(bx1), min(br[0], iw) - int(bx2)
                #     img_y = max(0, ul[1]) - int(by1), min(br[1], ih) - int(by2)

                #     print(g_x, g_y, img_x, img_y, x.shape, g.shape, face_mask.shape, g[g_y[0]:g_y[1], g_x[0]:g_x[1]].shape, \
                #                 face_mask[img_y[0]:img_y[1], img_x[0]:img_x[1]].shape)  # 

                #     face_mask[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                face_mask = (face_mask * 255).astype(body_cropped.dtype)
                results['face_mask'] = face_mask
        
        results['img'] = results[self.input_key]
        return results

    def get_affine_transform(self, center, scale, rot, output_size,
            shift=np.array([0, 0], dtype=np.float32), inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale * self.pixel_std
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    # the same rule: [1,0] [3,0] ==> [3,-2]  square triangle
    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    # rotation coordination
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        self.num_joints = 17
        self.sigma = 3
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        # 0: w; 1:h 
        target = np.zeros((self.num_joints,
                            self.body_h,
                            self.body_w),
                            dtype=np.float32)

        tmp_size = self.sigma * 3

        x_0, y_0 = self.results['abs_face_bbox'][0], self.results['abs_face_bbox'][1]
        for joint_id in range(self.num_joints):
            mu_x = int(joints[joint_id][0] - x_0 + 0.5)
            mu_y = int(joints[joint_id][1] - y_0 + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.body_w or ul[1] >= self.body_h \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.body_w) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.body_h) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.body_w)
            img_y = max(0, ul[1]), min(br[1], self.body_h)

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight