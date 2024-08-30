# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import matplotlib.pyplot as plt
import cv2
def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [model.CLASSES[lb] for lb in pred_label]

            img_metas = data['img_metas'].data[0]
            imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[i],
                    'pred_label': pred_label[i],
                    'pred_class': pred_class[i]
                }
                model.module.show_result(
                    img_show,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

# def single_gpu_test(model,
#                     data_loader,
#                     show=False,
#                     out_dir=None,
#                     **show_kwargs):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     for i, data in enumerate(data_loader):
#         files=[i['ori_filename'] for i in data['img_metas'].data[0]]
#         # print(files)
#         if 'COCO_val2014_000000000827.jpg' not in files :
#             batch_size = data['img'].size(0)
#             for _ in range(batch_size):
#                 prog_bar.update()
#             continue
#         else:
#             print("**********************************",flush=True)
#         with torch.no_grad():
#             result = model(return_loss=False, **data)
#
#         batch_size=len(result)
#
#         edge_index = result.reshape(batch_size//2,2,80,9)
#         classes=data_loader.dataset.CLASSES
#         if show or out_dir:
#             img_metas = data['img_metas'].data[0]
#             imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
#             assert len(imgs) == len(img_metas)
#
#             for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
#                 filename = img_meta['ori_filename']
#                 # if filename not in ['COCO_val2014_000000000415.jpg']:
#                 #     continue
#                 h, w, _ = img_meta['img_shape']
#                 img_show_init = img[:h, :w, :]
#                 ori_h, ori_w = img_meta['ori_shape'][:-1]
#                 img_show_init=mmcv.imresize(img_show_init, (ori_w, ori_h))
#                 PALETTE = data_loader.dataset.PALETTE
#                 target, edges = data['gt_label'][i], edge_index[i]
#                 index = np.arange(0, len(data_loader.dataset.CLASSES))
#                 index = index[target.numpy().astype(np.bool_)]
#                 w0 = 32/w*ori_w
#                 h0 = 32/h*ori_h
#                 edges=mmcv.load('./edge.json')[:2]
#                 edges=np.array(edges)
#                 index=[i for i in range(0,324)]
#                 for id in index:
#                     print(id)
#                     # edge_id=edges[:, id, :].sort(1).values
#                     # if (edge_id[0]==edge_id[1]).sum()!=9:
#                     #     continue
#                     for g in range(2):
#                         edge=edges[g]
#                         img_show =img_show_init.copy()
#                         out_file = './show/' + filename[:-4]+'_'+str(id)+'_'+str(g)+'.jpg'
#                         edg=edge[id][0]
#                         y = (edg // (ori_w // w0))
#                         x = edg % (ori_w // w0)
#                         x = int(w0 * (x))
#                         y = int(h0 * (y))
#                         img_show[y:y + int(h0), x:x + int(w0):] = img_show[y:y + int(h0), x:x + int(w0),
#                                                                   :] * 0.3 + np.array([255,0,0]).reshape(1,  1,
#                                                                                                                      -1) * 0.7
#
#                         for edg in edge[id][1:]:
#                             y = (edg // (ori_w//w0))
#                             x = edg % (ori_w//w0)
#                             x = int(w0 * (x))
#                             y = int(h0 * (y))
#                             img_show[y:y + int(h0),x:x + int(w0) :] = img_show[y:y + int(h0),x:x + int(w0), :] * 0.3 + np.array([255,255,0]).reshape(1,
#                                                                                                                           1,
#                                                                                                                          -1) * 0.7
#                         result_show = {
#                             # 'class': classes[id],
#                             # 'pred_label': pred_label[i],
#                             # 'pred_class': pred_class[i],
#                         }
#                         model.module.show_result(
#                             img_show,
#                             result_show,
#                             show=show,
#                             out_file=out_file,
#                             **show_kwargs)
#
#         batch_size = data['img'].size(0)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results
#
#
#
#

# def single_gpu_test(model,
#                     data_loader,
#                     show=False,
#                     out_dir=None,
#                     **show_kwargs):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     # useful_list=['COCO_val2014_000000003845.jpg', 'COCO_val2014_000000003938.jpg',
#     #    'COCO_val2014_000000004079.jpg', 'COCO_val2014_000000004227.jpg',
#     #    'COCO_val2014_000000004312.jpg', 'COCO_val2014_000000004395.jpg',
#     #    'COCO_val2014_000000005325.jpg', 'COCO_val2014_000000005437.jpg',
#     #    'COCO_val2014_000000006437.jpg', 'COCO_val2014_000000006673.jpg',
#     #    'COCO_val2014_000000006871.jpg', 'COCO_val2014_000000007320.jpg',
#     #    'COCO_val2014_000000007593.jpg', 'COCO_val2014_000000007784.jpg',
#     #    'COCO_val2014_000000008457.jpg', 'COCO_val2014_000000011721.jpg']
#     for i, data in enumerate(data_loader):
#         files=[i['ori_filename'] for i in data['img_metas'].data[0]]
#         # flag=np.sum([i in useful_list for i in files])
#         # if  flag<1:
#         # # print(files)
#         # # if 'COCO_val2014_000000000415.jpg' not in files and 'COCO_val2014_000000000827.jpg' not in files :
#         #     batch_size = data['img'].size(0)
#         #     for _ in range(batch_size):
#         #         prog_bar.update()
#         #     continue
#         with torch.no_grad():
#             result = model(return_loss=False, **data)
#
#         batch_size=len(result)
#
#         edge_index = result.reshape(batch_size//2,2,20,9)
#         classes=data_loader.dataset.CLASSES
#         if show or out_dir:
#             img_metas = data['img_metas'].data[0]
#             imgs = tensor2imgs(data['img'], **img_metas[0]['img_norm_cfg'])
#             assert len(imgs) == len(img_metas)
#
#             for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
#                 filename = img_meta['ori_filename']
#                 # if filename not in useful_list:
#                 #     continue
#                 # print(filename,flush=True)
#                 h, w, _ = img_meta['img_shape']
#                 img_show_init = img[:h, :w, :]
#                 ori_h, ori_w = img_meta['ori_shape'][:-1]
#                 img_show_init=mmcv.imresize(img_show_init, (ori_w, ori_h))
#                 model.module.show_result(
#                     img_show_init,
#                     {},
#                     show=False,
#                     out_file='show_usefulvoc/' + filename,
#                     **show_kwargs)
#                 # PALETTE = data_loader.dataset.PALETTE
#                 target, edges = data['gt_label'][i], edge_index[i]
#                 index = np.arange(0, len(data_loader.dataset.CLASSES))
#                 index = index[target.numpy().astype(np.bool_)]
#                 w0 = 32/w*ori_w
#                 h0 = 32/h*ori_h
#                 for id in index:
#                     # edge_id=edges[:, id, :].sort(1).values
#                     # if (edge_id[0]==edge_id[1]).sum()!=9:
#                     #     continue
#                     for g in range(2):
#                         edge=edges[g]
#                         img_show =img_show_init.copy()
#                         out_file = 'show_usefulvoc/' + filename[:-4]+'_'+str(id)+'_'+str(g)+'.jpg'
#                         for edg in edge[id]:
#                             y = (edg // (ori_w//w0))
#                             x = edg % (ori_w//w0)
#                             x = int(w0 * (x))
#                             y = int(h0 * (y))
#                             img_show[y:y + int(h0),x:x + int(w0) :] = img_show[y:y + int(h0),x:x + int(w0), :] * 0.1 + np.array([255,255,0]).reshape(1,
#                                                                                                                           1,
#                                                                                                                          -1) * 0.9
#                         result_show = {
#                             'class': classes[id],
#                             # 'pred_label': pred_label[i],
#                             # 'pred_class': pred_class[i],
#                         }
#                         model.module.show_result(
#                             img_show,
#                             result_show,
#                             show=False,
#                             out_file=out_file,
#                             **show_kwargs)
#
#         batch_size = data['img'].size(0)
#         for _ in range(batch_size):
#             prog_bar.update()
#     return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
