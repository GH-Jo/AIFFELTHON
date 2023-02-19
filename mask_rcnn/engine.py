import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from . import utils

#gt_msk_coords set([(x,y),....])
#pred_msk_coords set([(x,y),....])
def get_seg_iou(gt_msk, pred_msk):
    gt_msk_coords = np.where(gt_msk >= 1)
    pred_msk_coords = np.where(pred_msk >= 1)
    GT = set(gt_msk_coords)
    S  = set(pred_msk_coords)
    TP = GT&S
    FP = (GT|S)-GT
    FN = (GT|S)-S
    iou = len(TP)/(len(TP)+len(FP)+len(FN))
    return iou
#ground_truth [min_x,min_y, max_x,max_y]
#pred         [min_x,min_y, max_x,max_y]
#IoU = TP / (TP+FP+FN)
def get_bbox_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from .mylog import print_log

def imshow_grid(img_org, inst_bboxes, inst_masks,inst_areas, save_file=None):
    plt.close('all')
    plt.clf()
    col = 2
    row = 1
    fig = plt.figure(figsize=(int(8*col),int(8*row)))

    inp = img_org.transpose((1, 2, 0))
    inp = inp * 255 # 0~1 to 255
    inp = np.array(inp,dtype=np.uint8)
    im2 = inp.copy()
    area_list = []
    #colors = random.sample(range(1,255),len(inst_bboxes))
    for i in range(len(inst_bboxes)):
        msk=inst_masks[i]
        rec=inst_bboxes[i].astype(dtype=np.uint32)
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        im2[:, :, 0][msk>0] = r
        im2[:, :, 1][msk > 0] = g
        im2[:, :, 2][msk > 0] = b
        cv2.rectangle(im2, (rec[0],rec[1]),(rec[2],rec[3]),(b,g,r),2)
        area_list.append(int(np.sqrt(inst_areas[i])))

    title  = ','.join(str(e) for e in area_list)
    #print_log(title)
    ax = fig.add_subplot(row,col,1)
    ax.set_axis_off()
    ax.set_title(title)
    ax.imshow(inp)
    ax2 = fig.add_subplot(row,col,2)
    ax2.set_axis_off()
    ax2.set_title('bbox_num= '+str(len(inst_bboxes)))
    ax2.imshow(im2)
    if save_file is not None:
        plt.savefig(save_file)
        plt.close('all')
        plt.clf()
    else:
        plt.show()

import os
def iou_cal(model, images, targets, call_cnt, exp_dir_path):
    batch_cnt = len(images)
    total_bbox_iou =0.0
    total_seg_iou =0.0
    model.eval()
    with torch.no_grad():
        predict = model(images)
        for i in range(batch_cnt):
            gt=targets[i]
            r = predict[i]
            imshow_grid(images[i].detach().cpu().numpy(), \
                        gt['boxes'].detach().cpu().numpy(), \
                        gt['masks'].detach().cpu().numpy(), \
                        gt['area'].detach().cpu().numpy(), \
                        save_file=os.path.join(exp_dir_path,str(call_cnt*batch_cnt+i)+'check.png'))
            '''
            ap,precisions,recalls,overlaps =\
                compute_ap(gt['boxes'].cpu().numpy(), \
                    gt['labels'].cpu().numpy(), \
                        gt['masks'].cpu().numpy(), \
                    r['boxes'].cpu().numpy(), \
                        r['labels'].cpu().numpy(), \
                            r['scores'].cpu().numpy(),r['masks'].cpu().numpy())
            print(f'ap:{ap},precison:{precisions},recalls:{recalls},overlaps:{overlaps}')
            ''' 
            
    model.train()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,exp_dir_path):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    '''
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    '''
    call_cnt = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        
        #iou_cal(model, images, targets, call_cnt, exp_dir_path)
        call_cnt +=1

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
