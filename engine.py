# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.plot_utils import PoseResultPlotor

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    args=None,
                    board_writer: SummaryWriter = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    data_length = len(data_loader)
    record_freq = 10
    

    # prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    # samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.poses:
            # add the 3d model ground truth to the targets
            for t in targets:
                t["model_scales"] = model.model_3d_net.scale_list
                t["model_centers"] = model.model_3d_net.center_list
                t["model_points"] = model.model_3d_net.meshes
                t["fps_points"] = model.model_3d_net.fps_points
        if args.poses:
            outputs = model(samples)
        else:
            outputs, _ = model(samples)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        
        # print(loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        assert not torch.isnan(grad_total_norm).any()
        optimizer.step()


        if board_writer is not None and i % record_freq == 0:
            board_writer.add_scalar('Train/loss', loss_value, i + epoch * data_length)
            board_writer.add_scalar('Train/class_loss', loss_dict_reduced_scaled['loss_ce'], i + epoch * data_length)
            board_writer.add_scalar('Train/bbox_loss', loss_dict_reduced_scaled['loss_bbox'], i + epoch * data_length)
            board_writer.add_scalar('Train/GIou_loss', loss_dict_reduced_scaled['loss_giou'],
                                    i + epoch * data_length)
            if args.poses:
                board_writer.add_scalar('Train/model3d_scales_loss', loss_dict_reduced_scaled['model3d_scales'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Train/model3d_centers_loss', loss_dict_reduced_scaled['model3d_centers'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Train/model3d_points_loss', loss_dict_reduced_scaled['model3d_points'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Train/pose_6dof_class_loss', loss_dict_reduced_scaled['pose_6dof_class'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Train/pose_6dof_add_loss', loss_dict_reduced_scaled['pose_6dof_add'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Train/pose_6dof_fps_points_3d_loss',
                                        loss_dict_reduced_scaled['pose_6dof_fps_points_3d'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Train/pose_6dof_rotation',
                                        loss_dict_reduced_scaled['pose_6dof_rotation'],
                                        i + epoch * data_length)
            board_writer.add_scalar('Train/lr', optimizer.param_groups[0]["lr"], i + epoch * data_length)




        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
            args, epoch: int = None,
            board_writer: SummaryWriter = None):
    # model.eval()
    # criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    if epoch is None:
        header = 'Test:'
    else:
        header = "eval"
    data_length = len(data_loader)
    record_freq = 10

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    li3d_evaluator = None
    pose_result_plotor = None

    if args.is_show_result:
        pose_result_plotor = PoseResultPlotor(args.result_path)

    translation_error = []
    rotation_error = []

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, record_freq, header)):
        bs = samples.tensors.shape[0]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if args.poses:
            for t in targets:
                t["model_scales"] = model.model_3d_net.scale_list
                t["model_centers"] = model.model_3d_net.center_list
                t["model_points"] = model.model_3d_net.meshes
                t["fps_points"] = model.model_3d_net.fps_points

        if args.poses:
            outputs = model(samples)
        else:
            outputs, _ = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_value = sum(loss_dict_reduced_scaled.values()).item()

        # print(loss_value)

        if board_writer is not None and i % 5 == 0:
            board_writer.add_scalar('Eval/loss', loss_value, i + epoch * data_length)
            board_writer.add_scalar('Eval/class_loss', loss_dict_reduced_scaled['loss_ce'], i + epoch * data_length)
            board_writer.add_scalar('Eval/bbox_loss', loss_dict_reduced_scaled['loss_bbox'], i + epoch * data_length)
            board_writer.add_scalar('Eval/GIou_loss', loss_dict_reduced_scaled['loss_giou'],
                                    i + epoch * data_length)
            if args.poses:
                board_writer.add_scalar('Eval/model3d_scales_loss', loss_dict_reduced_scaled['model3d_scales'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Eval/model3d_centers_loss', loss_dict_reduced_scaled['model3d_centers'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Eval/model3d_points_loss', loss_dict_reduced_scaled['model3d_points'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Eval/pose_6dof_class_loss', loss_dict_reduced_scaled['pose_6dof_class'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Eval/pose_6dof_add_loss', loss_dict_reduced_scaled['pose_6dof_add'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Eval/pose_6dof_fps_points_3d_loss',
                                        loss_dict_reduced_scaled['pose_6dof_fps_points_3d'],
                                        i + epoch * data_length)
                board_writer.add_scalar('Eval/pose_6dof_rotation', 
                                        loss_dict_reduced_scaled['pose_6dof_rotation'],
                                        i + epoch * data_length)

        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        model_results = None

        if args.poses:
            src_results, model_results = postprocessors['pose'](outputs, orig_target_sizes)
        else:
            src_results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = [src_results, model_results]

        if li3d_evaluator is not None:
            li3d_evaluator.update(res)

        # if pose_result_plotor is not None and i % record_freq == 0:
        if pose_result_plotor is not None and i % 1 == 0:
            ploter_head = f'test_i{i}' if epoch is None else f'evl_e{epoch}_i{i}'
            pose_result_plotor(samples.tensors, res, targets, args, ploter_head)

    pose_result_plotor.summarize(args.dataset_name)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if li3d_evaluator is not None:
        li3d_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if li3d_evaluator is not None:
        li3d_evaluator.accumulate()
        li3d_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats, li3d_evaluator
