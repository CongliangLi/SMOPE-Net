# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
# from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from configs import config, cfg
from datasets_labelimg3d import build_dataset, get_orig_data_from_dataset
from torch.utils.tensorboard import SummaryWriter
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=cfg["lr"], type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=cfg["lr_backbone"], type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=cfg["batch_size"], type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=cfg["end_epoch"], type=int)
    parser.add_argument('--lr_drop', default=cfg["lr_drop"], type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # ================ Add ================
    parser.add_argument("--save_weights_nm", default=cfg["save_weights_num"])
    parser.add_argument("--class_num", default=config["class_num"], type=str)
    # Tensorboard
    parser.add_argument('--tensorboard', default=cfg["tensorboard"], type=bool)
    parser.add_argument("--is_show_result", default=config["is_show_result"], type=bool)
    parser.add_argument("--plot_threshold", default=config["plot_threshold"], type=float)


    # obj model
    parser.add_argument("--model_class_num", default=config["model"]["model_class_num"], type=int)
    parser.add_argument("--model_name", default=config["model"]["model_name"], type=list)
    parser.add_argument("--model_num_samples", default=config["model"]["model_num_samples"], type=int)
    parser.add_argument("--model_focal_gamma", default=cfg["model_focal_gamma"], type=float)



    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # * poses
    parser.add_argument('--poses', action='store_true',
                        default=config["poses"],
                        help="Train segmentation head if the flag is provided")



    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--focal_gamma', default=cfg["focal_gamma"], type=float)
    parser.add_argument('--focal_alpha', default=cfg["focal_alpha"], type=float)
    
    parser.add_argument('--cls_loss_weight', default=cfg["cls_loss_weight"], type=float)
    parser.add_argument('--bbox2d_loss_weight', default=cfg["bbox2d_loss_weight"], type=float)
    parser.add_argument('--giou_loss_weight', default=cfg["giou_loss_weight"], type=float)
    parser.add_argument('--model3d_loss_weight', default=cfg["model3d_loss_weight"], type=float)
    parser.add_argument('--pose6dof_loss_weight', default=cfg["pose6dof_loss_weight"], type=float)
    parser.add_argument('--model3d_scales_weight', default=cfg["model3d_scales_weight"], type=float)
    parser.add_argument('--model3d_centers_weight', default=cfg["model3d_centers_weight"], type=float)
    parser.add_argument('--model3d_points_weight', default=cfg["model3d_points_weight"], type=float)
    parser.add_argument('--model3d_chamfer_weight', default=cfg["model3d_chamfer_weight"], type=float)
    parser.add_argument('--model3d_edge_weight', default=cfg["model3d_edge_weight"], type=float)
    parser.add_argument('--model3d_normal_weight', default=cfg["model3d_normal_weight"], type=float)
    parser.add_argument('--model3d_laplacian_weight', default=cfg["model3d_laplacian_weight"], type=float)
    parser.add_argument('--model_6dof_class_weight', default=cfg["model_6dof_class_weight"], type=float)
    parser.add_argument('--model_6dof_add_weight', default=cfg["model_6dof_add_weight"], type=float)
    parser.add_argument('--model_6dof_fps_points_weight', default=cfg["model_6dof_fps_points_weight"], type=float)

    parser.add_argument("--model_class_weights", default=cfg["model_class_weights"], type=list)
                       


    
    # dataset parameters
    parser.add_argument('--dataset_name', default=config['dataset_name'])
    parser.add_argument('--dataset_path', default=config["dataset_path"], type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default=config["output_dir"],
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default=config["device"],
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=cfg["resume"], help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=config["eval"])
    parser.add_argument('--num_workers', default=cfg["num_workers"], type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_name == "KITTI3D" or "UA-DETRAC3D":
        base_ds = get_orig_data_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        # if not args.eval:
        #     test_stats, li3d_evaluator = evaluate(model, criterion, postprocessors,
        #                                       data_loader_val, base_ds, device, cfg["output_dir"], args)
    
    if args.eval:
        test_stats, li3d_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, cfg["output_dir"], args)
        # if args.output_dir:
        #     utils.save_on_master(li3d_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    writer = None
    if args.tensorboard:
        if args.board_path is not None:
            writer = SummaryWriter(args.board_path)
        else:
            writer = SummaryWriter(os.path.join(os.getcwd(), "tensorboard"))


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)


        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args,
            writer)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [Path(args.checkpoint_paths)/'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(Path(args.checkpoint_paths) / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            past_checkpoint_path = Path(args.checkpoint_paths)/ f'checkpoint{(epoch-args.save_weights_nm * 5):04}.pth'
            if os.path.exists(past_checkpoint_path):
                os.remove(past_checkpoint_path)

        # envaluate
        if epoch % 5 == 0:
            test_stats, li3d_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, 
                args.output_dir, args, epoch, writer
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir) / f'{args.backbone}_{args.dataset_name}_b{args.batch_size}_lr{args.lr}_nq{args.num_queries}_gamma{args.focal_gamma}_alpha{args.focal_alpha}'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.tensorboard and args.output_dir is not None:
        args.board_path = Path(os.path.join(args.output_dir, "tensorboard")) / f'{args.backbone}_{args.dataset_name}_b{args.batch_size}_lr{args.lr}_nq{args.num_queries}_gamma{args.focal_gamma}_alpha{args.focal_alpha}'
        args.board_path.mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        args.checkpoint_paths = os.path.join(args.output_dir, "checkpoints")

        if not os.path.exists(args.checkpoint_paths):
            os.makedirs(args.checkpoint_paths)

    if args.output_dir and args.is_show_result:
        args.result_path = os.path.join(args.output_dir, "results")
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
    main(args)
