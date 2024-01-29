import argparse
from argparse import Namespace
import random
# import wandb
import os


import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from data_loader.msrs_data import MSRS_data
from data_loader.seg_data import Seg_data
from models.ModelPlain import ModelPlain
from models.common import find_last_checkpoint
from utils.utils_logger import logger_info
import math

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='Train fusion model')
    # parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--is_train', type=bool, default=True, help='train or test')
    # 加载预训练的模型 通过检查迭代点的方式继续训练
    # 路径信息
    parser.add_argument('--dataset_train_IR', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/train_all/Infrared', help='path for train dataset IR')
    parser.add_argument('--dataset_train_VI_Y', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/train_all/Visible', help='path for train dataset VI_Y')
    parser.add_argument('--dataset_train_mask', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/train_all/Mask', help='path for train dataset mask')
    parser.add_argument('--dataset_train_label', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/train_all/Label', help='path for train dataset label')
    parser.add_argument('--dataset_test_IR', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Infrared', help='path for test dataset')
    parser.add_argument('--dataset_test_VI_Y', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/train_all/Visible', help='path for test dataset')
    parser.add_argument('--dataset_test_mask', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Mask', help='path for test dataset')
    parser.add_argument('--dataset_test_label', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Label', help='path for test dataset')
    parser.add_argument('--root', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg', help='root path for dataset')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/models', help='save model path')
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/logs', help='log path')
    parser.add_argument('--images_dir', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/images', help='images path')

    # 训练参数
    # 读取图片文件的通道数
    parser.add_argument('--n_channels', type=int, default=3, help='number of channels')
    parser.add_argument('--scale', type=int, default=1)  #border vif
    parser.add_argument('--batch_size', type=int, default=3, help='batch size for training')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')

    # net_Fusion_model
    parser.add_argument('--upscale', type=int, default=1, help='upscale factor')
    parser.add_argument('--in_chans', type=int, default=3, help='input channels')
    parser.add_argument('--image_size', type=int, default=384, help='image size')
    parser.add_argument('--window_size', type=int, default=8, help='window size')
    parser.add_argument('--img_range', type=float, default=1.0, help='img range')
    parser.add_argument('--depths', type=int, default=[6,6,6,6], nargs='+', help='depths for each layer')
    parser.add_argument('--embed_dim', type=int, default=60, help='embed dim for each layer')
    parser.add_argument('--num_heads', type=int, default=[6,6,6,6], nargs='+', help='num of heads for each layer')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='mlp ratio')
    parser.add_argument('--upsampler', type=str, help='upsampler')
    parser.add_argument('--resi_connection', type=str, default='1conv', help='resi connection')
    # parser.add_argument('--scale', type=int, default=1, help='scale factor')
    # net_Segmenter_model
    parser.add_argument('--n_classes', type=int, default=9, help='number of classes')

    # 调优参数
    # SegmenterG_param_strict
    parser.add_argument('--E_decay', type=float, default=0.999, help='decay rate for the ExponentialMovingAverage')
    parser.add_argument('--FusionG_optimizer_reuse', type=bool, default=True, help='reuse the optimizer for FusionG')
    parser.add_argument('--Segmenter1G_optimizer_reuse', type=bool, default=True, help='reuse the optimizer for Segmenter1G')
    parser.add_argument('--Segmenter2G_optimizer_reuse', type=bool, default=True, help='reuse the optimizer for Segmenter2G')
    parser.add_argument('--Mask_TransformerG_optimizer_reuse', type=bool, default=True, help='reuse the optimizer for Mask_TransformerG')
    parser.add_argument('--FusionG_optimizer_lr', default=2e-05, type=float, help='learning rate for G optimizer')
    parser.add_argument('--FusionG_optimizer_clipgrad', default=0.0, type=float, help='clip gradient for G optimizer')
    parser.add_argument('--FusionG_scheduler_milestones', default=[250000, 400000, 450000, 475000, 500000], nargs='+', type=int, help='milestones for G scheduler')
    parser.add_argument('--FusionG_scheduler_gamma', default=0.5, type=float, help='gamma for G scheduler')
    parser.add_argument('--FusionG_regularizer_clipstep', type=float, help='clip regularization for G')
    parser.add_argument('--FusionG_regularizer_orthstep', type=float, help='orthogonal regularization for G')
    parser.add_argument('--Segmenter1G_optimizer_lr', default=2e-05, type=float, help='learning rate for G optimizer')
    parser.add_argument('--Segmenter1G_optimizer_clipgrad', default=0.0, type=float, help='clip gradient for G optimizer')
    parser.add_argument('--Segmenter1G_scheduler_milestones', nargs='+', type=int, help='milestones for G scheduler')
    parser.add_argument('--Segmenter1G_scheduler_gamma', default=0.5, type=float, help='gamma for G scheduler')
    parser.add_argument('--Segmenter1G_regularizer_clipstep', type=float, help='clip regularization for G')
    parser.add_argument('--Segmenter1G_regularizer_orthstep', type=float, help='orthogonal regularization for G')
    parser.add_argument('--Segmenter2G_optimizer_lr', default=2e-05, type=float, help='learning rate for G optimizer')
    parser.add_argument('--Segmenter2G_optimizer_clipgrad', default=0.0, type=float, help='clip gradient for G optimizer')
    parser.add_argument('--Segmenter2G_scheduler_milestones', nargs='+', type=int, help='milestones for G scheduler')
    parser.add_argument('--Segmenter2G_scheduler_gamma', default=0.5, type=float, help='gamma for G scheduler')
    parser.add_argument('--Segmenter2G_regularizer_clipstep', type=float, help='clip regularization for G')
    parser.add_argument('--Segmenter2G_regularizer_orthstep', type=float, help='orthogonal regularization for G')
    parser.add_argument('--Mask_TransformerG_optimizer_lr', default=2e-05, type=float, help='learning rate for G optimizer')
    parser.add_argument('--Mask_TransformerG_optimizer_clipgrad', default=0.0, type=float, help='clip gradient for G optimizer')
    parser.add_argument('--Mask_TransformerG_scheduler_milestones', nargs='+', type=int, help='milestones for G scheduler')
    parser.add_argument('--Mask_TransformerG_scheduler_gamma', default=0.5, type=float, help='gamma for G scheduler')
    parser.add_argument('--Mask_TransformerG_regularizer_clipstep', type=float, help='clip regularization for G')
    parser.add_argument('--Mask_TransformerG_regularizer_orthstep', type=float, help='orthogonal regularization for G')
    parser.add_argument('--G_param_strict', type=bool, default=True, help='strict for G')
    parser.add_argument('--E_param_strict', type=bool, default=True, help='strict for E')
    parser.add_argument('--G_param_reuse', type=bool, default=True, help='reuse for G')
    parser.add_argument('--E_param_reuse', type=bool, default=True, help='reuse for E')
    parser.add_argument('--checkpoint_print', default=200, type=int, help='print frequency for checkpoint')
    parser.add_argument('--checkpoint_save', default=6000, type=int, help='save frequency for checkpoint')
    parser.add_argument('--checkpoint_test', default=1000, type=int, help='test frequency for checkpoint')
    # 损失权重
    parser.add_argument('--FusionG_lossfn_weight', type=float, default=1.0, help='weight for the G_lossfn')
    return parser.parse_args()

def init_seeds():
    seed = random.randint(1, 10000)
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark , cudnn.deterministic = (False, True) if seed == 0 else (True, False)

# 数据集加载
def train_fusion():
    logger.info('Fusion model training...')
    # 数据集加载
    logger.info('Loading dataset ...\n')
    train_dataset = Seg_data(args)
    # 日志
    train_size = int(math.ceil(len(train_dataset) / args.batch_size))
    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_dataset), train_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, drop_last = True, pin_memory=True)
    # test_dataset = MSRS_data(args.dataset_test_path)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 最大的迭代数作为最新的检查点
    init_iter_FusionG, init_path_FusionG = find_last_checkpoint(args.save_dir, net_type='FusionG')
    init_iter_FusionE, init_path_FusionE = find_last_checkpoint(args.save_dir, net_type='FusionE')
    args.pretrained_netFusionG_path = init_path_FusionG
    args.pretrained_netFusionE_path = init_path_FusionE
    init_iter_optimizer_FusionG, init_path_optimizer_FusionG = find_last_checkpoint(args.save_dir, net_type='optimizerFusionG')
    args.pretrained_optimizer_FusionG_path = init_path_optimizer_FusionG
    current_step_Fusion = max(init_iter_FusionG, init_iter_FusionE, init_iter_optimizer_FusionG)

    init_iter_Segmenter1G, init_path_Segmenter1G = find_last_checkpoint(args.save_dir, net_type='Segmenter1G')
    init_iter_Segmenter1E, init_path_Segmenter1E = find_last_checkpoint(args.save_dir, net_type='Segmenter1E')
    args.pretrained_netSegmenter1G_path = init_path_Segmenter1G
    args.pretrained_netSegmenter1E_path = init_path_Segmenter1E
    init_iter_optimizer_Segmenter1G, init_path_optimizer_Segmenter1G = find_last_checkpoint(args.save_dir, net_type='optimizerSegmenter1G')
    args.pretrained_optimizer_Segmenter1G_path = init_path_optimizer_Segmenter1G
    current_step_Segmenter1 = max(init_iter_Segmenter1G, init_iter_Segmenter1E, init_iter_optimizer_Segmenter1G)
    # model
    model = ModelPlain(args)
    model.define_FusionG()
    model.init_train_Fusion()
    model.define_Segmenter1G()
    model.init_train_Segmenter1()
    for epoch in range(1000):
        for i, train_data in enumerate(train_loader):
            current_step_Fusion += 1
            # ----------------------------
            # 1) update learning rate
            # ----------------------------
            model.update_learning_rate_Fusion(current_step_Fusion)

            # ----------------------------
            # 2) feed patch pairs
            # ----------------------------
            model.feed_data_Fusion(train_data)

            # ----------------------------
            # 3) optimize parameters
            # ----------------------------
            model.optimize_parameters_Fusion(current_step_Fusion)

            # ----------------------------
            # 4) training information
            # ----------------------------
            if current_step_Fusion % args.checkpoint_print == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, Fusion_iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step_Fusion, model.current_learning_rate_Fusion())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
                # wandb.log(logs)
            
            # ----------------------------
            # 5) save model
            # ----------------------------
            if current_step_Fusion % args.checkpoint_save == 0:
                save_dir = args.save_dir
                save_filename = '{}_{}.pth'.format(current_step_Fusion, 'E')
                save_path = os.path.join(save_dir, save_filename)
                model.save_Fusion(current_step_Fusion)
                logger.info('Saving the model Fusion. Save path is:{}'.format(save_path))


def train_seg1():

    logger.info('Segmenter1 Training ...\n')
    # 数据集加载
    logger.info('Loading dataset ...\n')
    train_dataset = Seg_data(args)
    # 日志
    train_size = int(math.ceil(len(train_dataset) / args.batch_size))
    logger.info('Number of segmenter_train images: {:,d}, iters: {:,d}'.format(len(train_dataset), train_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, drop_last = True, pin_memory=True)

    # 最大的迭代数作为最新的检查点
    init_iter_Segmenter1G, init_path_Segmenter1G = find_last_checkpoint(args.save_dir, net_type='Segmenter1G')
    init_iter_Segmenter1E, init_path_Segmenter1E = find_last_checkpoint(args.save_dir, net_type='Segmenter1E')
    args.pretrained_netSegmenter1G_path = init_path_Segmenter1G
    args.pretrained_netSegmenter1E_path = init_path_Segmenter1E
    init_iter_optimizer_Segmenter1G, init_path_optimizer_Segmenter1G = find_last_checkpoint(args.save_dir, net_type='optimizerSegmenter1G')
    args.pretrained_optimizer_Segmenter1G_path = init_path_optimizer_Segmenter1G
    current_step_Segmenter1 = max(init_iter_Segmenter1G, init_iter_Segmenter1E, init_iter_optimizer_Segmenter1G)

    init_iter_Mask_TransformerG, init_path_Mask_TransformerG = find_last_checkpoint(args.save_dir, net_type='Mask_TransformerG')
    init_iter_Mask_TransformerE, init_path_Mask_TransformerE = find_last_checkpoint(args.save_dir, net_type='Mask_TransformerE')
    args.pretrained_Mask_TransformerG_path = init_path_Mask_TransformerG
    args.pretrained_Mask_TransformerE_path = init_path_Mask_TransformerE
    init_iter_optimizer_Mask_TransformerG, init_path_optimizer_Mask_TransformerG = find_last_checkpoint(args.save_dir, net_type='optimizerMaskTransformerG')
    args.pretrained_optimizer_Mask_TransformerG_path = init_path_optimizer_Mask_TransformerG
    current_step_Mask_Transformer = max(init_iter_Mask_TransformerG, init_iter_Mask_TransformerE, init_iter_optimizer_Mask_TransformerG)

    model = ModelPlain(args)
    model.define_Segmenter1G()
    model.init_train_Segmenter1()

    model.define_Mask_TransformerG()
    model.init_train_Mask_Transformer()

    for epoch in range(256):
        for i, train_data in enumerate(train_loader):
            current_step_Segmenter1 += 1
            current_step_Mask_Transformer += 1
            model.update_learning_rate_Segmenter1(current_step_Segmenter1)
            model.update_learning_rate_Mask_Transformer(current_step_Mask_Transformer)
            model.feed_data_Segmenter1(train_data)
            model.optimize_parameters_Segmenter1(current_step_Segmenter1, current_step_Mask_Transformer)
            if current_step_Segmenter1 % args.checkpoint_print == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, Segmenter1_iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step_Segmenter1, model.current_learning_rate_Segmenter1())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
                # wandb.log(logs)
            if current_step_Mask_Transformer % args.checkpoint_print == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, Mask_Transformer_iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step_Mask_Transformer, model.current_learning_rate_Mask_Transformer())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
            
            if current_step_Segmenter1 % args.checkpoint_save == 0:
                save_dir = args.save_dir
                save_filename = '{}_{}.pth'.format(current_step_Segmenter1, 'E')
                save_path = os.path.join(save_dir, save_filename)
                model.save_Segmenter1(current_step_Segmenter1)
                logger.info('Saving the model. Save path is:{}'.format(save_path))

            if current_step_Mask_Transformer % args.checkpoint_save == 0:
                save_dir = args.save_dir
                save_filename = '{}_{}.pth'.format(current_step_Mask_Transformer, 'E')
                save_path = os.path.join(save_dir, save_filename)
                model.save_Mask_Transformer(current_step_Mask_Transformer)
                logger.info('Saving the model. Save path is:{}'.format(save_path))


def train_seg2():

    logger.info('Segmenter2 Training ...\n')
    # 数据集加载
    logger.info('Loading dataset ...\n')
    train_dataset = Seg_data(args)
    # 日志
    train_size = int(math.ceil(len(train_dataset) / args.batch_size))
    logger.info('Number of segmenter_train images: {:,d}, iters: {:,d}'.format(len(train_dataset), train_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, drop_last = True, pin_memory=True)

    # 最大的迭代数作为最新的检查点
    init_iter_Segmenter2G, init_path_Segmenter2G = find_last_checkpoint(args.save_dir, net_type='Segmenter2G')
    init_iter_Segmenter2E, init_path_Segmenter2E = find_last_checkpoint(args.save_dir, net_type='Segmenter2E')
    args.pretrained_netSegmenter2G_path = init_path_Segmenter2G
    args.pretrained_netSegmenter2E_path = init_path_Segmenter2E
    init_iter_optimizer_Segmenter2G, init_path_optimizer_Segmenter2G = find_last_checkpoint(args.save_dir, net_type='optimizerSegmenter2G')
    args.pretrained_optimizer_Segmenter2G_path = init_path_optimizer_Segmenter2G
    current_step_Segmenter2 = max(init_iter_Segmenter2G, init_iter_Segmenter2E, init_iter_optimizer_Segmenter2G)

    init_iter_Mask_TransformerG, init_path_Mask_TransformerG = find_last_checkpoint(args.save_dir, net_type='Mask_TransformerG')
    init_iter_Mask_TransformerE, init_path_Mask_TransformerE = find_last_checkpoint(args.save_dir, net_type='Mask_TransformerE')
    args.pretrained_Mask_TransformerG_path = init_path_Mask_TransformerG
    args.pretrained_Mask_TransformerE_path = init_path_Mask_TransformerE
    init_iter_optimizer_Mask_TransformerG, init_path_optimizer_Mask_TransformerG = find_last_checkpoint(args.save_dir, net_type='optimizer_Mask_TransformerG')
    args.pretrained_optimizer_Mask_TransformerG_path = init_path_optimizer_Mask_TransformerG
    current_step_Mask_Transformer = max(init_iter_Mask_TransformerG, init_iter_Mask_TransformerE, init_iter_optimizer_Mask_TransformerG)

    model = ModelPlain(args)
    model.define_Segmenter2G()
    model.init_train_Segmenter2()
    model.define_Mask_TransformerG()
    model.init_train_Mask_Transformer()


    for epoch in range(256):
        for i, train_data in enumerate(train_loader):
            current_step_Segmenter2 += 1
            current_step_Mask_Transformer += 1
            model.update_learning_rate_Segmenter2(current_step_Segmenter2)
            model.update_learning_rate_Mask_Transformer(current_step_Mask_Transformer)
            model.feed_data_Segmenter2(train_data)
            model.optimize_parameters_Segmenter2(current_step_Segmenter2, current_step_Mask_Transformer)
            if current_step_Segmenter2 % args.checkpoint_print == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, Segmenter2_iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step_Segmenter2, model.current_learning_rate_Segmenter2())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
                # wandb.log(logs)
            if current_step_Mask_Transformer % args.checkpoint_print == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, Mask_Transformer_iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step_Mask_Transformer, model.current_learning_rate_Mask_Transformer())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
            
            if current_step_Segmenter2 % args.checkpoint_save == 0:
                save_dir = args.save_dir
                save_filename = '{}_{}.pth'.format(current_step_Segmenter2, 'E')
                save_path = os.path.join(save_dir, save_filename)
                model.save_Segmenter2(current_step_Segmenter2)
                logger.info('Saving the model Segmenter2. Save path is:{}'.format(save_path))

            if current_step_Mask_Transformer % args.checkpoint_save == 0:
                save_dir = args.save_dir
                save_filename = '{}_{}.pth'.format(current_step_Mask_Transformer, 'E')
                save_path = os.path.join(save_dir, save_filename)
                model.save_Mask_Transformer(current_step_Mask_Transformer)
                logger.info('Saving the model MaskTransformer. Save path is:{}'.format(save_path))


if __name__ == '__main__':
    args = parse_args()

    roots = [args.save_dir, args.log_dir, args.images_dir]
    for directory in roots:
        if not os.path.exists(directory):
            os.makedirs(directory)
    logger_name = 'train'
    logger_info(logger_name, os.path.join(args.log_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(args)

    init_seeds()
    train_seg1()
    print("Training Done!")


