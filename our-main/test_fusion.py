
import argparse
from argparse import Namespace
import os
from utils.utils_logger import logger_info
import logging
import random
import numpy as np
import torch
import sys
from models.ModelPlain import ModelPlain
from data_loader.seg_data import Seg_data
from torch.utils.data import DataLoader
import time
from utils import utils_image as util
from PIL import Image



def parse_args() -> Namespace:

    parser = argparse.ArgumentParser(description='Train fusion model')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--is_train', type=bool, default=False, help='train or test')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/models', help='model path')
    parser.add_argument('--iter_number_fusion', type=int, default=78000, help='iter number')
    parser.add_argument('--iter_number_segmenter1', type=int, default=42000, help='iter number')
    parser.add_argument('--dataset_test_IR', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Infrared', help='path for test dataset')
    parser.add_argument('--dataset_test_VI_Y', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Visible', help='path for test dataset')
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/our-main', help='Images save path')
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/test/logs', help='log path')
    parser.add_argument('--n_channels', type=int, default=1, help='number of channels')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for swin transformer')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers for swin transformer')
    parser.add_argument('--image_size', type=int, help='image size for swin transformer')
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

def test_fusion():
    device = torch.device('cuda' if args.cuda else 'cpu')
    # 模型路径
    model_fusion_path = os.path.join(args.model_path, str(args.iter_number_fusion) + '_FusionG.pth')
    if os.path.exists(model_fusion_path):
        print('Loading model for G [{:s}] ...'.format(model_fusion_path))
    else:
        print('No model for G [{:s}] ...'.format(model_fusion_path))
        sys.exit()

    model_segmenter_path = os.path.join(args.model_path, str(args.iter_number_segmenter) + '_SegmenterG.pth')
    if os.path.exists(model_segmenter_path):
        print('Loading model for G [{:s}] ...'.format(model_segmenter_path))
    else:
        print('No model for G [{:s}] ...'.format(model_segmenter_path))
        sys.exit()

    # netFusion & netSegmenter
    model_fusion = define_model_fusion(args)
    model_fusion.eval()
    model_fusion = model_fusion.to(device)
    model_segmenter = define_model_segmenter1(args)
    model_segmenter.eval()
    model_segmenter = model_segmenter.to(device)
    # dataset_test
    test_set = Seg_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Test dataset length: ", len(test_loader))
    for i, test_data in enumerate(test_loader):
        img_name = test_data['inf_path'][0]
        img_inf = test_data['img_inf'].to(device)
        img_vis = test_data['img_vis'].to(device)
        starttime = time.time()

        with torch.no_grad():
            out0, out1 = model_segmenter.forward_fusion(img_inf, img_vis)
            image_fusion = model_fusion(img_inf, img_vis, out0, out1)

            image_vis_ycrcb = RGB2YCrCb(img_vis)
            fusion_ycrb = torch.cat((image_fusion, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]), dim=1)
            fusion_image = YCrCb2RGB(fusion_ycrb)
            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            # 图像数据进行缩放、维度重新排列和标准化
            fused_image = np.uint8(255.0 * fused_image)
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)

        endtime = time.time()
        image = fused_image[0, :, :, :]
        image = Image.fromarray(image)
        save_name = os.path.join(args.save_path, os.path.basename(img_name))
        image.save(save_name)
        print("[{}/{}]  Saving fused image to : {}, Processing time is {:4f} s".format(i+1, len(test_loader), save_name, endtime - starttime))

def define_model_fusion(args):
    from models.network_fusion import MyFusion as net
    model = net()
    fusion_model_path = os.path.join(args.model_path, str(args.iter_number_fusion) + '_FusionG.pth')
    pretrained_model = torch.load(fusion_model_path)
    model.load_state_dict(pretrained_model, strict=True)
    return model


def define_model_segmenter1(args):
    from models.network_segmenter1 import Segmenter as net
    model = net(image_size=[480,640])
    model_path = os.path.join(args.model_path, str(args.iter_number_segmenter1) + '_SegmenterG.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model, strict=True)
    return model


def YCrCb2RGB(input_im):
    device = torch.device('cuda' if args.cuda else 'cpu')
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def RGB2YCrCb(input_im):
    device = torch.device('cuda' if args.cuda else 'cpu')
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

if __name__ == '__main__':
    args = parse_args()

    roots = [args.save_path, args.log_dir]
    for directory in roots:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 日志信息
    logger_name = 'train'
    logger_info(logger_name, os.path.join(args.log_dir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(args)

    init_seeds()
    test_fusion()
    print("Training Done!")