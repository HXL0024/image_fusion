
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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from utils.utils_segmentationMetric import SegmentationMetric
import math
from tqdm import tqdm



def parse_args() -> Namespace:

    parser = argparse.ArgumentParser(description='Train fusion model')
    parser.add_argument('--cuda', type=bool, default=False, help='use GPU or not')
    parser.add_argument('--is_train', type=bool, default=False, help='train or test')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/models', help='model path')
    parser.add_argument('--iter_number_segmenter1', type=int, default=160000, help='iter number')
    parser.add_argument('--iter_number_mask_transformer', type=int, default=160000, help='iter number')
    parser.add_argument('--dataset_test_fusion', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/test/our-main', help='path for test dataset')
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/test/Images', help='Images save path')
    parser.add_argument('--log_dir', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/test/logs', help='log path')
    parser.add_argument('--n_channels', type=int, default=1, help='number of channels')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for swin transformer')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers for swin transformer')
    parser.add_argument('--image_size', type=int, help='image size for swin transformer')
    parser.add_argument('--mode', type=int, default=2, help='test mode')
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

def test_segmenter1():
    device = torch.device('cuda' if args.cuda else 'cpu')
    # 模型路径
    model_segmenter1_path = os.path.join(args.model_path, str(args.iter_number_Segmenter1) + '_Segmenter1G.pth')
    if os.path.exists(model_segmenter1_path):
        print('Loading model for G [{:s}] ...'.format(model_segmenter1_path))
    else:
        print('No model for G [{:s}] ...'.format(model_segmenter1_path))
        sys.exit()
    model_mask_transformer_path = os.path.join(args.model_path, str(args.iter_number_mask_transformer) + '_MaskTransformerG.pth')
    if os.path.exists(model_mask_transformer_path):
        print('Loading model for G [{:s}] ...'.format(model_mask_transformer_path))
    else:
        print('No model for G [{:s}] ...'.format(model_mask_transformer_path))
        sys.exit()

    # netFusion & netSegmenter
    model_Segmenter1 = define_model_Segmenter1(args)
    model_Segmenter1.eval()
    model_Segmenter1 = model_Segmenter1.to(device)
    model_Mask_Transformer = define_model_MaskTransformer(args)
    model_Mask_Transformer.eval()
    model_Mask_Transformer = model_Mask_Transformer.to(device)

    # dataset_test
    test_set = Seg_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Test dataset length: ", len(test_loader))




    for i, test_data in enumerate(test_loader):
        img_name = test_data['inf_path'][0]
        img_inf = test_data['img_inf'].to(device) 
        img_vis = test_data['img_vis'].to(device)
        img_label = test_data['img_label'].to(device)
        starttime = time.time()

        with torch.no_grad():
            out, H, W, H_ori, W_ori = model_Segmenter1(img_inf, img_vis)
            seg1 = model_Mask_Transformer(out, (H, W), (H_ori, W_ori))
            # 将模型的分割结果seg1进行插值，使其大小与img_label相同
            seg1 = F.interpolate(seg1, size=img_label.shape[1:], mode='bilinear', align_corners=False)
            print(np.shape(seg1))
            print(np.shape(img_label))
            # 将img_label从GPU移到CPU，然后转换为NumPy数组。
            # 使用squeeze函数去掉维度为1的轴，最后通过flatten将多维数组展平为一维数组。
            img_label = img_label.cpu().numpy().squeeze().flatten()
            # 从插值后的seg1中获取每个像素最可能的类别索引，即使用argmax(1)获取沿着通道维度的最大值索引。
            # 然后，将其从GPU移到CPU，进行展平操作，得到一维数组prediction，包含了模型对每个像素的预测类别。
            prediction = seg1.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480

def define_model_Segmenter1(args):
    from models.network_segmenter1 import Segmenter as net
    model = net(image_size=[480,640])
    model_path = os.path.join(args.model_path, str(args.iter_number_segmenter1) + '_Segmenter1G.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model, strict=True)
    return model

def define_model_MaskTransformer(args):
    from models.network_mask_vit import Decoder_Seg as net
    model = net(n_cls=9)
    model_path = os.path.join(args.model_path, str(args.iter_number_mask_transformer) + '_Mask_TransformerG.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model, strict=True)
    return model

class MscEval(object):
    def __init__(
        self,
        args,
        segmenter1,
        Mask_Transformer,
        dataloader,
        # scales=[1],
        scales=[0.75, 0.9, 1, 1.1, 1.2, 1.25],
        # scales=[1, 1.2],
        n_classes=9,
        lb_ignore=255,
        flip=True,
    ):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        ## dataloader
        self.dataloader = dataloader
        self.segmenter1 = segmenter1
        self.Mask_Transformer = Mask_Transformer
        self.args = args
        print(self.scales)

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0] - H, size[1] - W
        hst, hed = margin_h // 2, margin_h // 2 + H
        wst, wed = margin_w // 2, margin_w // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def get_palette(self):
        unlabelled = [0, 0, 0]
        car = [64, 0, 128]
        person = [64, 64, 0]
        bike = [0, 128, 192]
        curve = [0, 0, 192]
        car_stop = [128, 128, 0]
        guardrail = [64, 64, 128]
        color_cone = [192, 128, 128]
        bump = [192, 64, 0]
        palette = np.array(
            [
                unlabelled,
                car,
                person,
                bike,
                curve,
                car_stop,
                guardrail,
                color_cone,
                bump,
            ]
        )
        return palette

    def visualize(self, save_name, predictions):
        palette = self.get_palette()
        # print(predictions.shape)
        # 遍历predictions
        # for (i, pred) in enumerate(predictions):
        pred = predictions
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save(save_name)

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        N, C, H, W = im.size()
        long_size, short_size = (H, W) if H > W else (W, H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W - cropsize) / stride) + 1
            n_y = math.ceil((H - cropsize) / stride) + 1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = (
                        min(H, stride * iy + cropsize),
                        min(W, stride * ix + cropsize),
                    )
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        return prob

    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self, args, Method='NestFuse'):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        device = torch.device('cuda' if self.args.cuda else 'cpu')
        lb_ignore = [255]
        seg_metric = SegmentationMetric(n_classes, device=device)          
        dloader = tqdm(self.dataloader)
        for i, (imgs, label, fn) in enumerate(dloader):
            # if not fn[0] == '00037N.png':
            #     continue
            # print(fn[0])
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            probs_torch = torch.zeros((N, self.n_classes, H, W))
            probs_torch = probs_torch.to(device)
            probs_torch.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs_torch += prob
                probs += prob.detach().cpu()            
            seg_results = torch.argmax(probs_torch, dim=1, keepdim=True)
            seg_metric.addBatch(seg_results, label.to(device), lb_ignore)
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            for i in range(1):
                outpreds = preds[i]
                name = fn[i]
                folder_path = os.path.join('BANet', Method)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, name)
                # img = Image.fromarray(np.uint8(outpreds))
                # img.save(file_path)
                self.visualize(file_path, outpreds)
            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (
            np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist)
        )
        mIOU = np.mean(IOUs)
        mIOU = mIOU
        mIoU_torch = np.array(seg_metric.meanIntersectionOverUnion().item())
        IoU_list = IOUs.tolist()
        IoU_list.append(mIOU)
        IoU_list = [round(100 * i, 2) for i in IoU_list]
        # print('{} | IoU:{}, mIoU:{:.4f}'.format(Method, IoU_list, mIoU_torch))
        return mIOU, IoU_list

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
    test_segmenter1()
    print("Training Done!")