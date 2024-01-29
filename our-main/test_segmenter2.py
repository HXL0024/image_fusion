
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



def parse_args() -> Namespace:

    parser = argparse.ArgumentParser(description='Train fusion model')
    parser.add_argument('--cuda', type=bool, default=False, help='use GPU or not')
    parser.add_argument('--is_train', type=bool, default=False, help='train or test')
    parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/models', help='model path')
    parser.add_argument('--iter_number_fusion', type=int, default=54000, help='iter number')
    parser.add_argument('--iter_number_segmenter1', type=int, default=160000, help='iter number')
    parser.add_argument('--dataset_test_IR', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Infrared', help='path for test dataset')
    parser.add_argument('--dataset_test_VI_Y', type=str, default='/content/drive/MyDrive/ImageFusion/MFNet/test_all/Visible', help='path for test dataset')
    parser.add_argument('--save_path', type=str, default='/content/drive/MyDrive/ImageFusion/our-main/Model_Seg/test/Images', help='Images save path')
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

def test_segmenter2():
    device = torch.device('cuda' if args.cuda else 'cpu')
    # 模型路径
    model_segmenter2_path = os.path.join(args.model_path, str(args.iter_number_Segmenter2) + '_Segmenter2G.pth')
    if os.path.exists(model_segmenter2_path):
        print('Loading model for G [{:s}] ...'.format(model_segmenter2_path))
    else:
        print('No model for G [{:s}] ...'.format(model_segmenter2_path))
        sys.exit()
    model_mask_transformer_path = os.path.join(args.model_path, str(args.iter_number_mask_transformer) + '_MaskTransformerG.pth')
    if os.path.exists(model_mask_transformer_path):
        print('Loading model for G [{:s}] ...'.format(model_mask_transformer_path))
    else:
        print('No model for G [{:s}] ...'.format(model_mask_transformer_path))
        sys.exit()

    # netFusion & netSegmenter
    model_Segmenter2 = define_model_Segmenter2(args)
    model_Segmenter2.eval()
    model_Segmenter2 = model_Segmenter2.to(device)
    model_Mask_Transformer = define_model_MaskTransformer(args)
    model_Mask_Transformer.eval()
    model_Mask_Transformer = model_Mask_Transformer.to(device)

    # dataset_test
    test_set = Seg_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Test dataset length: ", len(test_loader))

    conf_total = np.zeros((9, 9))
    file = './val_seg_'+seg1+'.txt'
    file_o = open(file,'a+')

    for i, test_data in enumerate(test_loader):
        img_name = test_data['inf_path'][0]
        img_mask = test_data['img_mask'].to(device)
        img_label = test_data['img_label'].to(device)
        starttime = time.time()

        with torch.no_grad():
            out, H, W, H_ori, W_ori = model_Segmenter2(img_mask)
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
            
            conf = confusion_matrix(y_true=img_label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf

        # 每个类别的精确度、召回率和交并比
        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print('\n###########################################################################')
        print(
            "*precision_per_class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
            % (precision_per_class[0], precision_per_class[1], precision_per_class[2], precision_per_class[3], precision_per_class[4],
               precision_per_class[5], precision_per_class[6], precision_per_class[7], precision_per_class[8]))
        print(
            "* iou per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5],
            iou_per_class[6], iou_per_class[7], iou_per_class[8]))
        print("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        print("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write('\n###########################################################################'+'\n')
        file_o.write(
            "*precision_per_class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
            % (precision_per_class[0], precision_per_class[1], precision_per_class[2], precision_per_class[3], precision_per_class[4],
               precision_per_class[5], precision_per_class[6], precision_per_class[7], precision_per_class[8]))
        file_o.write(
            "* iou per class: \n    unlabeled: %.6f, car: %.6f, person: %.6f, bike: %.6f, curve: %.6f, car_stop: %.6f, guardrail: %.6f, color_cone: %.6f, bump: %.6f" \
            % (
            iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5],
            iou_per_class[6], iou_per_class[7], iou_per_class[8]))
        file_o.write("\n* average values (np.mean(x)): \n ACC: %.6f, iou: %.6f" \
              % (precision_per_class.mean(), iou_per_class.mean()))
        file_o.write("* average values (np.mean(np.nan_to_num(x))): \n ACC: %.6f, iou: %.6f" \
              % (np.mean(np.nan_to_num(precision_per_class)), np.mean(np.nan_to_num(iou_per_class))))
        file_o.write(str(epoch)+'===>'+str(precision_per_class.mean())+'---'+str(iou_per_class.mean())+'\n')
        file_o.close()
        print('\n###########################################################################')
        return iou_per_class.mean()

def define_model_Segmenter2(args):
    from models.network_segmenter2 import Segmenter as net
    model = net(image_size=[480,640])
    model_path = os.path.join(args.model_path, str(args.iter_number_segmenter2) + '_Segmenter2G.pth')
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
    

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class

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