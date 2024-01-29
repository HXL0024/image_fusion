import os

from PIL import Image
from torch.utils import data
from torchvision import transforms
import utils.utils_image as utils_image
import random



# data loader
class MSRS_data(data.Dataset):
    def __init__(self, args):
        super(MSRS_data, self).__init__()
        print('Dataset: MSRS for  Image Fusion.')
        self.args = args
        if(args.is_train):
            self.paths_inf = utils_image.get_image_paths(args.dataset_train_IR)
            self.paths_vis = utils_image.get_image_paths(args.dataset_train_VI_Y)
        elif(args.is_train == False):
            self.paths_inf = utils_image.get_image_paths(args.dataset_test_IR)
            self.paths_vis = utils_image.get_image_paths(args.dataset_test_VI_Y)
        self.n_channels = args.n_channels if args.n_channels else 3
        self.img_size = args.img_size if args.img_size else 256
    
    def __getitem__(self, index):
        inf_path = self.paths_inf[index]
        vis_path = self.paths_vis[index]

        img_inf = utils_image.imread_uint(inf_path, self.n_channels)
        img_vis = utils_image.imread_uint(vis_path, self.n_channels)

        if(self.args.is_train):
            H, W, _ = img_inf.shape
            rnd_h = random.randint(0, max(0, H - self.img_size))
            rnd_w = random.randint(0, max(0, W - self.img_size))
            patch_inf = img_inf[rnd_h:rnd_h + self.img_size, rnd_w:rnd_w + self.img_size, :]
            patch_vis = img_vis[rnd_h:rnd_h + self.img_size, rnd_w:rnd_w + self.img_size, :]

            mode = random.randint(0, 7)
            patch_inf = utils_image.augment_img(patch_inf, mode=mode)
            patch_vis = utils_image.augment_img(patch_vis, mode=mode)

            img_inf = utils_image.uint2tensor3(patch_inf)
            img_vis = utils_image.uint2tensor3(patch_vis)
            return {'img_inf': img_inf, 'img_vis': img_vis, 'inf_path': inf_path, 'vis_path': vis_path}
        
        elif(self.args.is_train == False):
            
            img_inf = utils_image.uint2tensor3(img_inf)
            img_vis = utils_image.uint2tensor3(img_vis)


            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_inf = utils_image.single2tensor3(img_inf)
            img_vis = utils_image.single2tensor3(img_vis)

            return {'img_inf': img_inf, 'img_vis': img_vis, 'inf_path': inf_path, 'vis_path': vis_path}

        
    
    def __len__(self):
        return len(self.paths_inf)
