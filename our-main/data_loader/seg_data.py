from torch.utils.data import Dataset
import numpy as np
import imageio
import utils.utils_image as utils_image
import random

# 加载mask和label
class Seg_data(Dataset):
    def __init__(self, args):
        super(Seg_data, self).__init__()
        print('Dataset: MSRS for  Image Segmenter.')
        self.args = args
        if(args.is_train):
            self.paths_inf = utils_image.get_image_paths(args.dataset_train_IR)
            self.paths_vis = utils_image.get_image_paths(args.dataset_train_VI_Y)
            self.paths_label = utils_image.get_image_paths(args.dataset_train_label)
            self.paths_mask = utils_image.get_image_paths(args.dataset_train_mask)
        elif(args.is_train == False):
            if(args.mode == 1):
                self.paths_inf = utils_image.get_image_paths(args.dataset_test_IR)
                self.paths_vis = utils_image.get_image_paths(args.dataset_test_VI_Y)
            elif(args.mode == 2):
                self.paths_fusion = utils_image.get_image_paths(args.dataset_test_fusion)
                self.paths_label = utils_image.get_image_paths(args.dataset_test_label)
        self.image_size_H = args.image_size_H if args.image_size_H else 256
        self.image_size_W = args.image_size_W if args.image_size_W else 256
    
    def __getitem__(self, index):
        if(self.args.is_train):
            inf_path = self.paths_inf[index]
            vis_path = self.paths_vis[index]
            label_path = self.paths_label[index]
            mask_path = self.paths_mask[index]
            # Fusion Image ：channels ：3
            img_inf = np.asarray(imageio.imread(inf_path))
            img_inf = img_inf[:,:,np.newaxis]
            img_inf = np.concatenate((img_inf,img_inf,img_inf),axis=2)

            img_vis = np.asarray(imageio.imread(vis_path))
            img_mask = np.asarray(imageio.imread(mask_path))
            img_mask = img_mask[:,:, np.newaxis]
            img_mask = np.concatenate([img_mask,img_mask,img_mask],axis=2)
            img_label = np.asarray(imageio.imread(label_path))
            H, W, _= np.shape(img_inf)
            rnd_h = random.randint(0, max(0, H - self.image_size_H))
            rnd_w = random.randint(0, max(0, W - self.image_size_W))
            patch_inf = img_inf[rnd_h:rnd_h + self.image_size_H, rnd_w:rnd_w + self.image_size_W, ]
            patch_vis = img_vis[rnd_h:rnd_h + self.image_size_H, rnd_w:rnd_w + self.image_size_W, ]
            patch_label = img_label[rnd_h:rnd_h + self.image_size_H, rnd_w:rnd_w + self.image_size_W, ]

            # mode = random.randint(0, 7)
            # patch_inf = utils_image.augment_img(patch_inf, mode=mode)
            # patch_vis = utils_image.augment_img(patch_vis, mode=mode)
            # patch_mask = utils_image.augment_img(patch_mask, mode=mode)
            # patch_label = utils_image.augment_img(patch_label, mode=mode)
            img_inf = utils_image.uint2tensor3(patch_inf)
            img_vis = utils_image.uint2tensor3(patch_vis)
            # img_mask = utils_image.uint2tensor3(img_mask)
            return {'img_label': img_label,'label_path': label_path, 'img_inf': img_inf, 'img_vis': img_vis, 'inf_path': inf_path, 'vis_path': vis_path, 'img_mask': img_mask, 'mask_path': mask_path}
        elif(self.args.is_train == False):
            if(self.args.mode == 1):
                inf_path = self.paths_inf[index]
                vis_path = self.paths_vis[index]
                # Fusion Image ：channels ：3
                img_inf = np.asarray(imageio.imread(inf_path))
                img_inf = img_inf[:,:,np.newaxis]
                img_inf = np.concatenate((img_inf,img_inf,img_inf),axis=2)
                img_vis = np.asarray(imageio.imread(vis_path))
                # img_mask = np.asarray(imageio.imread(mask_path))
                # img_mask = img_mask[:,:, np.newaxis]
                # img_mask = np.concatenate([img_mask,img_mask,img_mask],axis=2)

                img_inf = utils_image.uint2tensor3(img_inf)
                img_vis = utils_image.uint2tensor3(img_vis)
                # img_mask = utils_image.uint2tensor3(img_mask)

                return {'img_inf': img_inf, 'img_vis': img_vis, 'inf_path': inf_path, 'vis_path': vis_path}
            elif(self.args.mode == 2):
                fusion_path = self.paths_fusion[index]
                label_path = self.paths_label[index]

                img_fusion = np.asarray(imageio.imread(fusion_path))
                img_label = np.asarray(imageio.imread(label_path))
                img_fusion = utils_image.uint2tensor3(img_fusion)

                return {'img_fusion': img_fusion, 'img_label': img_label, 'fusion_path': fusion_path, 'label_path': label_path}

            
        

    
    def __len__(self):
        return len(self.paths_vis)