import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.network_vit import VisionTransformer , PatchUnEmbed
from models.network_mask_vit import MaskTransformer

class Segmenter(nn.Module):
    def __init__(
        self,
        image_size,
        n_cls = 9
    ):
        super().__init__()
        self.n_cls = n_cls
        # 还需要一些参数
        self.encoder = VisionTransformer(image_size = image_size, channels = 3, n_cls=n_cls, d_ff = 4*768, d_model = 768, n_heads = 6, n_layers = 6, patch_size = 16)
        # 参数还需要调整
        self.PatchUnEmbed = PatchUnEmbed(image_size = image_size, patch_size = 16, in_chans = 3, embed_dim = 768)
        self.patch_size = self.encoder1.patch_size

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder)
        # .union(
        #     append_prefix_no_weight_decay("decoder.", self.decoder))
        return nwd_params

    def forward(self, fuse_img):
        H_ori, W_ori = fuse_img.size(2), fuse_img.size(3)
        fuse_img = padding(inf_img, self.patch_size)
        H, W = fuse_img.size(2), fuse_img.size(3)
        inf_img = self.encoder(fuse_img)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1
        fuse_img = fuse_img[:, num_extra_tokens:]
        # 由B C H W变为B 2C H W
        return fuse_img, H, W, H_ori, W_ori


    def get_attention_map_enc(self, im, layer_id):
        return self.encoder1.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]
        return self.decoder.get_attention_map(x, layer_id)
    
def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded

