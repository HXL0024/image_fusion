
def define_G(args):
    from models.network_swinfusion import SwinFusion as net
    from models.network_myfusion import MyFusion as net1
    netG = net1(
            upscale=args.upscale,
            in_chans=args.in_chans,
            img_size=args.img_size,
            window_size=args.window_size,
            img_range=args.img_range,
            # depths=args.depths,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            upsampler=args.upsampler,
            resi_connection=args.resi_connection)
    
    # ----------------------------------------
    # initialize weights 暂时不需要
    # ----------------------------------------
    # if opt['is_train']:
    #     init_weights(netG,
    #                  init_type=opt_net['init_type'],
    #                  init_bn_type=opt_net['init_bn_type'],
    #                  gain=opt_net['init_gain'])
    
    return netG


def define_FusionG(args):
    from models.network_fusion import MyFusion as net
    netFusionG = net()
    return netFusionG

def define_Segmenter1G(args):
    from models.network_segmenter1 import Segmenter as net
    netSegmenter1G = net(image_size=[480, 640])
    return netSegmenter1G

def define_Segmenter2G(args):
    from models.network_segmenter2 import Segmenter as net
    netSegmenter2G = net(image_size=[480, 640])
    return netSegmenter2G

def define_Mask_TransformerG(args):
    from models.network_mask_vit import Decoder_Seg as net
    netMaskTransformer = net(n_cls=9)
    return netMaskTransformer
