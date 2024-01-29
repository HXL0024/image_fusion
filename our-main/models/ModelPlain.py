from collections import OrderedDict
from functools import wraps
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from utils.utils_regularizers import regularizer_orth, regularizer_clip


from models.ModelBase import ModelBase
from models.network_define import define_FusionG, define_Mask_TransformerG, define_Segmenter1G, define_Segmenter2G
import os
from torch.utils.tensorboard import SummaryWriter
class ModelPlain(ModelBase):
    def __init__(self, args):
        super(ModelPlain, self).__init__(args)
        self.args = args
        tensorboard_path = os.path.join(self.args.save_dir, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)

    # 定义模型
    # 融合
    def define_FusionG(self):
        self.netFusionG = define_FusionG(self.args)
        self.netFusionG = self.model_to_device(self.netFusionG)  
        if self.args.E_decay > 0:
            self.netFusionE = define_FusionG(self.args).to(self.device).eval()
    # 语义分割Encoder1
    def define_Segmenter1G(self):
        self.netSegmenter1G = define_Segmenter1G(self.args)
        self.netSegmenter1G = self.model_to_device(self.netSegmenter1G)  
        # if self.args.E_decay > 0:
        #     self.netSegmenter1E = define_Segmenter1G(self.args).to(self.device).eval()
    # 语义分割Encoder2
    def define_Segmenter2G(self):
        self.netSegmenter2G = define_Segmenter2G(self.args)
        self.netSegmenter2G = self.model_to_device(self.netSegmenter2G)  
        if self.args.E_decay > 0:
            self.netSegmenter2E = define_Segmenter2G(self.args).to(self.device).eval()
    # Decoder
    def define_Mask_TransformerG(self):
        self.Mask_TransformerG = define_Mask_TransformerG(self.args)
        self.Mask_TransformerG = self.model_to_device(self.Mask_TransformerG)  
        if self.args.E_decay > 0:
            self.Mask_TransformerE = define_Mask_TransformerG(self.args).to(self.device).eval()

    def init_train_Fusion(self):
        self.load_Fusion()
        self.netFusionG.train()
        self.define_loss_Fusion()
        self.define_optimizer_Fusion()
        self.load_optimizers_Fusion()
        self.define_scheduler_Fusion()
        self.log_dict = OrderedDict()

    def init_train_Segmenter1(self):
        self.load_Segmenter1()
        self.netSegmenter1G.train()
        # self.define_loss_Segmenter()
        self.define_optimizer_Segmenter1()
        self.load_optimizers_Segmenter1()
        self.define_scheduler_Segmenter1()

    def init_train_Segmenter2(self):
        self.load_Segmenter2()
        self.netSegmenter2G.train()
        # self.define_loss_Segmenter()
        self.define_optimizer_Segmenter2()
        self.load_optimizers_Segmenter2()
        self.define_scheduler_Segmenter2()


    def init_train_Mask_Transformer(self):
        self.load_Mask_Transformer()
        self.Mask_TransformerG.train()
        self.define_loss_Segmenter()
        self.define_optimizer_Mask_Transformer()
        self.load_optimizers_Mask_Transformer()
        self.define_scheduler_Mask_Transformer()
        self.log_dict = OrderedDict()


    # 加载segmenter模型参数
    # def init_train_fusion_Segmenter(self):
    #     load_path_SegmenterG = self.args.pretrained_netSegmenterG_path
    #     if load_path_SegmenterG is not None:
    #         print('Loading model for SegmenterG [{:s}] ...'.format(load_path_SegmenterG))
    #         self.load_network(load_path_SegmenterG, self.netSegmenterG, strict=self.args.G_param_strict, param_key='params')

    # ----------------------------------------
    # load pre-trained G model 加载预训练的模型
    # ----------------s------------------------
    def load_Fusion(self):
        load_path_FusionG = self.args.pretrained_netFusionG_path
        if load_path_FusionG is not None:
            print('Loading model for FusionG [{:s}] ...'.format(load_path_FusionG))
            self.load_network(load_path_FusionG, self.netFusionG, strict=self.args.G_param_strict, param_key='params')
        load_path_FusionE = self.args.pretrained_netFusionE_path
        if self.args.E_decay > 0:
            if load_path_FusionE is not None:
                print('Loading model for FusionE [{:s}] ...'.format(load_path_FusionE))
                self.load_network(load_path_FusionE, self.netFusionE, strict=self.args.E_param_strict, param_key='params_ema')
            else:
                print('Copying model for FusionE ...')
                self.update_FusionE(0)
            self.netFusionE.eval()
    
    def load_Segmenter1(self):
        load_path_Segmenter1G = self.args.pretrained_netSegmenter1G_path
        if load_path_Segmenter1G is not None:
            print('Loading model for Segmenter1G [{:s}] ...'.format(load_path_Segmenter1G))
            self.load_network(load_path_Segmenter1G, self.netSegmenter1G, strict=self.args.G_param_strict, param_key='params')
        # load_path_Segmenter1E = self.args.pretrained_netSegmenter1E_path
        # if self.args.E_decay > 0:
        #     if load_path_Segmenter1E is not None:
        #         print('Loading model for Segmenter1E [{:s}] ...'.format(load_path_Segmenter1E))
        #         self.load_network(load_path_Segmenter1E, self.netSegmenter1E, strict=self.args.E_param_strict, param_key='params_ema')
        #     else:
        #         print('Copying model for Segmenter1E ...')
        #         self.update_Segmenter1E(0)
        #     self.netSegmenter1E.eval()

    def load_Segmenter2(self):
        load_path_Segmenter2G = self.args.pretrained_netSegmenter2G_path
        if load_path_Segmenter2G is not None:
            print('Loading model for Segmenter2G [{:s}] ...'.format(load_path_Segmenter2G))
            self.load_network(load_path_Segmenter2G, self.netSegmenter2G, strict=self.args.G_param_strict, param_key='params')
        load_path_Segmenter2E = self.args.pretrained_netSegmenter2E_path
        if self.args.E_decay > 0:
            if load_path_Segmenter2E is not None:
                print('Loading model for Segmenter2E [{:s}] ...'.format(load_path_Segmenter2E))
                self.load_network(load_path_Segmenter2E, self.netSegmenter2E, strict=self.args.E_param_strict, param_key='params_ema')
            else:
                print('Copying model for Segmenter2E ...')
                self.update_Segmenter2E(0)
            self.netSegmenter2E.eval()

    def load_Mask_Transformer(self):
        load_path_Mask_TransformerG = self.args.pretrained_Mask_TransformerG_path
        if load_path_Mask_TransformerG is not None:
            print('Loading model for MaskTransformerG [{:s}] ...'.format(load_path_Mask_TransformerG))
            self.load_network(load_path_Mask_TransformerG, self.Mask_TransformerG, strict=self.args.G_param_strict, param_key='params')
        load_path_Mask_TransformerE = self.args.pretrained_Mask_TransformerE_path
        if self.args.E_decay > 0:
            if load_path_Mask_TransformerE is not None:
                print('Loading model for MaskTransformerE [{:s}] ...'.format(load_path_Mask_TransformerE))
                self.load_network(load_path_Mask_TransformerE, self.Mask_TransformerE, strict=self.args.E_param_strict, param_key='params_ema')
            else:
                print('Copying model for MaskTransformerE ...')
                self.update_Mask_TransformerE(0)
            self.Mask_TransformerE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers_Fusion(self):
        load_path_optimizer_FusionG = self.args.pretrained_optimizer_FusionG_path
        if load_path_optimizer_FusionG is not None and self.args.FusionG_optimizer_reuse:
            print('Loading optimizerFusionG [{:s}] ...'.format(load_path_optimizer_FusionG))
            self.load_optimizer(load_path_optimizer_FusionG, self.FusionG_optimizer)

    def load_optimizers_Segmenter1(self):
        load_path_optimizer_Segmenter1G = self.args.pretrained_optimizer_Segmenter1G_path
        if load_path_optimizer_Segmenter1G is not None and self.args.Segmenter1G_optimizer_reuse:
            print('Loading optimizerSegmenter1G [{:s}] ...'.format(load_path_optimizer_Segmenter1G))
            self.load_optimizer(load_path_optimizer_Segmenter1G, self.Segmenter1G_optimizer)

    def load_optimizers_Segmenter2(self):
        load_path_optimizer_Segmenter2G = self.args.pretrained_optimizer_Segmenter2G_path
        if load_path_optimizer_Segmenter2G is not None and self.args.Segmenter2G_optimizer_reuse:
            print('Loading optimizerSegmenter2G [{:s}] ...'.format(load_path_optimizer_Segmenter2G))
            self.load_optimizer(load_path_optimizer_Segmenter2G, self.Segmenter2G_optimizer)
    
    def load_optimizers_Mask_Transformer(self):
        load_path_optimizer_Mask_TransformerG = self.args.pretrained_optimizer_Mask_TransformerG_path
        if load_path_optimizer_Mask_TransformerG is not None and self.args.Mask_TransformerG_optimizer_reuse:
            print('Loading optimizerMaskTransformerG [{:s}] ...'.format(load_path_optimizer_Mask_TransformerG))
            self.load_optimizer(load_path_optimizer_Mask_TransformerG, self.Mask_TransformerG_optimizer)
    
    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save_Fusion(self, iter_label):
        self.save_network(self.save_dir, self.netFusionG, 'FusionG', iter_label)
        if self.args.E_decay > 0:
            self.save_network(self.save_dir, self.netFusionE, 'FusionE', iter_label)
        if self.args.FusionG_optimizer_reuse:
            self.save_optimizer(self.save_dir, self.FusionG_optimizer, 'optimizerFusionG', iter_label)

    def save_Segmenter1(self, iter_label):
        self.save_network(self.save_dir, self.netSegmenter1G, 'Segmenter1G', iter_label)
        if self.args.E_decay > 0:
            self.save_network(self.save_dir, self.netSegmenter1E, 'Segmenter1E', iter_label)
        if self.args.Segmenter1G_optimizer_reuse:
            self.save_optimizer(self.save_dir, self.Segmenter1G_optimizer, 'optimizerSegmenter1G', iter_label)

    def save_Segmenter2(self, iter_label):
        self.save_network(self.save_dir, self.netSegmenter2G, 'Segmenter2G', iter_label)
        if self.args.E_decay > 0:
            self.save_network(self.save_dir, self.netSegmenter2E, 'Segmenter2E', iter_label)
        if self.args.Segmenter2G_optimizer_reuse:
            self.save_optimizer(self.save_dir, self.Segmenter2G_optimizer, 'optimizerSegmenter2G', iter_label)

    
    def save_Mask_Transformer(self, iter_label):
        self.save_network(self.save_dir, self.Mask_TransformerG, 'Mask_TransformerG', iter_label)
        if self.args.E_decay > 0:
            self.save_network(self.save_dir, self.Mask_TransformerE, 'Mask_TransformerE', iter_label)
        if self.args.Mask_TransformerG_optimizer_reuse:
            self.save_optimizer(self.save_dir, self.Mask_TransformerG_optimizer, 'optimizerMaskTransformerG', iter_label)
    
    # ----------------------------------------
    # define loss 损失函数
    # ----------------------------------------
    def define_loss_Fusion(self):
        from loss.loss_vif import fusion_loss_vif
        self.FusionG_lossfn = fusion_loss_vif().to(self.device)
        self.FusionG_lossfn_weight = self.args.FusionG_lossfn_weight if self.args.FusionG_lossfn_weight else 1.0

    def define_loss_Segmenter(self):
        # ignore_index = 255为什么去掉？
        self.SegmenterG_lossfn = torch.nn.CrossEntropyLoss(ignore_index=255).to(self.device)

    # ----------------------------------------
    # define optimizer 定义优化器 为生成器网络中需要更新的参数创建一个Adam优化器
    # ----------------------------------------
    def define_optimizer_Fusion(self):
        FusionG_optim_params = []
        for k, v in self.netFusionG.named_parameters():
            if v.requires_grad:
                FusionG_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.FusionG_optimizer = Adam(FusionG_optim_params, lr=self.args.FusionG_optimizer_lr, weight_decay=0)

    def define_optimizer_Segmenter1(self):
        Segmenter1G_optim_params = []
        for k, v in self.netSegmenter1G.named_parameters():
            if v.requires_grad:
                Segmenter1G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.Segmenter1G_optimizer = Adam(Segmenter1G_optim_params, lr=self.args.Segmenter1G_optimizer_lr, weight_decay=0)

    def define_optimizer_Segmenter2(self):
        Segmenter2G_optim_params = []
        for k, v in self.netSegmenter2G.named_parameters():
            if v.requires_grad:
                Segmenter2G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.Segmenter2G_optimizer = Adam(Segmenter2G_optim_params, lr=self.args.Segmenter2G_optimizer_lr, weight_decay=0)

    def define_optimizer_Mask_Transformer(self):
        Mask_TransformerG_optim_params = []
        for k, v in self.Mask_TransformerG.named_parameters():
            if v.requires_grad:
                Mask_TransformerG_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.Mask_TransformerG_optimizer = Adam(Mask_TransformerG_optim_params, lr=self.args.Mask_TransformerG_optimizer_lr, weight_decay=0)
    
    # ----------------------------------------
    # define scheduler, only "MultiStepLR" is supported
    # ----------------------------------------  
    def define_scheduler_Fusion(self):
        self.schedulers_Fusion.append(lr_scheduler.MultiStepLR(self.FusionG_optimizer,
                                                        self.args.FusionG_scheduler_milestones if self.args.FusionG_scheduler_milestones else [250000, 400000, 450000, 475000, 500000],
                                                        self.args.FusionG_scheduler_gamma if self.args.FusionG_scheduler_gamma else 0.5))
    def define_scheduler_Segmenter1(self):
        self.schedulers_Segmenter1.append(lr_scheduler.MultiStepLR(self.Segmenter1G_optimizer,
                                                        self.args.Segmenter1G_scheduler_milestones if self.args.Segmenter1G_scheduler_milestones else [250000, 400000, 450000, 475000, 500000],
                                                        self.args.Segmenter1G_scheduler_gamma if self.args.Segmenter1G_scheduler_gamma else 0.5))                                                   

    def define_scheduler_Segmenter2(self):
        self.schedulers_Segmenter2.append(lr_scheduler.MultiStepLR(self.Segmenter2G_optimizer,
                                                        self.args.Segmenter2G_scheduler_milestones if self.args.Segmenter2G_scheduler_milestones else [250000, 400000, 450000, 475000, 500000],
                                                        self.args.Segmenter2G_scheduler_gamma if self.args.Segmenter2G_scheduler_gamma else 0.5))        

    def define_scheduler_Mask_Transformer(self):
        self.schedulers_Mask_Transformer.append(lr_scheduler.MultiStepLR(self.Mask_TransformerG_optimizer,
                                                        self.args.Mask_TransformerG_scheduler_milestones if self.args.Mask_TransformerG_scheduler_milestones else [250000, 400000, 450000, 475000, 500000],
                                                        self.args.Mask_TransformerG_scheduler_gamma if self.args.Mask_TransformerG_scheduler_gamma else 0.5))        
    # ----------------------------------------
    # feed under/over data
    # ----------------------------------------
    def feed_data_Fusion(self, data):
        self.img_inf = data['img_inf'].to(self.device)
        self.img_vis = data['img_vis'].to(self.device)

    def feed_data_Segmenter1(self, data):
        self.img_inf = data['img_inf'].to(self.device)
        self.img_vis = data['img_vis'].to(self.device)
        self.img_label = data['img_label'].to(self.device)

    def feed_data_Segmenter2(self, data):
        self.img_mask = data['img_mask'].to(self.device)
        self.img_label = data['img_label'].to(self.device)
    
    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netFusionG_forward(self):
        with torch.no_grad():
            seg_inf, seg_vis = self.netSegmenter1G.forward_fusion(self.img_inf, self.img_vis)
        # (B,C,H,W)
        self.FusionE = self.netFusionG(self.img_inf, self.img_vis, seg_inf, seg_vis)

    def netSegmenter1G_forward(self):
        out, H, W, H_ori, W_ori = self.netSegmenter1G(self.img_inf, self.img_vis)
        self.Segmenter1E = self.Mask_TransformerG(out, (H, W), (H_ori, W_ori))

    def netSegmenter2G_forward(self):
        out, H, W, H_ori, W_ori = self.netSegmenter2G(self.img_mask)
        self.Segmenter2E = self.Mask_TransformerG(out, (H, W), (H_ori, W_ori))


    # ----------------------------------------
    # update parameters and get loss 更新参数获取loss
    # ----------------------------------------
    def optimize_parameters_Fusion(self, current_step_Fusion):
        # 生成器的梯度清零，接收新一轮的梯度信息
        self.FusionG_optimizer.zero_grad()
        self.netFusionG_forward()
        img_vis = self.img_vis[:, 0:1, :, :]
        img_inf = self.img_inf[:, 0:1, :, :]

        ## 损失函数，需要修改添加什么的
        total_loss, loss_text, loss_int, loss_ssim = self.FusionG_lossfn(img_inf, img_vis, self.FusionE)
        G_loss = self.FusionG_lossfn_weight * total_loss      
        G_loss.backward()
        # ------------------------------------
        # clip_grad 设置梯度裁剪的阈值 ，
        # ------------------------------------
        FusionG_optimizer_clipgrad = self.args.FusionG_optimizer_clipgrad if self.args.FusionG_optimizer_clipgrad else 0
        if FusionG_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.FusionG_optimizer_clipgrad, norm_type=2)
        self.FusionG_optimizer.step()

        # ------------------------------------
        # regularizer 正则化器 通常用于对网络权重进行正则化，以防止过拟合。
        # ------------------------------------
        FusionG_regularizer_orthstep = self.args.FusionG_regularizer_orthstep if self.args.FusionG_regularizer_orthstep else 0
        if FusionG_regularizer_orthstep > 0 and current_step_Fusion % FusionG_regularizer_orthstep == 0 and current_step_Fusion % self.args.checkpoint_save != 0:
            self.netFusionG.apply(regularizer_orth)
        FusionG_regularizer_clipstep = self.args.FusionG_regularizer_clipstep if self.args.FusionG_regularizer_clipstep else 0
        if FusionG_regularizer_clipstep > 0 and current_step_Fusion % FusionG_regularizer_clipstep == 0 and current_step_Fusion % self.args.checkpoint_save != 0:
            self.netFusionG.apply(regularizer_clip)

        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['Text_loss'] = loss_text.item()
        self.log_dict['Int_loss'] = loss_int.item()
        self.log_dict['SSIM_loss'] = loss_ssim.item()

        self.writer.add_scalar('G_loss', G_loss.item(), current_step_Fusion)
        self.writer.add_scalar('Text_loss', loss_text.item(), current_step_Fusion)
        self.writer.add_scalar('Int_loss', loss_int.item(), current_step_Fusion)
        self.writer.add_scalar('SSIM_loss', loss_ssim.item(), current_step_Fusion)

        if self.args.E_decay > 0:
            self.update_FusionE(self.args.E_decay)
    
    def optimize_parameters_Segmenter1(self, current_step_Segmenter1, current_step_Mask_Transformer):
        self.Segmenter1G_optimizer.zero_grad()
        self.Mask_TransformerG_optimizer.zero_grad()
        self.netSegmenter1G_forward()
        Segmenter1_loss = self.SegmenterG_lossfn(self.Segmenter1E, self.img_label.type(torch.long))
        Segmenter1_loss.backward()
        
        # ------------------------------------
        # clip_grad 
        # ------------------------------------
        Segmenter1G_optimizer_clipgrad = self.args.Segmenter1G_optimizer_clipgrad if self.args.Segmenter1G_optimizer_clipgrad else 0
        if Segmenter1G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.Segmenter1G_optimizer_clipgrad, norm_type=2)
        self.Segmenter1G_optimizer.step()

        Mask_TransformerG_optimizer_clipgrad = self.args.Mask_TransformerG_optimizer_clipgrad if self.args.Mask_TransformerG_optimizer_clipgrad else 0
        if Mask_TransformerG_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.Mask_TransformerG_optimizer_clipgrad, norm_type=2)
        self.Mask_TransformerG_optimizer.step()

        # ------------------------------------
        # regularizer 
        # ------------------------------------
        Segmenter1G_regularizer_orthstep = self.args.Segmenter1G_regularizer_orthstep if self.args.Segmenter1G_regularizer_orthstep else 0
        if Segmenter1G_regularizer_orthstep > 0 and current_step_Segmenter1 % Segmenter1G_regularizer_orthstep == 0 and current_step_Segmenter1 % self.args.checkpoint_save != 0:
            self.netSegmenter1G.apply(regularizer_orth)
        G_regularizer_clipstep = self.args.Segmenter1G_regularizer_clipstep if self.args.Segmenter1G_regularizer_clipstep else 0
        if G_regularizer_clipstep > 0 and current_step_Segmenter1 % G_regularizer_clipstep == 0 and current_step_Segmenter1 % self.args.checkpoint_save != 0:
            self.netSegmenter1G.apply(regularizer_clip)

        Mask_TransformerG_regularizer_orthstep = self.args.Mask_TransformerG_regularizer_orthstep if self.args.Mask_TransformerG_regularizer_orthstep else 0
        if Mask_TransformerG_regularizer_orthstep > 0 and current_step_Mask_Transformer % Mask_TransformerG_regularizer_orthstep == 0 and current_step_Mask_Transformer % self.args.checkpoint_save != 0:
            self.Mask_TransformerG.apply(regularizer_orth)
        G_regularizer_clipstep = self.args.Mask_TransformerG_regularizer_clipstep if self.args.Mask_TransformerG_regularizer_clipstep else 0
        if G_regularizer_clipstep > 0 and current_step_Mask_Transformer % G_regularizer_clipstep == 0 and current_step_Mask_Transformer % self.args.checkpoint_save != 0:
            self.Mask_TransformerG.apply(regularizer_clip)

        self.log_dict['Segmenter1_loss'] = Segmenter1_loss.item()
        self.writer.add_scalar('Segmenter1_loss', Segmenter1_loss.item(), current_step_Segmenter1)

    def optimize_parameters_Segmenter2(self, current_step_Segmenter2, current_step_Mask_Transformer):
        self.Segmenter2G_optimizer.zero_grad()
        self.Mask_TransformerG_optimizer.zero_grad()
        self.netSegmenter2G_forward()
        Segmenter2_loss = self.SegmenterG_lossfn(self.Segmenter2E, self.img_label.type(torch.long))
        Segmenter2_loss.backward()
        
        # ------------------------------------
        # clip_grad 
        # ------------------------------------
        Segmenter2G_optimizer_clipgrad = self.args.Segmenter2G_optimizer_clipgrad if self.args.Segmenter2G_optimizer_clipgrad else 0
        if Segmenter2G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.Segmenter2G_optimizer_clipgrad, norm_type=2)
        self.Segmenter2G_optimizer.step()

        Mask_TransformerG_optimizer_clipgrad = self.args.Mask_TransformerG_optimizer_clipgrad if self.args.Mask_TransformerG_optimizer_clipgrad else 0
        if Mask_TransformerG_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.Mask_TransformerG_optimizer_clipgrad, norm_type=2)
        self.Mask_TransformerG_optimizer.step()

        # ------------------------------------
        # regularizer 
        # ------------------------------------
        Segmenter2G_regularizer_orthstep = self.args.Segmenter2G_regularizer_orthstep if self.args.Segmenter2G_regularizer_orthstep else 0
        if Segmenter2G_regularizer_orthstep > 0 and current_step_Segmenter2 % Segmenter2G_regularizer_orthstep == 0 and current_step_Segmenter2 % self.args.checkpoint_save != 0:
            self.netSegmenter2G.apply(regularizer_orth)
        G_regularizer_clipstep = self.args.Segmenter2G_regularizer_clipstep if self.args.Segmenter2G_regularizer_clipstep else 0
        if G_regularizer_clipstep > 0 and current_step_Segmenter2 % G_regularizer_clipstep == 0 and current_step_Segmenter2 % self.args.checkpoint_save != 0:
            self.netSegmenter2G.apply(regularizer_clip)
        
        Mask_TransformerG_regularizer_orthstep = self.args.Mask_TransformerG_regularizer_orthstep if self.args.Mask_TransformerG_regularizer_orthstep else 0
        if Mask_TransformerG_regularizer_orthstep > 0 and current_step_Mask_Transformer % Mask_TransformerG_regularizer_orthstep == 0 and current_step_Mask_Transformer % self.args.checkpoint_save != 0:
            self.Mask_TransformerG.apply(regularizer_orth)
        G_regularizer_clipstep = self.args.Mask_TransformerG_regularizer_clipstep if self.args.Mask_TransformerG_regularizer_clipstep else 0
        if G_regularizer_clipstep > 0 and current_step_Mask_Transformer % G_regularizer_clipstep == 0 and current_step_Mask_Transformer % self.args.checkpoint_save != 0:
            self.Mask_TransformerG.apply(regularizer_clip)

        self.log_dict['Segmenter2_loss'] = Segmenter2_loss.item()
        self.writer.add_scalar('Segmenter2_loss', Segmenter2_loss.item(), current_step_Segmenter2)


    # ----------------------------------------
    # test / inference
    # ----------------------------------------   
    def test_Fusion(self):
        self.netFusionG.eval()
        with torch.no_grad():
            self.netFusionG_forward(phase='test')
        self.netFusionG.train()

    def test_Segmenter1(self):
        self.netSegmenter1G.eval()
        with torch.no_grad():
            self.netSegmenter1G_forward(phase='test')
        self.netSegmenter1G.train()

    def test_Segmenter2(self):
        self.netSegmenter2G.eval()
        with torch.no_grad():
            self.netSegmenter2G_forward(phase='test')
        self.netSegmenter2G.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict
