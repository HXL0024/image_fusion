import torch
import os
from torch.nn.parallel import DataParallel
# 模型加载训练的基础类，其他模型继承自这个类，共享一些通用的功能和方法
# 训练、保存、加载模型；定义损失、优化器和学习率调度器；
# 提供训练数据，优化模型参数、获取当前的可视化数据，获取当前损失值，更新学习率，设置是否需要梯度
# 网络结构和参数的信息获取
# 模型的保存和加载
class ModelBase():
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.is_train = args.is_train
        self.schedulers_Fusion = []
        self.schedulers_Segmenter1 = []
        self.schedulers_Segmenter2 = []
        self.schedulers_Mask_Transformer = []

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def define_loss(self):
        pass

    def define_scheduler(self):
        pass

    def feed_data(self):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate_Fusion(self, n):
        for scheduler in self.schedulers_Fusion:
            scheduler.step(n)

    def update_learning_rate_Segmenter1(self, n):
        for scheduler in self.schedulers_Segmenter1:
            scheduler.step(n)

    def update_learning_rate_Segmenter2(self, n):
        for scheduler in self.schedulers_Segmenter2:
            scheduler.step(n)

    def update_learning_rate_Mask_Transformer(self, n):
        for scheduler in self.schedulers_Mask_Transformer:
            scheduler.step(n)
        
    def current_learning_rate_Fusion(self):
        return self.schedulers_Fusion[0].get_lr()[0]
    
    def current_learning_rate_Segmenter1(self):
        return self.schedulers_Segmenter1[0].get_lr()[0]
    
    def current_learning_rate_Segmenter2(self):
        return self.schedulers_Segmenter2[0].get_lr()[0]
    
    def current_learning_rate_Mask_Transformer(self):
        return self.schedulers_Mask_Transformer[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, DataParallel):
            network = network.module
        return network

    def model_to_device(self, network):
        network = network.to(self.device)
        # 使用DataParallel包装模型，使其可以在多个GPU上运行
        # network = DataParallel(network)
        return network

    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # 加载预训练的神经网络模型的权重
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        # 如果strict为true，直接加载状态字典，否则适用于加载先前版本的模型。
        # 将加载的状态字典应用于网络模型
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))


    # 更新编码器的参数，使编码器的参数逐渐趋向于生成器，用来实现参数的平滑迁移和模型的知识蒸馏
    def update_FusionE(self, decay=0.999):
        netFusionG = self.get_bare_model(self.netFusionG)
        netFusionG_params = dict(netFusionG.named_parameters())
        netFusionE_params = dict(self.netFusionE.named_parameters())
        for k in netFusionG_params.keys():
            netFusionE_params[k].data.mul_(decay).add_(netFusionG_params[k].data, alpha=1-decay)
    
    def update_Segmenter1E(self, decay=0.999):
        netSegmenter1G = self.get_bare_model(self.netSegmenter1G)
        netSegmenter1G_params = dict(netSegmenter1G.named_parameters())
        netSegmenter1E_params = dict(self.netSegmenter1E.named_parameters())
        for k in netSegmenter1G_params.keys():
            netSegmenter1E_params[k].data.mul_(decay).add_(netSegmenter1G_params[k].data, alpha=1-decay)

    def update_Segmenter2E(self, decay=0.999):
        netSegmenter2G = self.get_bare_model(self.netSegmenter2G)
        netSegmenter2G_params = dict(netSegmenter2G.named_parameters())
        netSegmenter2E_params = dict(self.netSegmenter2E.named_parameters())
        for k in netSegmenter2G_params.keys():
            netSegmenter2E_params[k].data.mul_(decay).add_(netSegmenter2G_params[k].data, alpha=1-decay)

    def update_Mask_TransformerE(self, decay=0.999):
        Mask_TransformerG = self.get_bare_model(self.Mask_TransformerG)
        Mask_TransformerG_params = dict(Mask_TransformerG.named_parameters())
        Mask_TransformerE_params = dict(self.Mask_TransformerE.named_parameters())
        for k in Mask_TransformerG_params.keys():
            Mask_TransformerE_params[k].data.mul_(decay).add_(Mask_TransformerG_params[k].data, alpha=1-decay)
    # # ----------------------------------------
    # # merge bn during training
    # # ----------------------------------------
    # def merge_bnorm_train(self):
    #     merge_bn(self.netG)
    #     tidy_sequential(self.netG)
    #     self.define_optimizer()
    #     self.define_scheduler()

    # # ----------------------------------------
    # # merge bn before testing
    # # ----------------------------------------
    # def merge_bnorm_test(self):
    #     merge_bn(self.netG)
    #     tidy_sequential(self.netG)