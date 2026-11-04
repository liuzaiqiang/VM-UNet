# ====== 面向医学图像分割初学者的中文源码导读（只添加注释） ======
# 说明：VMUNet 是网络最外层任务封装；内部持有 VSSM。
# 说明：当输入只有一个通道时，复制为三通道；二分类用 sigmoid，多分类返回未归一化 logits。
# 说明：load_from 将预训练 VMamba 编码器参数加载后，再把部分编码器参数按键名映射到解码器。
# 原始可执行代码和参数完全保留；复现配置及潜在错误见 docs 目录。
# ==============================================================
from .vmamba import VSSM
import torch
from torch import nn


# [学习注释] 类 VMUNet：分割任务外壳，统一处理通道数、输出激活以及预训练参数。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class VMUNet(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1,
                 depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                        )
    
    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.vmunet(x)
        if self.num_classes == 1: return torch.sigmoid(logits)
        else: return logits
    
    # [学习注释] 函数 load_from：从预训练 checkpoint 加载匹配键名的参数并对解码器做部分权重映射。 
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k: 
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k: 
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k: 
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k: 
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")