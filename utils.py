# ====== 面向医学图像分割初学者的中文源码导读（只添加注释） ======
# 说明：工具集：随机性、日志、优化器和调度器、多种损失、预处理、模型统计、分割指标。
# 说明：Synapse 评估函数在 GT 为空而预测不为空时 Dice=1，是非常规纯假阳性处理，已单独记入审计。
# 说明：ISIC 使用二分类 BCE+Dice，Synapse 使用多类 CE+Dice；loss 的输入和标签格式不同。
# 原始可执行代码和参数完全保留；复现配置及潜在错误见 docs 目录。
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric


# [学习注释] 函数 set_seed：设置随机种子与 cuDNN 确定性标志，有助于但不能保证跨 GPU 位级一致。 
def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


# [学习注释] 函数 get_logger：定义该模块的局部计算或辅助操作；请结合输入、调用方及输出理解其作用。 
def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
    return logger


# [学习注释] 函数 log_config_info：定义该模块的局部计算或辅助操作；请结合输入、调用方及输出理解其作用。 
def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



# [学习注释] 函数 get_optimizer：根据 opt 选出优化器；只使用对应 if/elif 分支中的参数。 
def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


# [学习注释] 函数 get_scheduler：根据 sch 选出学习率调度器；T_max 单位是 scheduler.step 调用次数。 
def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
    return scheduler



# [学习注释] 函数 save_imgs：保存预测可视化；概率图与二值 mask 的可视化方法应区分。 
def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # 处理图像数据
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    
    # 根据数据集处理掩码
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        att_msk = np.squeeze(msk_pred, axis=0)
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    # 设置画布大小
    plt.figure(figsize=(10, 20))

    # 调整子图间距，减少白色边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)

    # 原图
    plt.subplot(4, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')

    # Ground Truth 掩码
    plt.subplot(4, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth Mask')

    # 预测掩码
    plt.subplot(4, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')
    plt.title('Predicted Mask')

    # 注意力图谱叠加到原图上
    plt.subplot(4, 1, 4)
    plt.imshow(img)
    plt.imshow(att_msk, cmap='jet', alpha=0.5)  # 热力图叠加到原图上，alpha设置透明度
    plt.axis('off')
    plt.title('Attention Map Overlay')

    # 保存图片
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    


# [学习注释] 类 BCELoss：二分类交叉熵损失：比较预测概率和二值分割掩膜。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class BCELoss(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return self.bceloss(pred_, target_)


# [学习注释] 类 DiceLoss：软 Dice 损失：通过预测与 GT 的重叠提高前景区域分割质量。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class DiceLoss(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self):
        super(DiceLoss, self).__init__()

    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return dice_loss
    

# [学习注释] 类 nDiceLoss：多类别 Dice：对整数标签 one-hot 编码，并逐类别衡量重叠。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class nDiceLoss(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    # [学习注释] 函数 _one_hot_encoder：定义该模块的局部计算或辅助操作；请结合输入、调用方及输出理解其作用。 
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return output_tensor.float()

    # [学习注释] 函数 _dice_loss：定义该模块的局部计算或辅助操作；请结合输入、调用方及输出理解其作用。 
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return loss

    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return loss / self.n_classes


# [学习注释] 类 CeDiceLoss：多类 CrossEntropy + Dice 的加权和；本项目 Synapse 权重均为 1。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class CeDiceLoss(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return loss


# [学习注释] 类 BceDiceLoss：二分类 BCE + Dice 的加权和；ISIC 权重均为 1。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class BceDiceLoss(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return loss
    

# [学习注释] 类 GT_BceDiceLoss：带辅助预测的深监督损失类；是否实际启用需看训练入口。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class GT_BceDiceLoss(nn.Module):
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    # [学习注释] 函数 forward：前向传播：接受输入张量并输出特征或预测；建议逐行在纸上跟踪 B、C、H、W 变化。 
    # [学习注释] 形状提示：常见格式是 [B,C,H,W] 或 [B,H,W,C]，遇到 permute / view / reshape 要逐项核实。
    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return bcediceloss + gt_loss



# [学习注释] 类 myToTensor：HWC 图像转为 CHW 的 Tensor，以适配 PyTorch 图像批次格式。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class myToTensor:
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self):
        pass
    # [学习注释] 函数 __call__：将当前变换对象应用于数据；检查图像与分割标签同步变化。 
    def __call__(self, data):
        image, mask = data
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

# [学习注释] 类 myResize：对图像及标签执行尺寸缩放；必须确认掩膜插值和边界处理。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class myResize:
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    # [学习注释] 函数 __call__：将当前变换对象应用于数据；检查图像与分割标签同步变化。 
    def __call__(self, data):
        image, mask = data
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

# [学习注释] 类 myRandomHorizontalFlip：对图像和掩膜同时随机水平翻转，确保标签空间对应。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class myRandomHorizontalFlip:
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, p=0.5):
        self.p = p
    # [学习注释] 函数 __call__：将当前变换对象应用于数据；检查图像与分割标签同步变化。 
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

# [学习注释] 类 myRandomVerticalFlip：对图像和掩膜同时随机竖直翻转。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class myRandomVerticalFlip:
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, p=0.5):
        self.p = p
    # [学习注释] 函数 __call__：将当前变换对象应用于数据；检查图像与分割标签同步变化。 
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


# [学习注释] 类 myRandomRotation：对图像和掩膜同时随机旋转；旋转范围和插值方式影响结果。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class myRandomRotation:
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    # [学习注释] 函数 __call__：将当前变换对象应用于数据；检查图像与分割标签同步变化。 
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


# [学习注释] 类 myNormalize：按指定数据集进行图像归一化处理；训练和测试配置可能不同。 
# [学习注释] 检查要点：输入维度、可学习参数、是否下采样/上采样、残差和输出维度。
class myNormalize:
    # [学习注释] 函数 __init__：构造模块并注册层/参数；实例化不等于执行 forward。 
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
            
    # [学习注释] 函数 __call__：将当前变换对象应用于数据；检查图像与分割标签同步变化。 
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return img_normalized, msk
    


from thop import profile		 ## 导入thop模块
# [学习注释] 函数 cal_params_flops：定义该模块的局部计算或辅助操作；请结合输入、调用方及输出理解其作用。 
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')






# [学习注释] 函数 calculate_metric_percase：Synapse 每个器官病例的 Dice、HD95；尤其注意空掩膜分支。 
def calculate_metric_percase(pred, gt):
    # [学习注释] 空 GT / 纯假阳性情况的原始处理会影响 Dice 统计；审计文档有详细说明。
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # [学习注释] 空 GT / 纯假阳性情况的原始处理会影响 Dice 统计；审计文档有详细说明。
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return 1, 0
    else:
        # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
        return 0, 0



# [学习注释] 函数 test_single_volume：对单个 3D CT 病例按 2D 切片推理并回拼体数据；逐类别计算 Dice/HD95。 
def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            # [学习注释] 验证和测试时禁用梯度，以减少显存和计算。
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        # [学习注释] 验证和测试时禁用梯度，以减少显存和计算。
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+ case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    # [学习注释] 返回值可能为 logits、概率、特征、loss 或指标；结合上游调用核实。
    return metric_list