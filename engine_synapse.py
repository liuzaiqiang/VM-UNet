# ========== VM-UNet 初学者中文注释：仅增加说明，不修改执行逻辑 ==========
# 文件说明：Synapse 的训练和3D病例验证；train loss、验证 Dice、最终测试 Dice 的概念不同。
# 建议按“输入→张量形状→操作→输出→评价/损失”顺序阅读；发现的原始问题见 docs/REPRODUCTION_ISSUES_ZH.md。
# ====================================================================
import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume

import time


# [中文学习注释] 函数 train_one_epoch：训练一个 epoch：批量前向、loss、反向、优化器更新，末尾步进学习率。 
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    # [中文学习注释] 切换训练态：Dropout/BatchNorm 等层使用训练规则。
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        # [中文学习注释] 每个批次梯度清零，避免不同 batch 意外累加。
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()   

        if config.amp:
            with autocast():
                out = model(images)
                # [中文学习注释] 计算监督损失：二分类是 BCE+Dice，多类是 CE+Dice。
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            # [中文学习注释] 计算监督损失：二分类是 BCE+Dice，多类是 CE+Dice。
            loss = criterion(out, targets)
            # [中文学习注释] 从当前损失计算参数梯度；不会在此行直接改变参数。
            loss.backward()
            # [中文学习注释] 根据梯度执行一次权重更新。
            optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    # [中文学习注释] 按当前代码调用频率调整学习率；与 T_max 的单位共同决定曲线。
    scheduler.step()
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return mean_loss





# [中文学习注释] 函数 val_one_epoch：验证：切换 eval、停止梯度，计算 loss 和 Dice 等指标；注意按像素还是按病例聚合。 
def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    # [中文学习注释] 切换测试态：保证评估时使用推理行为。
    model.eval()
    # [中文学习注释] 禁用自动微分：测试时不保存反向传播计算图。
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        for i in range(1, config.num_classes):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return performance, mean_hd95