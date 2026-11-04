# ========== VM-UNet 初学者中文注释：仅增加说明，不修改执行逻辑 ==========
# 文件说明：ISIC的训练、验证、测试；预测按所有像素展平后计混淆矩阵，结果不是逐病例 Dice 均值。
# 建议按“输入→张量形状→操作→输出→评价/损失”顺序阅读；发现的原始问题见 docs/REPRODUCTION_ISSUES_ZH.md。
# ====================================================================
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


# [中文学习注释] 函数 train_one_epoch：训练一个 epoch：批量前向、loss、反向、优化器更新，末尾步进学习率。 
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    # [中文学习注释] 切换训练态：Dropout/BatchNorm 等层使用训练规则。
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        # [中文学习注释] 每个批次梯度清零，避免不同 batch 意外累加。
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        # [中文学习注释] 计算监督损失：二分类是 BCE+Dice，多类是 CE+Dice。
        loss = criterion(out, targets)

        # [中文学习注释] 从当前损失计算参数梯度；不会在此行直接改变参数。
        loss.backward()
        # [中文学习注释] 根据梯度执行一次权重更新。
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    # [中文学习注释] 按当前代码调用频率调整学习率；与 T_max 的单位共同决定曲线。
    scheduler.step() 
    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return step


# [中文学习注释] 函数 val_one_epoch：验证：切换 eval、停止梯度，计算 loss 和 Dice 等指标；注意按像素还是按病例聚合。 
def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    # [中文学习注释] 切换测试态：保证评估时使用推理行为。
    model.eval()
    preds = []
    gts = []
    loss_list = []
    # [中文学习注释] 禁用自动微分：测试时不保存反向传播计算图。
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            # [中文学习注释] 计算监督损失：二分类是 BCE+Dice，多类是 CE+Dice。
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    # [中文学习注释] 周期性报告指标：该间隔不等于训练或梯度更新间隔。
    if epoch % config.val_interval == 0:
        # [中文学习注释] 把各批次输出整理为统一数组，进一步做全像素汇总。
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # [中文学习注释] 把 sigmoid 概率用阈值转换成二值标签，此阈值影响 Dice 和 IoU。
        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        # [中文学习注释] 混淆矩阵按 TN/FP/FN/TP 统计；只有单一类别时可能不是 2×2。
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return np.mean(loss_list)


# [中文学习注释] 函数 test_one_epoch：独立测试函数：模型 eval，阈值化概率，汇总二分类混淆矩阵指标。 
def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    # [中文学习注释] 切换测试态：保证评估时使用推理行为。
    model.eval()
    preds = []
    gts = []
    loss_list = []
    # [中文学习注释] 禁用自动微分：测试时不保存反向传播计算图。
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            # [中文学习注释] 计算监督损失：二分类是 BCE+Dice，多类是 CE+Dice。
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        # [中文学习注释] 把各批次输出整理为统一数组，进一步做全像素汇总。
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # [中文学习注释] 把 sigmoid 概率用阈值转换成二值标签，此阈值影响 Dice 和 IoU。
        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        # [中文学习注释] 混淆矩阵按 TN/FP/FN/TP 统计；只有单一类别时可能不是 2×2。
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return np.mean(loss_list)