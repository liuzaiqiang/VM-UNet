# ========== VM-UNet 初学者中文注释：仅增加说明，不修改执行逻辑 ==========
# 文件说明：ISIC：RGB图像+灰度mask；Synapse：训练读取2D npz，验证读取3D h5。注意检查文件命名、切片与体数据。
# 建议按“输入→张量形状→操作→输出→评价/损失”顺序阅读；发现的原始问题见 docs/REPRODUCTION_ISSUES_ZH.md。
# ====================================================================
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


# [中文学习注释] 类 NPY_datasets：ISIC 读取器；目录中图像与 mask 通过排序位置配对，应核对文件 stem 一致性。 
class NPY_datasets(Dataset):
    # [中文学习注释] 函数 __init__：构造函数：记录输入参数并初始化成员变量，具体数据流发生在后续调用。 
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    # [中文学习注释] 函数 __getitem__：根据索引读取单个样本；确认图像形状、标签类型与返回字典字段。 
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
        return img, msk

    # [中文学习注释] 函数 __len__：返回数据集大小；DataLoader 决定每个 epoch 的迭代次数。 
    def __len__(self):
        # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
        return len(self.data)
    


# [中文学习注释] 函数 random_rot_flip：对图像与标签同步做 90° 旋转和翻转，保持像素一一对应。 
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return image, label


# [中文学习注释] 函数 random_rotate：对输入与标签同步做随机旋转；需要检查插值阶数与标签离散性。 
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
    return image, label


# [中文学习注释] 类 RandomGenerator：同步对原图和标签做增强、缩放、Tensor 转换；标签使用最近邻防止出现非法类别。 
class RandomGenerator(object):
    # [中文学习注释] 函数 __init__：构造函数：记录输入参数并初始化成员变量，具体数据流发生在后续调用。 
    def __init__(self, output_size):
        self.output_size = output_size

    # [中文学习注释] 函数 __call__：作为数据处理变换被调用；注意是否同步变换图像和掩膜。 
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
        return sample


# [中文学习注释] 类 Synapse_dataset：Synapse 数据集；train 用切片 npz，test_vol 用体数据 h5。 
class Synapse_dataset(Dataset):
    # [中文学习注释] 函数 __init__：构造函数：记录输入参数并初始化成员变量，具体数据流发生在后续调用。 
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    # [中文学习注释] 函数 __len__：返回数据集大小；DataLoader 决定每个 epoch 的迭代次数。 
    def __len__(self):
        # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
        return len(self.sample_list)

    # [中文学习注释] 函数 __getitem__：根据索引读取单个样本；确认图像形状、标签类型与返回字典字段。 
    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            # [中文学习注释] 实验配置：必须对照当前选中的数据集与分支判断是否生效。
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # [中文学习注释] 返回值交给调用方；请区分 loss、预测张量和评价指标。
        return sample
        
    