# VM-UNet 中文源码阅读路线图（初学者）

## 1. 整体数据流

```text
ISIC 数据目录 -> NPY_datasets -> train.py -> VMUNet -> VSSM -> Sigmoid 概率
                      |                   |                    |
                   图像/Mask       engine.py train/val       BCE+Dice Loss
                                                |
                                          AdamW / CosineAnnealingLR

Synapse 切片 NPZ -> Synapse_dataset -> train_synapse.py -> VMUNet -> 多类 Logits
Synapse 体数据 H5 -> Synapse_dataset -> test_single_volume -> 3D Dice、HD95
```

## 2. 推荐学习顺序（从容易到难）

1. `README.md`：了解数据来源、任务名称、文件夹组织。
2. `configs/config_setting.py` / `configs/config_setting_synapse.py`：实际运行时生效的参数及数据集选择。只选择某个 opt 和 sch，对应 elif 的其他值没有同时生效。
3. `datasets/dataset.py`：`__getitem__` 如何把原始图像/Mask 变为 Tensor，训练集怎么做同步增强，Synapse 与 ISIC 的差异。
4. `models/vmunet/vmunet.py`：`forward` 最简单；单通道复制到 3 通道，二分类 sigmoid，多类 logits。
5. `models/vmunet/vmamba.py`：从 `PatchEmbed2D` 和 `PatchMerging2D` 开始，然后读 `SS2D`、`VSSBlock`、`VSSLayer`、`VSSM` 的编码解码过程。
6. `utils.py`：软 Dice、BCE/CE 损失、数据预处理、Dice 与 HD95 的实现和空 GT 分支。
7. `train.py` / `train_synapse.py` 与 `engine.py` / `engine_synapse.py`：串联 batch、forward、loss、backward、step、保存权重以及验证。
8. `docs/REPRODUCTION_GUIDE_ZH.md` 和 `docs/REPRODUCTION_ISSUES_ZH.md`：最后复盘论文设置、仓库代码默认值与代码风险。

## 3. 核心概念词典

- **Batch [B,C,H,W]**：B 批量、C 通道、H 高、W 宽。部分 VSS 层转换为 [B,H,W,C]；不要混淆 LayerNorm 的最后一维。
- **Patch Embedding**：用 stride=4 的投影从图像获得视觉 token，H/W 通常成为 1/4。
- **Encoder**：在多尺度上学习越来越抽象的语义信息，通常逐级降低分辨率。
- **Decoder**：逐级恢复高分辨率分割预测，同时接收相应尺度的编码器特征。
- **Skip Connection**：编码器与解码器对应尺度做相加，帮助保留空间细节；本论文跳连是加法而不是复杂注意力。
- **SSM**：状态空间模型，状态更新可写为 h_t=A_bar h_(t-1)+B_bar x_t，输出 y_t=C h_t+D x_t；Mamba 通过输入相关参数让状态转移具有选择性。
- **SS2D**：二维特征按四个方向展开为序列，经过 selective scan 后合并回空间位置。
- **VSS Block**：LayerNorm + SS2D + 门控/激活 + 残差，论文 Fig.2。
- **Logits / Sigmoid / Softmax**：logits 是未归一化类别分数；sigmoid 输出二分类前景概率；softmax 将多类 logits 转为互斥类别概率。
- **BCE / CE**：逐像素分类损失；BCE 适合二分类，CE 适合互斥多分类。
- **Soft Dice**：可微的区域重叠近似；与硬阈值评价 Dice 不同。
- **HD95**：95% Hausdorff 距离，是边界误差的稳健度量；物理 spacing 和空掩膜约定极为重要。
- **Epoch / Iteration**：Epoch 遍历所有训练样本；Iteration 更新一个 batch；学习率调度器 `step` 的调用频率决定 T_max 的解释。
- **Checkpoint**：模型参数快照；最佳 checkpoint 和最新 checkpoint 有不同用途。

## 4. 读 SS2D 时建议的张量表

| 阶段 | 概念上观察什么 | 重点核对 |
|---|---|---|
| PatchEmbed | [B,3,H,W] -> [B,H/4,W/4,C] | Conv 步长、是否使用 LayerNorm |
| PatchMerging | H/W 下降，C 上升 | 拼接顺序、padding、线性层 |
| SS2D 扫描输入 | [B,H,W,C] -> [B,D,H,W] -> 多方向序列 | permute 和 view 后的元素位置 |
| Selective Scan | [B,方向×通道,长度] 的状态迭代 | dt、A、B、C、D 的含义及维数 |
| Scan Merge | 四向特征重新恢复网格 | 翻转、transpose 和求和 |
| Decoder | 深层低分辨率 -> 浅层高分辨率 | skip 是相加，空间尺寸要匹配 |
| Final Head | 最终特征 -> [B,num_classes,H,W] | 二分类 sigmoid 与多类 logits |

具体张量精确形状以 `vmamba.py` 当前模块参数、运行时配置为准。不能单凭表格替代真实代码检查。

## 5. 自己检查理解程度的五个问题

1. ISIC2018 的 `batch_size=32` 会使一次前向的输入 Tensor 是什么形状？
2. Synapse 原图是单通道时，VMUNet.forward 为什么要扩展到 3 通道？
3. VM-UNet 中 SS2D 为什么需要四个扫描方向？合并后如何回到原空间？
4. `train_synapse.py` 的最佳模型到底按训练 loss 还是验证 Dice 选？
5. `calculate_metric_percase` 为什么可能对纯假阳性给 Dice=1？这会如何污染论文报告的均值？

以上每个问题都能由源码中的中文注释和两份审计文档定位。
