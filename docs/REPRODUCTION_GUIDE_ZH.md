# VM-UNet 精确复现手册：ISIC2017、ISIC2018、Synapse

> 审阅范围：本仓库 `VM-UNet-codex-comment-global` 分支的所有 Python 源码，以及 Ruan et al., *VM-UNet: Vision Mamba UNet for Medical Image Segmentation*, arXiv:2402.02491v2 (2024-11-08), III、IV-A、IV-B 节与作者 README。**本分支没有论文 PDF**，无法核对用户所说的“较为正式.pdf”是否与 arXiv v2 完全相同。本文不修改任何代码或配置；建议把用户持有的最终 PDF 再做一次人工比对。
>
> 标记规则：**【论文】**=论文明确写出；**【代码】**=当前分支默认值；**【冲突】**=两者不一致；**【未披露】**=论文没有提供足够细节，不得声称可逐位复现。

## 1. 三套实验总表

| 超参数 | ISIC2017 | ISIC2018 | Synapse | 来源 |
|---|---|---|---|---|
| 脚本入口 | `python train.py` | `python train.py` | `python train_synapse.py` | 【代码】 |
| 配置 | `configs/config_setting.py` 选 `isic17` | 同文件默认 `isic18` | `configs/config_setting_synapse.py` | 【代码】 |
| 数据量/划分 | 1500 训练、650 测试 | 1886 训练、808 测试 | 18 CT 训练、12 CT 测试 | 【论文 IV-A】 |
| 图像分辨率 | 256×256 | 256×256 | 224×224 | 【论文 IV-B】【代码】 |
| 类别输出 | 1（二分类 sigmoid） | 1（二分类 sigmoid） | 9（背景+8 类，多类 logits） | 【代码】 |
| 输入通道 | 3（RGB） | 3（RGB） | 配置 3，读取单通道后模型内部复制为三通道 | 【代码】 |
| Epoch | 300 | 300 | 300 | 【论文 IV-B】【代码】 |
| Batch Size | 32 | 32 | 32 | 【论文 IV-B】【代码】 |
| 优化器 | AdamW | AdamW | AdamW | 【论文 IV-B】【代码】 |
| 初始学习率 | 0.001 | 0.001 | 0.001 | 【论文 IV-B】【代码】 |
| weight_decay | 0.01 | 0.01 | 0.01 | 【代码】 |
| AdamW betas / eps | (0.9,0.999) / 1e-8 | 同左 | 同左 | 【代码】 |
| 学习率调度器 | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | 【论文 IV-B】【代码】 |
| T_max | 50 | 50 | **论文 50；代码 100** | 【冲突】 |
| eta_min | 1e-5 | 1e-5 | 1e-5 | 【论文 IV-B】【代码】 |
| 随机种子 | 42 | 42 | 42 | 【代码；论文未披露】 |
| Loss | BCE+Dice，权重 1:1 | 同左 | CE+Dice，权重 1:1 | 【论文 III-D】【代码】 |
| 编码器 VSS 层深度 | [2,2,2,2] | 同左 | 同左 | 【论文 III-B】【代码】 |
| 解码器 VSS 层深度 | [2,2,2,1] | 同左 | 同左 | 【论文 III-B】【代码】 |
| drop_path_rate | 0.2 | 0.2 | 0.2 | 【代码】 |
| 预训练 | ImageNet-1k VMamba-S | 同左 | 同左 | 【论文 IV-B】【代码】 |
| AMP | False | False | False | 【代码】 |
| num_workers | 0 | 0 | 0 | 【代码】 |
| 主指标 | Dice、mIoU、Acc、Sen、Spe | 同左 | Dice、HD95（8器官） | 【论文 IV-A】【代码】 |

**注意：论文明确写“CosineAnnealingLR maximum 50 iterations”，而代码是在每个 epoch 调用一次 `scheduler.step()`。因此 ISIC 代码中 `T_max=50` 与论文一致，Synapse 代码的 `T_max=100` 不一致；不可在不改代码的前提下声称两种配置都逐项等于论文。**

## 2. ISIC2017 运行协议

**【代码】** 修改前必须确认 `configs/config_setting.py` 的 `datasets = 'isic17'`，否则默认实际运行 ISIC2018。本分支保持原代码不变，这里仅说明如何在用户本地单独实验副本设置。

```text
./data/isic2017/
  train/images/*.png
  train/masks/*.png
  val/images/*.png
  val/masks/*.png
```

- 【论文 IV-A】总 2150 张，1500/650。论文没有列出逐文件划分清单；作者 README 提供预划分资源。要核验文件名单、数量及 image/mask 对应关系。
- 【代码】模型为 VMUNet：`num_classes=1`、`input_channels=3`、`depths=[2,2,2,2]`、`depths_decoder=[2,2,2,1]`、`drop_path_rate=.2`。
- 【代码】训练增强依次为数据集归一化、Tensor、随机水平翻转 p=.5、垂直翻转 p=.5、随机角度旋转 p=.5、resize 256×256；测试/验证只归一化、Tensor、resize。精确归一化请读 `utils.myNormalize` 实现。
- 【代码】BceDiceLoss(wb=1,wd=1)；AdamW(lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=.01)；CosineAnnealingLR(T_max=50,eta_min=1e-5)；epoch=300，batch=32，seed=42。
- 【代码】`threshold=0.5`；`val_interval=30` 指每 30 epoch 报告完整分割指标，但每个 epoch 都计算验证 loss；`best.pth` 以验证 loss 最小选择。
- 【评估】`engine.py` 将验证/测试全体预测像素展平后构造单个 confusion matrix，计算全局 Dice、IoU；这与单图 Dice 后取均值的 macro 指标不同。
- 【风险】`train.py` 在末尾继续把 `val` 数据当作“Testing”；若 val 被用来选择最优 checkpoint，则不是完全独立的最终测试集。

## 3. ISIC2018 运行协议

**【代码】** `configs/config_setting.py` 默认 `datasets='isic18'`；目录为 `./data/isic2018/`，必须含 `train/images`、`train/masks`、`val/images`、`val/masks`。运行 `python train.py`。

- 【论文 IV-A】总 2694 张，1886/808；务必核实是否使用作者相同划分。仅划分比例一致不代表同一实验。
- 【代码】其余网络、超参数、增广、loss、优化器、调度器完全同上一节 ISIC2017；工作目录命名包含具体数据集和时间。
- 【评估】`engine.test_one_epoch` 输出全像素 confusion matrix 的 mIoU、DSC、Acc、Spe、Sen。
- 【验证】训练样本与掩膜是按各自 sorted 列表的相同位置配对，未保证 stem 对齐；运行前应检查样本数与名称完全一致。

## 4. Synapse 运行协议

**【论文 IV-A】** 30 个腹部 CT（共 3779 个 2D 轴向图像），18 例训练 / 12 例测试；八类器官：Aorta、Gallbladder、Left kidney、Right kidney、Liver、Pancreas、Spleen、Stomach。类别 0 是背景。

```text
./data/Synapse/
  train_npz/<slice>.npz       # 必须有 image、label
  test_vol_h5/<case>.npy.h5  # 必须有 image、label
  lists/lists_Synapse/train.txt
  lists/lists_Synapse/test_vol.txt
```

- 【冲突】README 写 `lists/list_Synapse`，代码 `configs/config_setting_synapse.py` 写 `lists/lists_Synapse`。不要混用；按实际配置核对。
- 【代码】`datasets.Synapse_dataset` 的训练 split 为 `train`，测试/验证 split 为 `test_vol`；训练对 npz 切片增强（90°旋转、翻转、随机旋转、缩放），测试沿 3D CT 逐 2D 切片推理，回拼体掩膜。
- 【代码】输入 224×224、9 类、batch 32、300 epoch、CE+Dice 各 1、AdamW lr1e-3 wd=.01、seed42、`z_spacing=1`、`val_interval=100`。
- **【冲突】当前 `T_max=100`、论文 `T_max=50`。若目标仅是跑“仓库原版”，保留 100；若专门对齐论文超参数，须在独立实验配置中改为 50 并记实验版本；当前注释分支不修改。**
- 【重要】`train_synapse.py` 的 best.pth 是按训练 loss 最小挑选，不是按 Dice 最高；不要将周期性验证 Dice 最高误认为最终 checkpoint。
- 【重要】`utils.calculate_metric_percase` 在 GT 为空、预测有前景时返回 Dice=1、HD95=0；双方都为空时返回 Dice=0、HD95=0。这属于有争议且不常规的空类别约定，会影响器官均值。
- 【代码】`test_single_volume` 对不同原始空间分辨率的切片使用三阶图像插值，对预测 mask 使用 0 阶最近邻还原；按 case 后取器官指标。
- 【未披露】CT 原始图像强度预处理、真实体素 spacing、精确病人划分文件、独立测试协议中的空掩膜定义没有在论文中完整列明；需要作者数据包或进一步联系作者。

## 5. 环境与必需文件

作者 README 给出 Python 3.8、PyTorch 1.13.0、torchvision 0.14.0、torchaudio 0.13.0、`timm==0.4.12`、`triton==2.0.0`、`causal_conv1d==1.0.0`、`mamba_ssm==1.0.1`，以及其他库（packaging, pytest, chardet, yacs, termcolor, submitit, tensorboardX, scikit-learn, matplotlib, thop, h5py, SimpleITK, scikit-image, medpy 等）。**Synapse 配置另需 `ml_collections`，但 README 安装步骤未显式安装。**

作者给出的 CUDA 安装行是 cu117，两个扩展轮子示例标注 cu118：ABI 兼容问题需按真实系统确认，并非照抄即成功。优先在 Linux+CUDA GPU 上复现原始版本。论文所用硬件为单张 NVIDIA RTX A6000。

必须另行获取数据集以及 `./pre_trained_weights/vmamba_small_e238_ema.pth`，仓库对应目录仅有 README 占位。作者 README 在“Prepare the pre_trained weights”一节另写 `./pretrained_weights/`，但代码实际是 `./pre_trained_weights/`，应以代码字段为准并记录差异。

## 6. 复现实验检查清单

- 记录 `git rev-parse HEAD`、Python/CUDA/PyTorch/Mamba 版本、GPU 型号、seed、所有超参数。
- 确认 ISIC 图片/标签文件名一一配对，Synapse 训练/测试病例严格分开。
- 核对作者预训练权重的 SHA256 和实际载入键数。
- 明确结果是“论文协议（T_max=50）”还是“源码默认（Synapse T_max=100）”。
- 明确 Dice 计算范围、空掩膜及 HD95 的 spacing 处理。
- 论文未给出所有划分与环境细节，不能承诺指标与论文绝对一致。

## 7. 参考和目标值（不是保证值）

论文 Table I：ISIC2017 mIoU 80.23%、DSC 89.03%；ISIC2018 mIoU 81.35%、DSC 89.71%。论文 Table II：Synapse DSC 81.08%、HD95 19.21。目标值仅供对照，训练差异不能直接判定为实现错误。

- 论文：https://arxiv.org/html/2402.02491 （Methods、Experiments IV-A/IV-B、Tables I–II）
- 作者 README：https://github.com/JCruan519/VM-UNet
- 本仓库源码：configs/、datasets/dataset.py、train.py、train_synapse.py、engine.py、engine_synapse.py、utils.py
