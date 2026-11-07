# VM-UNet 复现前必读：源码审计、论文对照与实验执行手册

> **适用仓库**：https://github.com/liuzaiqiang/VM-UNet  
> **适用分支**：`VM-UNet-codex-comment-global`  
> **审计基准提交**：`7f13b53477e667daf978592ec7348b75d841c922`（本文件创建前）  
> **原作者仓库**：https://github.com/JCruan519/VM-UNet  
> **原作者基准提交**：`b87827eb5a5faff00bd4b7b6505e56d3ca813ca2`（2025-09-03）  
> **论文**：Jiacheng Ruan 等，*VM-UNet: Vision Mamba UNet for Medical Image Segmentation*，https://arxiv.org/abs/2402.02491；另见正式出版版本，核对具体版本后再逐表对照。  
> **审计方式**：逐文件静态代码审阅、当前分支对原作者主分支的提交关系检查、对照原作者 README 和公开论文实验描述。**未获得本地数据、真实预训练权重和 GPU 实测日志，故不宣称已跑通，也不宣称已复现论文数值。**  
> **修改范围**：只新增本文档；不修改模型、数据处理、损失、训练、评估或配置代码。  
> **阅读原则**：【已确认】=可直接从当前源码或论文核对；【风险】=可能影响成功率、数值或公平性的实现；【建议】=实验操作建议，不等于作者声明；【待验证】=需要真实环境、文件、论文版本或实验日志才能确认。

---

## 0. 先看结论：不要直接开始 300 Epoch

**代码来源已确认**：用户仓库的 `main` 与原作者 `JCruan519/VM-UNet` 主分支处于同一个提交 `b87827e...`；当前 `VM-UNet-codex-comment-global` 仅比该提交多一笔中文注释/文档提交。对比记录显示，原 Python 源码仅新增注释行、无可执行代码删改。因此当前分支可以作为**作者公开实现**的基准，但“作者公开实现”不等于“与论文每一处描述完全一致”。

**最重要的前置风险**：

| 编号 | 等级 | 结论 | 定位/操作 |
|---|---|---|---|
| P0-1 | 严重 | Synapse：GT 为空而预测非空时 Dice 被写成 `1`、HD95 为 `0` | `utils.py:580-595`；务必明确复现采用的评价规则 |
| P0-2 | 高 | Synapse 源码 `T_max=100`，论文实验描述 `T_max=50` | `configs/config_setting_synapse.py`；分开记录「源码默认」和「论文对齐」 |
| P0-3 | 高 | ISIC 旋转角 `self.angle` 只在变换对象构造时生成一次，且构造发生在 `set_seed(42)` 之前 | `utils.py:512-525`，`configs/config_setting.py`，`train.py:50-53` |
| P0-4 | 高 | ISIC 图像与 Mask 共用 `TF.resize` 默认插值；Mask 可能出现 `0~1` 中间值 | `utils.py:472-481` |
| P0-5 | 高 | Synapse `medpy.metric.binary.hd95` 未传入 `voxelspacing`；`z_spacing=1` 仅用于写 NIfTI | `utils.py:580-646`；不可无条件称实际毫米 HD95 |
| P0-6 | 高 | 环境、数据、预训练权重不完整：仓库不含三个数据集和 `vmamba_small_e238_ema.pth` | `README.md`、`data/`、`pre_trained_weights/` |
| P0-7 | 高 | README 中多个数据/预训练目录名和代码不一致 | 优先按照 `configs/` 的实际路径 |
| P1-1 | 高 | Synapse 仅测试模式直接引用未定义的 `epoch` | `train_synapse.py:140-158`；原版不能直接使用该开关 |
| P1-2 | 中 | Synapse 最佳模型按**训练 Loss** 选，非按 Dice | `train_synapse.py:185-203` |
| P1-3 | 中 | ISIC 采用 val Loss 选最佳权重，却再用同一个 val 做 `Testing` | `train.py:170-212`；不是独立测试 |
| P1-4 | 中 | `vmamba.py` 对 selective-scan 导入失败使用静默 `except: pass` | `models/vmunet/vmamba.py:19-29`；安装失败可延迟到前向才暴露 |
| P1-5 | 中 | ISIC 的 `cal_params_flops` 在真正训练前运行一次 GPU 前向 | `train.py:92-95`；可能在 FLOPs 统计阶段先报错 |
| P1-6 | 中 | `load_from` 按参数名过滤、未按形状过滤，权重不匹配时可能失败 | `models/vmunet/vmunet.py` |
| P1-7 | 中 | ISIC 训练 `step += iter`，TensorBoard step 不是真正全局优化步 | `engine.py:33-35`；主要影响日志而非反向传播 |
| P1-8 | 中 | README 的 PyTorch cu117 与扩展 wheel 示例 cu118 需核对 ABI/驱动/GPU | 不要盲目按最新版本替换 |
| P2-1 | 中 | 对静态全灰度/全颜色 ISIC 图，`myNormalize` 的分母可能为零 | `utils.py:526-558`；数据异常时出现 NaN |
| P2-2 | 中 | ISIC `confusion_matrix` 不固定 `labels=[0,1]`，整套预测/标注单类别时索引可能失败 | `engine.py:106-107,171-172` |
| P2-3 | 中 | AdamW 统一对 `model.parameters()` 设置衰减，没有读取部分参数的 `_no_weight_decay` 标记 | `utils.py:118-127`；原实现细节，不要静默改动 |
| P2-4 | 中 | 默认只使用单 GPU；`distributed=False`，不能假设多卡自动并行 | 两套 `configs/` 和两套 `train` |
| P2-5 | 中 | `myNormalize` 在标准化后再逐图 min-max 到 0~255，不能当作普通 0~1 归一化 | `utils.py:552-558` |

**启动准入要求**：完成 `第 2~5 节` 的检查，再做 `第 8 节` 的冒烟测试，最后才考虑长时间训练。

---

## 1. 源码结构与真正的数据流

| 文件 | 角色 | 需要读懂什么 |
|---|---|---|
| `README.md` | 作者公开使用说明 | 论文、预训练下载、三数据集路径；但不是唯一真相 |
| `configs/config_setting.py` | ISIC 配置 | 默认 `isic18`、256、batch 32、300 epoch、变换 |
| `configs/config_setting_synapse.py` | Synapse 配置 | 224、9 类、train_npz/test_vol_h5、T_max=100 |
| `datasets/dataset.py` | 数据加载 | ISIC RGB/mask，Synapse 的 NPZ 切片和 H5 体数据 |
| `models/vmunet/vmunet.py` | 顶层网络接口 | 单通道复制三通道、二分类 sigmoid、多类 logits、预训练映射 |
| `models/vmunet/vmamba.py` | 网络主体 | PatchEmbed、PatchMerging、SS2D、VSS Block、加法 skip、上采样 |
| `utils.py` | Loss、变换、优化器、HD95、文件保存 | 重点风险集中处 |
| `train.py` + `engine.py` | ISIC 训练/验证/测试 | val Loss 选模；全体像素统计 |
| `train_synapse.py` + `engine_synapse.py` | Synapse 训练/评估 | train Loss 选模；按 3D 病例评估 |
| `docs/CODE_READING_GUIDE_ZH.md` | 已有中文代码导读 | 建议和本文一起使用 |
| `docs/REPRODUCTION_GUIDE_ZH.md` | 已有论文配置对照 | 核对参数和指标 |
| `docs/REPRODUCTION_ISSUES_ZH.md` | 已有缺陷清单 | 与本文相互交叉确认 |

### 1.1 两种任务不要混淆

`ISIC17/ISIC18` 是 **RGB 皮肤病灶二分类**：

`PNG 图像与 Mask -> NPY_datasets -> 256×256 -> VMUNet(1类) -> sigmoid 概率 -> BCE + soft Dice -> 阈值0.5 -> 全像素 Dice/IoU等指标`

`Synapse` 是 **2D 切片训练、3D 体数据测试的八器官互斥多分类**：

`train_npz 的 2D CT -> RandomGenerator(224×224) -> VMUNet(9类，含背景) -> logits -> CE + soft Dice`  
`test_vol_h5 的 3D CT -> 逐切片 resize/forward/argmax -> 拼回3D -> 按病例、按八器官求 Dice/HD95 -> 平均`

**关键理解**：两者不是把同一个训练脚本中的数据路径替换一下就能互换。输入维度、输出激活、损失函数、评估单位均不同。

### 1.2 网络结构的张量追踪

配置使用 `depths=[2,2,2,2]`、`depths_decoder=[2,2,2,1]`，通道依次为 `96,192,384,768`，跳连为**逐元素相加**而非拼接。理论形状：

| 阶段 | ISIC（256×256） | Synapse（224×224） |
|---|---|---|
| 输入，三通道 | `[B,3,256,256]` | `[B,3,224,224]`（灰度先复制） |
| PatchEmbed stride 4 | `[B,64,64,96]` | `[B,56,56,96]` |
| Encoder stage 1 | `[B,32,32,192]` | `[B,28,28,192]` |
| Encoder stage 2 | `[B,16,16,384]` | `[B,14,14,384]` |
| Encoder stage 3 | `[B,8,8,768]` | `[B,7,7,768]` |
| Decoder | `8→16→32→64` | `7→14→28→56` |
| Final_PatchExpand×4 | 恢复 `256×256` | 恢复 `224×224` |
| 最终分割输出 | `[B,1,256,256]` | `[B,9,224,224]` |

这些是依源码推导的默认形状，实际计算时还要考虑内部 `BHWC` / `BCHW` 转置和每层编码器跳连的采样时点。`SS2D` 将特征拉直成多个方向的长度 `H×W` 序列，在 `mamba_ssm` 的 selective scan 上运行，之后复原空间布局并融合四个方向。**这部分需要 GPU 原生算子支持，不能仅凭模型能实例化认定可训练。**

---

## 2. 环境：先验证可用，再安装整套依赖

### 2.1 推荐实验策略

【已确认】作者 README 给出的环境**历史版本**：Python 3.8；PyTorch 1.13.0；torchvision 0.14.0；torchaudio 0.13.0；timm 0.4.12；triton 2.0.0；causal_conv1d 1.0.0；mamba_ssm 1.0.1；还依赖 medpy、SimpleITK、h5py、scipy、einops、sklearn、PIL、matplotlib、thop、tensorboardX 等。`configs/config_setting_synapse.py` 还导入 `ml_collections`，作者 README 未显式列出它。

【建议】优先尝试 **Linux + NVIDIA CUDA GPU + 独立 Conda 环境**，并记录 GPU 型号和 NVIDIA 驱动。新显卡、较新的 Linux、不同 GCC/CUDA ABI 下，旧 wheel 可能不能直接用；**不要简单地把所有依赖升级到最新版，然后把结果叫作严格原版复现。**

先执行：

```bash
git clone https://github.com/liuzaiqiang/VM-UNet.git
cd VM-UNet
git checkout VM-UNet-codex-comment-global
git rev-parse HEAD
nvidia-smi
nvcc --version                  # 若系统未提供 nvcc，不等于没有 NVIDIA 驱动
python --version
```

再按仓库 `README.md` 原版命令建隔离环境，并特别补充检查：

```bash
python -m pip check
python - <<'PY'
import sys, torch, torchvision, timm
print("python:", sys.version)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
python - <<'PY'
import ml_collections, einops, h5py, scipy, medpy, SimpleITK, thop, tensorboardX
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("basic dependency and selective_scan imports: OK")
PY
```

【风险】作者 README 中安装 PyTorch 的示例为 `cu117`，而两种自定义 CUDA 扩展示例 wheel 文件名含 `cu118`；还必须匹配 Python 3.8、torch 1.13、Linux、C++ ABI 等。**示例 wheel 名称不是跨机器通用安装保证。**遇到 `undefined symbol`、`no kernel image`、`invalid device function`、`CUDA_HOME` 或 `ninja` 构建失败时，首先检查版本矩阵、扩展 wheel 来源及 GPU 架构，暂勿改动模型。

【风险】`vmamba.py` 用宽泛 `try/except` 忽略 `selective_scan_fn` 的导入异常，但实际 `SS2D.forward_corev0` 仍直接使用它；因此可能**导入网络成功、第一次 forward 才 NameError**。必须进行后面的真实前向冒烟测试。

【重要】`torch.load` 可反序列化对象：只加载来自可信来源的权重；对论文原版所需旧版 PyTorch，文件来源可信尤为重要。禁止把未经核实的第三方权重当作作者权重。

### 2.2 GPU、显存与资源分配

原论文实验描述使用单张 NVIDIA RTX A6000；但不同型号、显存和 CUDA kernel 支持可能导致不同吞吐与数值结果。首次运行推荐只选一张卡，**先 batch=1 做模型前向**，再完整按作者参数跑 batch=32。调小 batch 可以验证能否跑通，却不是与 batch=32 相同的复现实验；不要混淆。

- ISIC：`configs/config_setting.py` 有 `gpu_id='0'`，`train.py` 调用 `model.cuda()`；并非 CPU 脚本。
- Synapse：`train_synapse.py` 顶层写死 `CUDA_VISIBLE_DEVICES="0"`，`gpu_ids=[0]`，默认 `distributed=False` 且用 `DataParallel` 包装；不能假设双卡机器会自动用两张卡。
- CPU 试跑不是作者原方案；很多 selective scan 扩展依赖 CUDA，CPU 环境下可能无法完成 forward。
- `num_workers=0` 是源码默认值。别为追求吞吐提前修改工作进程数；先观察基线性能和随机性。
- `amp=False` 为默认值。调为 True 会改变计算精度、显存和数值轨迹，应单独命名实验。

---

## 3. 数据集准备：目录正确只是第一步

### 3.1 ISIC 目录（按**代码**，不是 README 的简写）

```text
VM-UNet/
└── data/
    ├── isic2017/
    │   ├── train/images/*.png
    │   ├── train/masks/*.png
    │   ├── val/images/*.png
    │   └── val/masks/*.png
    └── isic2018/
        ├── train/images/*.png
        ├── train/masks/*.png
        ├── val/images/*.png
        └── val/masks/*.png
```

**目录命名冲突**：【已确认】作者 README 写 `isic17` / `isic18`，源码 `configs/config_setting.py` 用 `isic2017` / `isic2018`。先按代码落盘，不要同时复制多份而不记录。

**数据划分**：【论文】ISIC2017 共 2150 幅，训练 1500、测试 650；ISIC2018 共 2694 幅，训练 1886、测试 808。论文/README 给出预划分资源。**随机按 7:3 重新抽样不等于论文相同划分**。代码使用 `val` 文件夹作验证和最终“测试”，需在报告中如实写明。

**样本配对**：【已确认】`NPY_datasets` 分别排序 images 和 masks，再按索引配对；如果文件名数量相等但名称错位，程序可能正常训练却用错标签。先做名称和数量核对：

```bash
python - <<'PY'
from pathlib import Path
for ds in ("isic2017", "isic2018"):
    for split in ("train", "val"):
        root=Path("data")/ds/split
        imgs=sorted((root/"images").glob("*.png"))
        masks=sorted((root/"masks").glob("*.png"))
        print(ds, split, "images",len(imgs),"masks",len(masks))
        assert imgs and len(imgs)==len(masks), "数量不匹配或没有PNG"
        # 如果作者原包的 image/mask 文件名规则不同，需用映射表核对，而非强制同名。
        mismatches=[(i.name,m.name) for i,m in zip(imgs,masks) if i.stem != m.stem]
        print("different stems example:",mismatches[:5])
PY
```

**不要误判文件名**：作者数据包可能在 mask 名称中增加后缀；有差异应查看一一对应关系与可视化，而不是不加验证地重命名。

**标签和值域**：【已确认】`dataset.py` 把 RGB 读成三通道；Mask 灰度读取后 `/255`。确认原始掩膜是否只含 0/255、是否有抗锯齿灰度、是否与图片尺寸一致。灰度边缘可能导致 soft label；这属于数据协议的一部分。

### 3.2 ISIC 的两个容易忽略的实现细节

**A. 旋转随机性并非你想象的“每张图随机一个角度”。**

`myRandomRotation.__init__` 中执行 `self.angle=random.uniform(0,360)`，随后每张图 `__call__` 只随机决定**是否**按这个固定角度旋转。由于 `config_setting.py` 在模块导入期间就创建了该变换，而 `train.py` 直到 `main` 内才 `set_seed(42)`，不同启动进程的固定角度可能不同。**因此配置里 seed=42 不保证这部分数据增强在不同运行之间一致。** 原版基线先记录现象；若为规范性研究修复，应在新分支将角度抽样移到 `__call__` 并控制初始化时序，但此举改变原版训练分布。

**B. Mask resize 不是自动“最近邻”。**

`myResize` 把 Image 和 Mask 都交给 `TF.resize`，没有指定插值方式。此处经 `myToTensor` 得到 tensor，torchvision 常用版本中默认是双线性插值；Mask 本应为离散类别标签，却可能出现 0 与 1 之间的像素。训练 BCE/Dice 接收该连续 Mask，而测试又以 `>=0.5` 二值化 GT。**原版结果与“最近邻 Mask resize”修复版本不能混为同一个基线。**

**C. 图像归一化的实际含义。**

`myNormalize` 首先按配置的 mean/std 标准化，紧接着对当前图像 min-max 映射到 `0~255`。由于第二步会抵消正数 std 和 mean 的线性缩放，正常非恒定图上实际效果近似“每张图独立映射到 0~255”，而**不是**常见的 0~1 或 ImageNet mean/std 归一化。若整图像素恒定，`max-min=0` 会引发 NaN。别随意换为 `transforms.Normalize` 后称仍复现原方法。

### 3.3 Synapse 目录（以代码为准）

```text
VM-UNet/
└── data/Synapse/
    ├── train_npz/
    │   └── caseXXXX_sliceXXX.npz   # 实际名称按 train.txt
    ├── test_vol_h5/
    │   └── caseXXXX.npy.h5        # 实际名称按 test_vol.txt
    └── lists/
        └── lists_Synapse/
            ├── train.txt
            └── test_vol.txt
```

【已确认】`config_setting_synapse.py` 采用 `lists/lists_Synapse`；README 写 `lists/list_Synapse`（少一个 s）。`Synapse_dataset` 不看 `all.lst`，运行所需是 `train.txt` 和 `test_vol.txt`；训练按列表逐行读取 `名称.npz`，体测试按列表读取 `名称.npy.h5`。

【论文】通常按 18 个 CT 病例训练、12 个病例测试；**应按病例级划分，不是先把 3D CT 切成 2D 后逐切片随机划分**。后一种会使同一患者不同切片跨训练/测试，构成泄漏风险。

可做下列抽样检查：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np, h5py
root=Path("data/Synapse")
lst=root/"lists/lists_Synapse"
train=[x.strip() for x in (lst/"train.txt").read_text().splitlines() if x.strip()]
test=[x.strip() for x in (lst/"test_vol.txt").read_text().splitlines() if x.strip()]
print("train slices:",len(train),"test volumes:",len(test))
print("train first:",train[:3],"test first:",test[:3])
assert train and test, "列表为空"
missing_train=[x for x in train if not (root/"train_npz"/(x+".npz")).is_file()]
missing_test=[x for x in test if not (root/"test_vol_h5"/(x+".npy.h5")).is_file()]
print("missing train:",missing_train[:10],"missing test:",missing_test[:10])
assert not missing_train and not missing_test, "列表与文件不一致"
with np.load(root/"train_npz"/(train[0]+".npz")) as d:
    im, lab=d["image"], d["label"]
    print("train shape/dtype:",im.shape,im.dtype,lab.shape,lab.dtype)
    print("train labels:",np.unique(lab))
    assert im.shape==lab.shape and set(np.unique(lab)).issubset(set(range(9)))
with h5py.File(root/"test_vol_h5"/(test[0]+".npy.h5"),"r") as d:
    im, lab=d["image"][:], d["label"][:]
    print("volume shape/dtype:",im.shape,im.dtype,lab.shape,lab.dtype)
    print("test labels:",np.unique(lab))
    assert im.shape==lab.shape and set(np.unique(lab)).issubset(set(range(9)))
PY
```

上述检查只核对文件存在与抽样标签合法性；还需独立核对**18/12 个病例名、患者身份、原始 CT 强度预处理、所有标签的编号与器官映射**。例如 train.txt 文件名含 slice 后缀时，必须用真实命名规则提取 case ID，检查训练/测试患者集合交集为空。

### 3.4 Synapse 增强和推理路径的实质

- `RandomGenerator`：先概率性地 90° 旋转+翻转，或另一概率分支进行约 ±20° 随机旋转；`elif random.random()>0.5` 是独立第二次抽样，**两分支的实际概率不能都简化成 50%**。原版 `random_rotate` 对 image 和 label 均采用最近邻 `order=0`。
- 训练切片：图像在需要 resize 时用三阶 `zoom(...,order=3)`，标签用最近邻 `order=0`；变为 `[1,224,224]`，标签为 Long。
- 测试体：H5 的 3D volume 一片片推理，图像三阶插值到 224×224，类别预测最近邻还原至原切片大小，最终以原体素网格聚合每器官的体级指标。
- 输入 CT 是单通道；`VMUNet.forward` 把一个通道 repeat 为三通道，与 `input_channels=3` 不冲突。
- 预处理并没有在此项目里完整定义原始 DICOM/NIfTI 到 `train_npz`、`test_vol_h5` 的制作过程。**不同来源的“Synapse NPZ包”不一定可互换。**

---

## 4. 预训练权重：名称、路径、结构三重校验

【已确认】`model_config['load_ckpt_path']` 指向：

```text
./pre_trained_weights/vmamba_small_e238_ema.pth
```

仓库的 `pre_trained_weights/` 仅包含说明文档，不包含实际权重。作者 README 的某段写 `./pretrained_weights/`，另一段和源码使用 `pre_trained_weights/`，应以当前源码为准。

**启动前**：

```bash
ls -lh pre_trained_weights/vmamba_small_e238_ema.pth
sha256sum pre_trained_weights/vmamba_small_e238_ema.pth
```

【已确认】`VMUNet.load_from` 读取 `torch.load(path)['model']`，先把同名编码器参数装入网络，再映射 `layers.0/1/2/3` 到 `layers_up.3/2/1/0`。**这不等于自动全部匹配加载**，应记录两轮控制台打印的 `Total model_dict`、`Total pretrained_dict`、`update` 数量及未加载键列表。

**权重检查清单**：

1. 权重必须来自可信的原作者或确认一致的镜像；保存下载链接和 SHA256。
2. 核对 checkpoint 顶层是否含 `model` 键；结构不符需另开适配实验，不能改原版后仍声称零差异。
3. 核对载入键名及 tensor shape；`load_from` 的过滤基于键名，不预先筛 shape，若尺寸不符会抛出 `size mismatch`。
4. 最终训练 checkpoint 与**预训练初始化权重**不是同一种文件：`latest.pth` 是包含优化器等状态的字典；`best.pth` 只存模型 state_dict。不能混用读取接口。
5. 不能只观察“程序没有报错”就认定预训练充分生效；须检查实际匹配参数数量。

---

## 5. 论文与源码：哪些超参可对齐，哪些有冲突

| 条目 | ISIC17 | ISIC18 | Synapse | 判断 |
|---|---|---|---|---|
| 数据尺寸 | 256×256 | 256×256 | 224×224 | 论文/配置一致 |
| 图像类型 | RGB | RGB | 单通道CT→重复为3通道 | 代码 |
| 输出通道 | 1 | 1 | 9，含背景+8器官 | 代码 |
| 主程序 | `train.py` | `train.py` | `train_synapse.py` | 代码 |
| 数据集开关 | 手动改 `datasets='isic17'` | 默认 `datasets='isic18'` | `datasets_name='synapse'` | 代码 |
| Epoch | 300 | 300 | 300 | 论文/代码 |
| Batch | 32 | 32 | 32 | 论文/代码 |
| AdamW lr | 1e-3 | 1e-3 | 1e-3 | 论文/代码 |
| AdamW wd | 0.01 | 0.01 | 0.01 | 代码 |
| AdamW betas | (.9,.999) | 同左 | 同左 | 代码 |
| Loss | BCE + Dice，各1 | 同左 | CE + Dice，各1 | 论文/代码 |
| Scheduler | CosineAnnealingLR | 同左 | 同左 | 论文/代码 |
| T_max | 50 | 50 | **100** | **Synapse 与论文50冲突** |
| eta_min | 1e-5 | 1e-5 | 1e-5 | 论文/代码 |
| seed | 42 | 42 | 42 | 代码；论文未充分披露 |
| AMP | False | False | False | 代码 |
| 深度（编码器） | [2,2,2,2] | 同左 | 同左 | 论文/代码 |
| 深度（解码器） | [2,2,2,1] | 同左 | 同左 | 论文/代码 |
| 预训练 | VMamba-S/ImageNet-1k | 同左 | 同左 | 论文/代码 |

**T_max 的含义不能读错**：`engine.py`、`engine_synapse.py` 都在**每个 epoch 结束**调用一次 `scheduler.step()`。因此 `T_max` 以 epoch 调度步数计，而不是 mini-batch 数。`CosineAnnealingLR` 在运行超过 `T_max` 之后不能简单理解为永远固定 `eta_min`，也不能把它与 `CosineAnnealingWarmRestarts` 混为一谈。**300 epoch + T_max=50/100 会形成不同的学习率轨迹**。请记录每个 epoch 的实际 lr，必要时画曲线验证。

**三种实验协议必须分别命名**：

- **A / 原版源码基线**：什么都不改，Synapse `T_max=100`、原指标实现、原选模方式。用于确认仓库作者实现能否重现已有结果。
- **B / 论文设置对齐实验**：在新的实验分支将 Synapse `T_max` 改 50；其余保持并记录所有变化。**不是 A**。
- **C / 评价/数据增强规范化实验**：在新实验分支修正空标签评价、HD95 spacing、Mask 插值/随机旋转等。用于方法学合理性或新论文；**不是 A，也不应拿 C 的指标与论文旧表直接进行同协议比较**。

建议实验命名：`vmunet_original_code`、`vmunet_paper_tmax50`、`vmunet_eval_protocol_v2`。任何改动都保留独立 Git commit、日志、参数快照。

---

## 6. 评估协议：最容易“跑出高分但不能用于论文”的地方

### 6.1 ISIC：全像素聚合与独立测试问题

`engine.py` 的 `val_one_epoch/test_one_epoch` 将全部预测和 GT 展平后汇总一个 `confusion_matrix`，由 `TP/FP/FN/TN` 计算：

```text
DSC = 2TP/(2TP + FP + FN)
IoU = TP/(TP + FP + FN)
ACC = (TP + TN)/(TP + TN + FP + FN)
SEN = TP/(TP + FN)
SPE = TN/(TN + FP)
```

这属于整个集合的**像素级 micro/pooled 统计**，不是“每张图算 Dice 再平均”的 macro 统计。论文中的 `mIoU` 标签也不能自动被理解成“前景与背景 IoU 的算术平均”——**实际代码计算的是前景 IoU**。比较其他论文时必须先统一指标定义。

当前 `train.py` 每个 epoch 计算 val loss，按最低 val loss 保存 `best.pth`；最后将这份 best 再对同一个 `val` 运行 `test_one_epoch`。如果称之为“完全未参与模型选择的独立测试集”，就不准确。若今后发表自己的方法，应引入分开的 train/val/test 或预注册固定模型选择规则；新的协议与原论文划分要分别陈述。

**两个细节**：

- 指标仅每 `val_interval=30` epoch 在验证阶段打印，但 val loss 每轮算、best 每轮可更新；不要误以为 best 仅在 30 的倍数轮产生。
- `confusion_matrix(y_true,y_pre)` 没有显式指定两类标签。当整个数据集和预测只出现一种类别时，返回的矩阵可能不是 2×2，随后索引 `[1,1]` 报错；虽然正常 ISIC 集通常包含前景，极端冒烟样本仍需警惕。

### 6.2 Synapse：空类别逻辑会直接扭曲指标

`utils.calculate_metric_percase` 代码语义如下：

| GT 是否有器官 | 预测是否有器官 | 当前代码 Dice | 当前代码 HD95 | 常规解读 |
|---|---|---:|---:|---|
| 有 | 有 | 调用 medpy | 调用 medpy | 可能有效 |
| 有 | 无 | 0 | 0 | 漏检，HD95不应被解读为完美 |
| 无 | 有 | **1** | **0** | **纯假阳性却得满分，明显反常** |
| 无 | 无 | 0 | 0 | 双空，如何计分需事先约定 |

这个 `elif pred.sum()>0 and gt.sum()==0: return 1,0` 不是普通数值噪声，而是**实现上的明确分支**。它是否在作者实际划分的完整 3D 病例中触发，需要用数据统计；未运行时不能推断其对论文 `81.08` 的确切影响幅度。

对未来论文建议定义并公开：
- GT 与预测都非空：Dice 正常计算，HD95 按固定 spacing 计算；
- GT 非空、预测空：Dice=0；HD95 应定义失败惩罚、缺失处理规则或另行报告，不能误解为 0mm；
- GT 空、预测非空：Dice=0；记录 FP，HD95 的不可定义/惩罚规则独立说明；
- 双空：Dice 可以约定 1 或 skip，但必须**明确与全体病例如何聚合**；
- 推荐同时披露每器官空 GT、空 Pred、纯 FP/纯 FN 的计数。

不要在原始基线中悄悄更改这个函数，否则结果无法同作者代码直接对比。

### 6.3 Synapse：HD95“单位 mm”尚未通过源码证明

`metric.binary.hd95(pred, gt)` **没有传入 voxelspacing**。此处 medpy 按默认像素/体素间距处理，等效使用单位体素距离。`test_single_volume` 内部使用 `z_spacing=1` 只是给**保存的** NIfTI 图像设置 spacing，并不反向改变前面已经计算完的 HD95。因此本实现输出的 HD95 **不能未经额外证据就标为原始 CT 物理毫米距离**。

若目标是复现作者源码数值，应保留当前算法并注明其单位假设；若目标是规范的医学物理距离评估，应从原 CT/H5 数据元信息恢复每体 spacing，并在新的评估协议传给 HD95，同时说明是否重采样、采用原空间或统一空间。**不能拿修正后的 mm 数值直接与原版表中 19.21 比大小。**

### 6.4 Synapse：平均方式要对齐

`engine_synapse.py` 对每例 8 器官数组求和，除以 `len(test_datasets)`，然后对器官取平均：**先按病人平均每个器官，再对 8 器官等权平均**。不包括背景 0；这不是所有病例体素合并的大 Dice，也不是按器官体积加权平均。比较基线和创新点必须使用同一病例清单、同一个评估函数、相同背景/空掩膜约定。

### 6.5 Synapse：最佳模型与验证时机

`train_synapse.py` 每个 epoch 用训练平均 loss 更新 `min_loss` 并保存 best，仅在第 100/200/300 个 epoch 评估 `test_vol`，最终装入最低训练 loss 权重执行测试。**这不是选择最高验证 Dice 的 epoch**。

`test_vol` 被训练过程周期性读取；即使源程序不自动拿它选 best，人工反复试参数后依据这个结果挑模型，也可能形成测试集调参。未来对比创新模块，建议事先固定超参、选择规则和测试次数；最终结果不要只选 seed 最好的一次。

---

## 7. 权重、日志与运行方式的几个隐藏坑

### 7.1 best 和 latest 不一样

| 路径 | 内容 | 正确用途 |
|---|---|---|
| `results/.../checkpoints/latest.pth` | `epoch/min_loss/min_epoch/loss/model_state_dict/optimizer_state_dict/scheduler_state_dict` | 恢复训练；原代码按存在即自动恢复 |
| `results/.../checkpoints/best.pth` | 模型参数字典，未加其他字段 | 评估；最终会重命名成含 epoch 和指标的文件 |
| `pre_trained_weights/vmamba_small_e238_ema.pth` | VMamba 预训练，需含 `model` 键 | 初始化 VM-UNet，并非训练完成的分割模型 |

【风险】默认 `work_dir` 按时间戳自动创建，启动时同名旧实验的 `latest.pth` 才会触发恢复。断点续训未保存完整随机生成器和 DataLoader 状态，不能保证与不间断训练逐位一致。保存日志和配置快照，勿拿时间戳目录名代替 commit ID。

### 7.2 原版 Synapse “只测试模式”目前不安全

`train_synapse.py` 的 `only_test_and_save_figs=True` 分支调用 `val_one_epoch(..., epoch,...)`，但这时 `epoch` 尚未赋值，按 Python 执行路径将出现 `UnboundLocalError`。此外 `outputs` 路径在修改 `config.work_dir` 前就已计算，`img_save_path` 与最终写入位置也需要检查。**原版不要直接使用这个模式**；若要仅推理，应另开独立修复分支修正并测试，不能在当前原版分支掩盖问题。

ISIC 的 `only_test_and_save_figs` 有独立实现，读取 `best_ckpt_path` 后执行 `test_one_epoch`，但需保证检查点是**纯模型 state_dict**、`img_save_path` 格式含正确目录分隔符、输出目录存在。

### 7.3 其他日志和计算量问题

- `engine.py` 的 `step += iter` 不等于每个 batch 加 1，TensorBoard 的 `global_step` 会异常增长；**不要用它直接推断实际优化器更新次数**。
- `train.py` 启动前调用 `cal_params_flops(model,256,logger)`，实际会先运行 `thop.profile` 的 GPU 前向。selective scan 自定义算子可能无法被 THOP 完整识别，得到的 FLOPs 估计可能少算；评估 FLOPs 时要记录统计工具和自定义算子处理方法。
- `set_seed(42)` 与 `cudnn.deterministic=True` 不能覆盖全部随机源和 CUDA 算子确定性；尤其 ISIC 旋转角初始化早于 seed。
- `nDiceLoss` 使用概率软 Dice，分母包含预测/真值的平方和，并且**训练 loss 遍历 9 类，包含背景**；最终报告的 Synapse Dice 排除背景。这两种 Dice 不能当成同一公式。
- `VSSM` 的部分参数设置了 `_no_weight_decay` 标记，但 `get_optimizer` 不依据这些标记分组；仍给所有参数统一 AdamW weight_decay。不要只依据参数标记猜测实际优化策略。
- `warnings.filterwarnings("ignore")` 可能掩盖版本弃用和数值警告；出错时在独立诊断副本中开启警告，但不要把这个改动混入正式基线。

---

## 8. 真正开始训练前的分层冒烟测试

**Gate 1：源码及环境**。在项目根目录运行（运行环境须安装基础依赖）：

```bash
git status --short
git rev-parse HEAD
python -m compileall -q configs datasets models engine.py engine_synapse.py train.py train_synapse.py utils.py
python -m pip check
nvidia-smi
```

语法检查通过只说明 Python 语法没问题；**不证明 GPU 扩展、数据、预训练、loss 兼容**。

**Gate 2：原始数据检查**。执行第 3 节的两份检查脚本，人工抽检至少 10 对 ISIC 图像/mask，查看尺寸与掩膜叠加；Synapse 至少抽检每个器官标签和全部病例清单，检查患者级泄漏。

**Gate 3：模型与 CUDA 算子前向**。在不需要数据集文件的情况下，先测试随机输入。以下代码会触发真实 selective-scan 计算；建议先用单个样本，避免过早耗尽显存：

```bash
python - <<'PY'
import torch
from models.vmunet.vmunet import VMUNet
assert torch.cuda.is_available(), "本项目原版需要可用的CUDA环境"
for n,size in [(1,256),(9,224)]:
    model=VMUNet(input_channels=3,num_classes=n,depths=[2,2,2,2],
                 depths_decoder=[2,2,2,1],drop_path_rate=.2,load_ckpt_path=None).cuda().eval()
    x=torch.randn(1,3,size,size,device="cuda")
    with torch.no_grad():
        y=model(x)
    print("class",n,"input",tuple(x.shape),"output",tuple(y.shape),
          "finite",torch.isfinite(y).all().item())
    assert y.shape==(1,n,size,size) and torch.isfinite(y).all()
    del model,x,y
PY
```

**Gate 4：权重载入**。完成第 4 节校验，真实调用一次 `model.load_from()`，记录加载键数量、异常、是否出现缺失键或形状不一致。此步骤必须有下载好的权重，不能以 `load_ckpt_path=None` 的 Gate 3 替代。

**Gate 5：真实 batch 的 loss/backward**。使用一个真实 ISIC mini-batch 和一个真实 Synapse mini-batch（若都准备了数据）做一次前向、Loss、反向和优化器更新，确认：
- `shape`、`dtype` 正确；
- 图像、Mask/label 没有 NaN/Inf，Synapse 标签为整数 0~8；
- loss 有限，关键参数梯度非空且有限；
- GPU 显存足够，速度和温度正常；
- 将冒烟输出写到独立目录，不要覆盖正式实验 `results`；
- 冒烟成功后**重新启动全新正式实验进程**，避免把已更新一次的权重带进基线。

**Gate 6：跑 1 个 epoch 的诊断实验**。在单独临时实验副本/分支把 `epochs` 暂设为 1，检查完整读写日志/权重；随后回到冻结的原始配置跑正式 300 epoch。**把试跑与正式运行明确分开**。

**Gate 7：正式实验冻结**。保存 Git SHA、数据包版本/清单哈希、权重 SHA256、`pip freeze`、`nvidia-smi`、命令行、config 快照和 42 随机种子；记录是否使用论文 T_max 或原版 T_max。

---

## 9. 训练运行及结果验收

### 9.1 ISIC 训练

默认 `configs/config_setting.py`：`datasets='isic18'`，执行：

```bash
python train.py
```

要跑 ISIC2017，在**独立实验副本或单独 Git 分支**将 `datasets` 设为 `isic17` 并记录 diff，再执行相同命令。不能只把 `isic2017` 数据放进目录而不修改默认数据集开关。

日志注意看：`train loss`、`val loss`、每 30 epoch 的 `f1_or_dsc/miou`、最终 `test of best model`。记住最终所谓 test 仍使用 val 数据集。

### 9.2 Synapse 训练

```bash
python train_synapse.py
```

日志注意看：每轮 `train loss`，每 100 epoch 记录的 `mean_dice/mean_hd95`，最后 best（由训练 loss 选出）对应的指标。**不要误将第 300 轮日志分数当作 best 模型结果**。

### 9.3 论文报告值——对照目标，不是保证值

| 数据集 | 论文报告 DSC | 论文报告其他指标 |
|---|---:|---|
| ISIC2017 | **89.03%** | IoU 80.23% |
| ISIC2018 | **89.71%** | IoU 81.35% |
| Synapse | **81.08%** | HD95 19.21（论文表述；实现的物理单位需核查） |

这些数值来自论文公开版本和原有中文对照文档。最终应以**确认版次的原文表格**为准，不得把来源不同的扩展实验、重训练结果、最优 seed 混在同一列。

**结果差异排查顺序**（不建议一出现差异就试新模块）：

1. 是否真实加载了作者 VMamba-S 预训练权重？加载匹配键多少？
2. 是否采用作者相同患者/图片划分？数据来源、图像和标签是否对应？
3. 是否使用了正确输入尺寸和类别数？是否跑错 ISIC17/18？
4. 是否用了正确的 T_max、batch、lr、wd、Loss、AMP 和随机策略？
5. 是否把 ISIC micro Dice 与 macro Dice、Synapse 体级 Dice 与切片级 Dice 混算？
6. Synapse 是否触发空 GT/纯 FP/纯 FN？HD95 有无真实 spacing？
7. 模型选择是按 train Loss、val Loss 还是验证 Dice？对照的是哪一个 checkpoint？
8. 以上一致后，再检查 GPU/CUDA 算子和非确定性，必要时重复 3 个以上随机种子，报告均值与标准差。

---

## 10. 面向硕士研究的复现阶段规划

**阶段 I：工程可运行**（不考虑论文数值）。环境检查→数据核验→真实前向→预训练权重→一次 loss/backward→单 epoch→完整日志和模型保存。交付物：环境锁定信息、数据清单、权重哈希、冒烟日志。

**阶段 II：源码复现**（禁止改原版语义）。使用原作者公开配置跑 ISIC17、ISIC18、Synapse；对每个任务保留最终的 Dice/IoU 或 Dice/HD95、训练曲线、随机种子及 Git commit。出现明显差异先定位，不要为了“涨到论文数字”任意修改多个变量。

**阶段 III：论文协议核验**。逐项核对论文正文/Table/Figure 与实际输出，明确代码 T_max 冲突、数据划分、模型选择、HD95 单位和空掩膜约定。对于无法确认的地方标为“待核验”，不能写“完全一致”。

**阶段 IV：形成可发表的公平基线**。先冻结和公开你的**修正/规范化评估协议**，在同一协议下重测原版与新增模块；任何修复都在独立实验分支实现，至少 3 个种子，最好记录各器官 Dice、HD95、参数量、GFLOPs、显存与推理时间。Synapse 只有少量病例，数值上涨需检查是否由一两例驱动。

### 10.1 推荐结果登记模板

| 实验ID | commit | 数据划分哈希 | 权重SHA256 | seed | lr / T_max | 评估协议 | 选模依据 | DSC | HD95 | 说明 |
|---|---|---|---|---|---|---|---|---|---|---|
| original_synapse_s42 | 待填 | 待填 | 待填 | 42 | 1e-3 / 100 | 作者原版 | train loss | 待测 | 待测 | 代码基线 |
| paper_tmax50_s42 | 待填 | 同上 | 同上 | 42 | 1e-3 / 50 | 作者原版 | train loss | 待测 | 待测 | 仅调度参数变化 |
| eval_v2_synapse_s42 | 待填 | 同上 | 同上 | 42 | 实际值 | 明确空类别及 spacing | 已声明 | 待测 | 待测 | 不和旧指标直接等价 |

### 10.2 理解了哪些才算掌握 VM-UNet？

建议能独立回答下列问题，而非只会运行 `train_synapse.py`：

1. 为什么二维图像要做四方向 selective scan？每个方向的输入形状、反转恢复、合并过程是什么？
2. `PatchEmbed2D`、`PatchMerging2D`、`PatchExpand2D`、`Final_PatchExpand2D` 各自改变什么张量维度？
3. `VSSBlock` 内 `LayerNorm→SS2D→DropPath→残差` 如何执行？SS2D 中 `A/B/C/D/dt` 的角色是什么？
4. 为什么编码器/解码器深度不对称？跳连为什么用加法？各层相加的通道数如何对齐？
5. 为什么 ISIC 输出 sigmoid，而 Synapse 直接输出 logits？二者与 BCE/CE 的接口是什么？
6. 为什么 Synapse 采用 2D 切片训练却在 3D 病例上评估？体素间距对 HD95 的影响是什么？
7. 为什么训练 Soft Dice、ISIC 汇总 Dice、Synapse 按病例/器官 Dice 是不同的定义？
8. 为什么随机种子 42 仍不意味着完全可复现？当前旋转角生命周期是什么？
9. 为什么 best.pth、latest.pth、vmamba 预训练权重不能互换？
10. 为什么“单次 Dice 高于论文表格”不足以证明新模块有效？如何建立固定协议、消融和统计重复？

---

## 11. 已知风险的优先处理顺序：不破坏原版基线

- **训练前必须解决的工程阻断**：环境/自定义算子不能导入，数据/预训练权重缺失，数据路径错误、形状/标签非法、显存不够。
- **训练前必须声明的协议差异**：Synapse T_max、ISIC 随机角和 Mask 插值、Synapse 空标签、HD95 物理 spacing、模型选择规则。
- **可先记录、再在新分支研究**：TensorBoard step、THOP FLOPs 统计、原版单卡代码、多卡 DDP、参数分组 weight decay。
- **不应在现有分支直接“顺手修”的事项**：任何会改变 Loss、预测、评估、预处理或权重初始化的改动。务必分别提交、分别跑基线、分别记录实验表。

## 12. 官方资料和分支内相关文档

- 原论文：https://arxiv.org/abs/2402.02491
- 原作者 GitHub：https://github.com/JCruan519/VM-UNet
- 本分支 README：../README.md
- 本分支已有配置对照：./REPRODUCTION_GUIDE_ZH.md
- 本分支已有问题审计：./REPRODUCTION_ISSUES_ZH.md
- 本分支已有源码阅读路线：./CODE_READING_GUIDE_ZH.md
- 本文件：`docs/VM_UNET_PREFLIGHT_AUDIT_ZH.md`

> **最终审计结论**：当前分支适合用作 VM-UNet 作者公开实现的学习与原版复现基线；但在完成数据与权重核查、CUDA 真实前向、明确 Synapse 指标/学习率协议、分离验证集与测试集解释之前，不能将训练结果直接表述为“完全复现原论文”。本审计是一份**静态代码与论文对照手册**，不是已验证通过的运行报告。
