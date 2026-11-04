# VM-UNet 缺失文件与代码风险审计（只记录，不修复）

审计范围：`VM-UNet-codex-comment-global`，起始提交 `b87827eb5a5faff00bd4b7b6505e56d3ca813ca2`。基于静态代码审阅、原始 README 与公开论文；**未在具备数据/预训练模型的 GPU 环境完成训练及数值验证**。下列“高/中”代表重现风险，不代表所有路径均已在运行时复现。

| 风险级别 | 位置 | 观察到的情况 | 影响及建议 |
|---|---|---|---|
| 严重 | `utils.py` `calculate_metric_percase` | 纯假阳性（GT 空，pred 非空）`return 1,0`，Dice 反而等于 1；双方空 `return 0,0` | 与通常 Dice 语义相反，可提高错误结果的分数；先界定作者论文评估协议，再在单独修复分支处理 |
| 高 | `train_synapse.py` `only_test_and_save_figs` 分支 | 调用 `val_one_epoch(...,epoch,...)` 时，`epoch` 未在该分支此前赋值 | 仅测试模式触发 `UnboundLocalError`；不得直接运行该模式，除非另开修复分支 |
| 高 | `configs/config_setting_synapse.py` 对比论文 IV-B | 代码 `CosineAnnealingLR T_max=100`，论文写 50 | 学习率轨迹不同；复现报告区分两种设置 |
| 高 | `configs/config_setting.py` 对比 README | ISIC 代码默认 `./data/isic2017` / `isic2018`，README 写 `./data/isic17` / `isic18` | 按错误目录布局可能无法加载数据 |
| 高 | `configs/config_setting_synapse.py` 对比 README | 代码 `lists/lists_Synapse`，README 写 `lists/list_Synapse` | 文件列表查找失败，核对用户实际目录 |
| 高 | `configs/*.py` 和 `models/vmunet/vmunet.py` | 预训练权重文件 `./pre_trained_weights/vmamba_small_e238_ema.pth` 不在 git 仓库 | 首次运行会无法载入，除非另行下载权重；记录哈希 |
| 高 | `data/` 与 `datasets/dataset.py` | 未包含 ISIC 图片及掩膜、Synapse NPZ/H5、train/test 列表 | 训练、测试不能直接运行；由数据提供方取得并核对划分 |
| 中 | `train_synapse.py` | best checkpoint 根据 **训练 loss** 最低而非验证 Dice 最高选出 | 报告“最优”需说明选择标准；可能偏离读者直觉 |
| 中 | `train.py` | best 按 val loss 选，结束后仍在同一 val 目录测试 | 测试集与验证集角色混合；不能直接称为完全独立的无偏测试集 |
| 中 | `engine.py` 的 `confusion_matrix` | 不显式传 `labels=[0,1]`，之后固定按 2×2 索引 | 当整个集只有一个类别时可能索引出错 |
| 中 | `datasets/dataset.py` `NPY_datasets` | 分别排序 image 和 mask 文件名，再按数组下标配对 | 文件名错位或数量不符会错配/越界；先核对 stem 集合 |
| 中 | `configs/config_setting_synapse.py` | 显式 import `ml_collections`，README 未列该依赖 | 全新环境中可能 `ModuleNotFoundError` |
| 中 | README 安装指导 | PyTorch 示例 cu117，与 `mamba_ssm` / `causal_conv1d` wheel 示例 cu118 不一致 | 自定义扩展动态链接或 ABI 兼容风险；核实本机 CUDA 环境 |
| 中 | `utils.py` `test_single_volume` | 输出 NIfTI 的 `z_spacing` 在配置中固定为 1 | 与真实毫米空间间距可能不同；不得默认 HD95 为严格物理尺度 mm |
| 中 | `train_synapse.py` | 评估 split 名为 `test_vol`，并周期性记录指标 | 明确测试数据是否曾用于调参，避免数据泄漏描述不清 |
| 中 | `models/vmunet/vmunet.py` `load_from` | 仅按 key 是否存在筛选预训练参数，未显式判断 shape | 当权重文件版本不匹配时可能 `size mismatch` |
| 低 | 项目环境 | 未见锁定所有包及系统版本的 `conda-lock`/哈希清单 | 跨机器复现难；在实验日志另外保存版本记录 |

## 阅读与安全提示

1. 本分支不修任何上述代码问题；所有解决方案只能在新的修复或消融实验分支中实施，以保持论文基线行为原封不动。
2. Python 编译检查只能证明语法，不证明训练流程、CUDA Kernel、预训练权重、数据集及评价逻辑正确。
3. 使用 PyTorch 加载不可信 `.pth` 可能有反序列化风险，应仅使用可信权重源并验证 SHA256。
4. 若上传了用户所持正式论文 PDF，可进一步逐页核对参数表和实现内容，避免不同论文版本的差异。

参考：https://arxiv.org/html/2402.02491 ，https://github.com/JCruan519/VM-UNet
