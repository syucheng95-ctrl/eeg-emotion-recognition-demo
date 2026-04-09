# 脑电情绪识别项目代码与实验整理

本仓库用于整理一个面向比赛场景的脑电情绪识别项目，包括：

- 核心代码
- 各阶段实验脚本
- 分阶段实验分析文本
- 最终生成的 Word 报告

这个仓库不是通用算法库，而是一个以实验推进为主的研究型代码仓库。项目的核心问题始终是：

在当前比赛数据规模、被试级评估协议和群体混杂背景下，如何在保留 trial 级统计特征稳定性的前提下，以受约束的方式引入时序动态信息，从而得到更稳健的跨被试情绪识别模型。

## 当前主线结论

在当前数据与评估协议下，项目最终收束出的正式方法主线是一个轻量双分支模型：

- 统计分支作为稳定主干
- GRU 时序分支作为动态补充
- `seq_gate` 用于统计到时序的单向条件交互
- vector gated fusion 用于两分支融合
- 轻度群体多任务监督，`group_loss_weight = 0.02`

正式结构配置如下：

- `stat_hidden_channels = 96`
- `stat_num_layers = 2`
- `temporal_type = gru`
- `gru_layers = 2`
- `fusion_type = gated`
- `gate_mode = vector`
- `interaction_mode = seq_gate`
- `group_loss_weight = 0.02`
- `dropout = 0.3`
- `standardize_sequence_input = True`

在固定正式架构不变的前提下，最新训练策略搜索得到的最佳训练配方为：

- `lr = 1e-3`
- `batch_size = 64`
- `weight_decay = 2e-4`
- `early_stopping_patience = 15`
- `epochs = 200`

当前最佳单模型结果为：

- `hold-out emotion accuracy = 0.78125`

## 仓库结构

- `eeg_pipeline/`
  核心模型、特征提取、预处理、归一化和数据相关模块

- `scripts/`
  训练、预测、批量 benchmark、实验 runner 等可执行脚本

- `实验分析/`
  各阶段实验分析文本，按阶段记录“做了什么、为什么做、结果怎样、结论是什么”

- `output/doc/`
  已生成的 Word 报告

- `README_pipeline.md`
  一个更偏命令说明的简版流程说明

## 仓库中包含什么

本仓库公开保留的内容主要有：

- 代码
- 脚本
- 实验分析文本
- 生成好的文档报告

本仓库刻意不包含以下内容：

- 原始比赛数据
- 特征缓存
- 训练权重
- 大体积实验输出
- 临时文件

这些内容已经通过 `.gitignore` 排除。

## 环境说明

本项目主要在本地 conda 环境 `pytorch` 中运行。

常见运行方式如下：

```powershell
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\train_multitask_dual_branch.py
```

如果你本机的 `conda run` 工作正常，也可以写成：

```powershell
conda run -n pytorch python scripts\train_multitask_dual_branch.py
```

## 常用脚本入口

核心脚本包括：

- `scripts/prepare_features.py`
  构建 trial 级统计 DE 特征

- `scripts/prepare_graph_features.py`
  构建窗口级序列张量

- `scripts/train_baseline.py`
  传统机器学习基线

- `scripts/train_multitask_dual_branch.py`
  当前正式版多任务双分支模型训练入口

- `scripts/run_training_strategy_sweep.py`
  固定正式架构后，仅对训练超参数做小规模搜索

- `scripts/run_final_strategy_suite.py`
  面向正式版主线的 seed ensemble 评估

更多脚本说明见：

- `scripts/README.md`
- `README_pipeline.md`

## 评估协议

项目主要比较结果基于以下统一协议：

- 被试级 `GroupKFold`
- `5` 折交叉验证
- 额外固定一组 hold-out subjects 作为伪 private 验证集

这点非常重要。项目主结论不是基于 trial 随机切分得出的，而是基于更严格的被试级划分。

## 上传到 GitHub 时不建议包含的内容

以下目录或文件不建议公开上传：

- `官方数据集/`
- `artifacts/`
- `outputs_v2/`
- `outputs_legacy/`
- `tmp/`
- 其他本地缓存、大文件、模型权重

如果需要展示结果，更建议公开：

- 核心代码
- 分析文本
- 少量总结性 CSV / JSON
- Word 报告
- README 中的结果说明

## 当前报告

当前已生成的报告包括：

- `output/doc/项目全程尝试总结.docx`
- `output/doc/当前正式模型与最终结果说明.docx`

## 当前状态

这个仓库记录的是一个已经基本完成主线收束的研究过程。后续如果还要继续做，更值得投入的方向通常不是继续大规模开新结构，而是：

- 更严格的训练流程
- 更正式的模型选择标准
- 更好的结果整理与报告表达
- 在新的数据设定下重新验证当前主线是否仍然成立

