# 脚本目录说明

本目录存放项目中主要的可执行 Python 脚本，包括数据准备、训练、预测、批量 benchmark 和内部辅助脚本。

## 1. 数据准备

- `prepare_features.py`
  构建 trial 级统计 DE 特征缓存

- `prepare_graph_features.py`
  构建窗口级序列/图输入张量

- `prepare_raw_patch_features.py`
  构建 raw-patch 模型所需的更细粒度输入

## 2. 训练脚本

- `train_baseline.py`
  训练传统机器学习基线

- `train_window_model.py`
  训练非图窗口序列模型

- `train_gnn.py`
  训练最初的图神经网络基线

- `train_graph_model.py`
  训练图序列模型

- `train_graph_dann.py`
  训练带域适配的图模型

- `train_raw_patch_model.py`
  训练 raw-patch A/B/C 模型

- `train_dual_branch_model.py`
  训练轻量双分支融合模型

- `train_multitask_dual_branch.py`
  训练当前正式版多任务双分支模型（情绪头 + 群体头）

- `train_adversarial_dual_branch.py`
  训练带 gradient reversal 的对抗式双分支模型

## 3. 预测脚本

- `predict_test.py`
  用传统基线导出公开测试集预测结果

- `predict_gnn_test.py`
  用图模型导出公开测试集预测结果

## 4. 批量实验与 benchmark 脚本

- `run_experiments.py`
  传统基线批量实验

- `run_fair_benchmarks.py`
  公平对比实验套件

- `run_gnn_experiments.py`
  原始 GNN 路线批量实验

- `run_graph_progression.py`
  图模型阶段式推进实验

- `run_preprocessing_ablation.py`
  预处理消融实验

- `run_raw_patch_benchmarks.py`
  raw-patch 路线 benchmark

- `run_fusion_benchmarks.py`
  输出层概率融合实验

- `run_dual_branch_benchmarks.py`
  双分支模型 benchmark

- `run_multitask_benchmarks.py`
  多任务双分支 benchmark

- `run_multitask_extreme_search.py`
  面向多任务 gated 双分支的大规模分阶段搜索

- `run_ensemble_experiments.py`
  对多候选模型做 hold-out ensemble 评估

- `run_final_strategy_suite.py`
  面向阶段 14 正式版主线的 seed ensemble 评估

- `run_training_strategy_sweep.py`
  固定正式架构后，仅围绕训练超参数做小规模搜索

- `run_adversarial_benchmarks.py`
  对抗式群体约束 benchmark

- `run_stat_branch_benchmarks.py`
  强化统计分支 benchmark

- `run_interaction_benchmarks.py`
  更强双分支交互 benchmark

## 5. 内部辅助脚本

- `_bootstrap.py`
  负责把项目根目录加入 `sys.path`，方便脚本直接运行

## 6. 使用建议

如果你刚接触这个仓库，建议优先看：

1. `prepare_features.py`
2. `train_baseline.py`
3. `train_multitask_dual_branch.py`
4. `run_training_strategy_sweep.py`

如果你想理解项目完整实验脉络，建议同时阅读：

- `README.md`
- `README_pipeline.md`
- `实验分析/` 目录下的各阶段文本

