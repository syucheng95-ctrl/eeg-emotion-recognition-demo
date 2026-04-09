# 项目基础流程说明

下面给出这个项目最基础的一套运行流程，主要用于快速理解“特征准备 - 训练 - 预测 - 批量实验”的基本顺序。

建议在本地已有的 `pytorch` conda 环境中运行。

## 运行环境

推荐命令形式：

```powershell
conda run -n pytorch python scripts/prepare_features.py
```

如果你的机器上 `conda run` 不稳定，也可以直接调用环境里的 Python：

```powershell
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\prepare_features.py
```

## 一条基础执行链

```powershell
conda run -n pytorch python scripts/prepare_features.py
conda run -n pytorch python scripts/train_baseline.py
conda run -n pytorch python scripts/predict_test.py
conda run -n pytorch python scripts/run_experiments.py
conda run -n pytorch python scripts/prepare_graph_features.py
conda run -n pytorch python scripts/train_gnn.py
conda run -n pytorch python scripts/predict_gnn_test.py
```

## 这些命令分别做什么

- `scripts/prepare_features.py`
  生成 trial 级统计特征缓存

- `scripts/train_baseline.py`
  训练传统机器学习基线

- `scripts/predict_test.py`
  用传统基线对公开测试集导出预测结果

- `scripts/run_experiments.py`
  批量运行传统基线相关实验

- `scripts/prepare_graph_features.py`
  生成窗口级图/序列输入缓存

- `scripts/train_gnn.py`
  训练最初的图神经网络基线

- `scripts/predict_gnn_test.py`
  用图模型对公开测试集导出预测结果

## 常见输出文件

基础流程通常会产生以下输出：

- `artifacts/train_features.npz`
- `artifacts/test_features.npz`
- `outputs_v2/baseline_cv_report.json`
- `outputs_v2/baseline_model.pkl`
- `outputs_v2/submission_public_test.xlsx`
- `outputs_v2/experiment_summary.csv`
- `outputs_v2/experiment_details.json`
- `outputs_v2/gnn_cv_report.json`
- `outputs_v2/gnn_model.pt`
- `outputs_v2/submission_public_test_gnn.xlsx`

## 说明

这个文件只负责给出“最基础的执行链”。如果你想看更完整的脚本功能列表，请看：

- `README.md`
- `scripts/README.md`

