# Scripts Overview

All runnable Python entrypoints live in this folder.

## Data Preparation

- `prepare_features.py`: build trial-level statistical DE features.
- `prepare_graph_features.py`: build window-level sequence tensors.
- `prepare_raw_patch_features.py`: build raw patch tensors for raw-patch models.

## Training

- `train_baseline.py`: traditional ML baseline on trial-level features.
- `train_window_model.py`: non-graph window sequence model.
- `train_gnn.py`: original GNN baseline.
- `train_graph_model.py`: graph sequence models.
- `train_graph_dann.py`: graph model with domain adaptation.
- `train_raw_patch_model.py`: raw-patch A/B/C models.
- `train_dual_branch_model.py`: new lightweight dual-branch fusion model.
- `train_multitask_dual_branch.py`: multitask dual-branch model with emotion/group heads.
- `train_adversarial_dual_branch.py`: adversarial dual-branch model with gradient-reversal group head.

## Prediction

- `predict_test.py`: baseline public-test prediction export.
- `predict_gnn_test.py`: GNN public-test prediction export.

## Experiment Runners

- `run_experiments.py`: traditional baseline sweeps.
- `run_fair_benchmarks.py`: fair comparison benchmark suite.
- `run_gnn_experiments.py`: original GNN benchmark sweep.
- `run_graph_progression.py`: stage-wise graph progression benchmarks.
- `run_preprocessing_ablation.py`: preprocessing ablation benchmarks.
- `run_raw_patch_benchmarks.py`: raw-patch benchmark suite.
- `run_fusion_benchmarks.py`: new weighted-probability fusion benchmarks.
- `run_dual_branch_benchmarks.py`: new dual-branch benchmark suite.
- `run_multitask_benchmarks.py`: multitask dual-branch benchmark suite.
- `run_multitask_extreme_search.py`: staged large-grid search for multitask gated dual-branch tuning.
- `run_ensemble_experiments.py`: evaluate holdout-only ensemble variants over trained multitask candidates.
- `run_final_strategy_suite.py`: evaluate seed-ensemble strategies for the finalized phase-14 multitask mainline.
- `run_training_strategy_sweep.py`: compact training-only sweep over `lr` / `batch_size` / `weight_decay` / early-stopping patience.
- `run_adversarial_benchmarks.py`: adversarial group-constraint benchmark suite.
- `run_stat_branch_benchmarks.py`: stronger-statistical-branch benchmark suite.
- `run_interaction_benchmarks.py`: stronger branch-interaction benchmark suite.

## Internal Helper

- `_bootstrap.py`: adds the project root to `sys.path` so scripts can run from this folder.
