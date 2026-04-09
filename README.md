# EEG Emotion Recognition Demo

This repository contains the code, experiment scripts, and written analysis for an EEG-based emotion recognition project developed for a competition setting.

The project is not organized as a generic library. It is an experiment-driven research repo. The core question throughout the work is:

How can we preserve the stability of trial-level statistical EEG features while introducing dynamic temporal information in a controlled way under subject-level evaluation?

## Current Conclusion

Under the current dataset and evaluation protocol, the most reliable method family is a lightweight dual-branch architecture:

- statistical branch as the main trunk
- GRU temporal branch as dynamic supplement
- `seq_gate` for one-way stat-to-sequence interaction
- vector gated fusion for branch fusion
- light multitask group supervision with `group_loss_weight=0.02`

The finalized structural configuration is:

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

The best training recipe found in the latest sweep is:

- `lr = 1e-3`
- `batch_size = 64`
- `weight_decay = 2e-4`
- `early_stopping_patience = 15`
- `epochs = 200`

With that recipe, the current best single-model hold-out result is:

- `hold-out emotion accuracy = 0.78125`

## Repository Layout

- `eeg_pipeline/`: model definitions, feature extraction, preprocessing, normalization, and dataset utilities
- `scripts/`: runnable training, evaluation, benchmark, and utility entrypoints
- `实验分析/`: stage-by-stage experiment writeups in Chinese
- `上下文/`: internal planning and direction notes
- `output/doc/`: generated Word reports
- `README_pipeline.md`: short command-oriented pipeline note

## What Is Included

This repo is intended to include:

- source code
- experiment scripts
- textual analysis and stage summaries
- generated lightweight documentation

This repo is intentionally configured to exclude:

- raw competition dataset
- generated feature caches
- model checkpoints
- large experiment outputs
- temporary files

Those paths are ignored via `.gitignore`.

## Environment

The project has been run in the local conda environment named `pytorch`.

Typical command pattern:

```powershell
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\train_multitask_dual_branch.py
```

Or, if `conda run` works correctly on your machine:

```powershell
conda run -n pytorch python scripts/train_multitask_dual_branch.py
```

## Main Entry Points

Core scripts:

- `scripts/prepare_features.py`: build trial-level statistical DE features
- `scripts/prepare_graph_features.py`: build window-level sequence tensors
- `scripts/train_baseline.py`: traditional ML baseline
- `scripts/train_multitask_dual_branch.py`: finalized multitask dual-branch training entrypoint
- `scripts/run_training_strategy_sweep.py`: compact training-only hyperparameter sweep
- `scripts/run_final_strategy_suite.py`: seed-ensemble evaluation for the finalized mainline

Additional benchmark runners are listed in [README_pipeline.md](README_pipeline.md) and `scripts/README.md`.

## Evaluation Protocol

All main experiments are compared under:

- subject-level `GroupKFold` with 5 folds
- one fixed hold-out subject split used as a pseudo-private validation set

This is important. Trial-level random splitting is not used for the main conclusions.

## Notes For Public Upload

If you upload this repo to GitHub, do not include:

- `官方数据集/`
- `artifacts/`
- `outputs_v2/`
- `outputs_legacy/`
- `tmp/`

If you need to share results, prefer exporting:

- selected JSON/CSV summaries
- the Word reports under `output/doc/`
- screenshots or tables in the README / project report

## Reports

Current generated reports:

- `output/doc/项目全程尝试总结.docx`
- `output/doc/当前正式模型与最终结果说明.docx`

## Status

This repo captures a research workflow that has already converged on a clear mainline. Future work, if any, should focus more on better training protocol, model selection criteria, and documentation quality than on opening many new architecture branches.

