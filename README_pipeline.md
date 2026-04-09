# EEG Baseline Pipeline

Run inside the existing `pytorch` conda environment:

```powershell
conda run -n pytorch python scripts/prepare_features.py
conda run -n pytorch python scripts/train_baseline.py
conda run -n pytorch python scripts/predict_test.py
conda run -n pytorch python scripts/run_experiments.py
conda run -n pytorch python scripts/prepare_graph_features.py
conda run -n pytorch python scripts/train_gnn.py
conda run -n pytorch python scripts/predict_gnn_test.py
```

Outputs:

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
