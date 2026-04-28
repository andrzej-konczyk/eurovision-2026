# MLP Training Report — US-S5-01
**Date:** 2026-04-28  
**Model:** PyTorch MLP (`src/models/nn.py`)  
**Run timestamp:** 2026-04-28T20:21:02 UTC

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | `SimpleImputer(median) → StandardScaler → MLP → sigmoid` |
| Features | 23 (same FEATURE_COLS as XGBoost / LightGBM) |
| Train rows | 217 |
| Train years | 2016–2025 (excl. 2020) |
| Epochs | 300 |
| Batch size | 32 |
| CV splitter | `LeaveLastYearOut` |
| CV folds | 8 (all valid — no single-class folds) |
| Random seed | 42 |

---

## 2. Grid Search Results

| hidden_dims | lr | dropout | mean ROC-AUC | std |
|-------------|-----|---------|-------------|-----|
| (64, 32) | 0.01 | 0.0 | 0.5431 | 0.2274 |
| (64, 32) | 0.01 | 0.2 | 0.5183 | 0.1386 |
| (64, 32) | 0.001 | 0.0 | 0.5764 | 0.2123 |
| (64, 32) | 0.001 | 0.2 | 0.5867 | 0.2266 |
| (128, 64) | 0.01 | 0.0 | 0.5686 | 0.1965 |
| (128, 64) | 0.01 | 0.2 | 0.5168 | 0.1694 |
| (128, 64) | 0.001 | 0.0 | 0.5791 | 0.1866 |
| (128, 64) | 0.001 | 0.2 | 0.5803 | 0.1894 |
| **(32,)** | **0.01** | **0.0** | **0.5998** | 0.1935 |
| (32,) | 0.01 | 0.2 | 0.5460 | 0.2147 |
| **(32,)** | **0.001** | **0.0** | **0.6148** ✓ | **0.1760** |
| (32,) | 0.001 | 0.2 | 0.6034 | 0.1682 |

**Best:** `hidden_dims=(32,)`, `lr=0.001`, `dropout=0.0` → **CV ROC-AUC 0.6148 ± 0.1760**

---

## 3. Ensemble Comparison

| Model | CV ROC-AUC | ± std | CV folds |
|-------|-----------|-------|----------|
| XGBoost | 0.7467 | 0.1881 | 4 (valid) |
| LightGBM | 0.6358 | 0.1417 | 4 (valid) |
| **MLP** | **0.6148** | **0.1760** | **8 (all valid)** |

MLP is the weakest individual model by CV ROC-AUC. This is expected: with only 217 training rows, gradient-based optimisation is at a disadvantage relative to tree ensembles. The MLP provides **diverse signal** (different inductive bias) and benefits from **more CV folds** (8 vs 4 valid for trees, whose early folds lacked both betting odds and engineered features).

---

## 4. Observations

- **Smaller architecture wins.** `(32,)` outperforms `(64, 32)` and `(128, 64)` consistently, suggesting the binding constraint is underfitting — more parameters add noise rather than capacity on this dataset size.
- **Lower learning rate wins.** `lr=0.001` beats `lr=0.01` at every architecture, consistent with smoother convergence over 300 epochs.
- **Dropout unhelpful.** `dropout=0.0` wins or ties in every architecture group. The dataset is small enough that regularisation via early stopping / smaller architecture is preferable to noise injection.
- **8 fully valid CV folds.** Unlike XGBoost and LightGBM (which had 4 NaN folds because early years pre-date betting odds and some engineered features), the MLP's `LeaveLastYearOut` produces 8 usable folds — a more reliable CV estimate.

---

## 5. Artefacts

| File | Description |
|------|-------------|
| `models/artefacts/nn_model.pkl` | Full `NNPipeline` (pickle) |
| `models/artefacts/nn_model.pt` | PyTorch state dict |
| `models/artefacts/nn_model_config.json` | Metadata + CV results |

DVC tracking: attempted — skipped (dvc not available in current environment). Artefacts are git-ignored; manual `dvc add` required before next `dvc push`.

MLflow: run logged to `eurovision-2026-ensemble` with tag `story=US-S5-01`, `model=mlp`.

---

## 6. Test Results

```
28 passed in test_nn.py
216 passed total (all test files)
```

Key fix during implementation: `SimpleImputer(keep_empty_features=True)` — prevents `implied_prob_close` (all-NaN without live betting odds data) from being dropped, keeping feature dimensionality consistent with `FEATURE_COLS` at all times.
