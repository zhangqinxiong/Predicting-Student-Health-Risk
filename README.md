# Predicting Student Health Risk

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series%20S6E7-blue)](https://kaggle.com/competitions/playground-series-s6e7)

**Public LB: 0.95044** 🏆 — Multiclass classification predicting student health status (**fit / at-risk / unhealthy**) from lifestyle and physiological indicators.

> ⚠️ **Reproducibility note**: GPU training introduces minor numerical non-determinism even with fixed seeds. LB scores fluctuate ±0.0001–0.0002 between runs. Run 3× and pick the best submission.

## Problem

| Metric | Value |
|--------|-------|
| Classes | fit (5.8%), at-risk (85.9%), unhealthy (8.4%) |
| Train size | 690,088 |
| Test size | 295,753 |
| Evaluation | **Balanced Accuracy** (per-class recall averaged) |

Severe class imbalance — naive majority-class prediction yields 0.333 BA.

## Pipeline

### 1. Missing Value Imputation
- **7 numeric features**: median filling (train column median)
- **6 categorical features**: `'missing'` string filling

### 2. Baseline Features — 13 dimensions
```
7 numeric (passthrough)
6 categorical (integer codes, models use native categorical handling)
```

### 3. Per-Fold Multiclass Target Encoding — +18 dimensions
Custom smoothed target encoding generating **K columns per categorical feature** (one per class):

```
P(Y=k | X=c) = (count_k + α × prior_k) / (total + α)
```

- 3 classes × 6 = 18 dims
- Laplace smoothing with α=10
- Unseen categories fall back to fold prior

Computed **inside each fold** to prevent data leakage.

Joined with base features → **31 total dimensions**.

### 4. 3 Models × 5-Fold Cross Validation

| Model | Key Params | Categorical Handling | Best Model |
|-------|-----------|---------------------|------------|
| CatBoost GPU | lr=0.03, 3000 iter, MultiClass, Balanced, ES=100 | Native (cat_features) | `use_best_model=True` |
| XGBoost CUDA | lr=0.03, 3000 iter, mlogloss, balanced weights, ES=100, **min_delta=0.002** | Integer codes (no native) | `EarlyStopping(save_best=True)` |
| LightGBM GPU | lr=0.03, 3000 iter, multi_logloss, balanced, ES=100, **min_delta=0.002** | Native (categorical_feature) | `predict_proba(num_iteration=best_iter_)` |

### 5. Dirichlet-optimized Ensemble
2000 Dirichlet random trials on OOF predictions → optimal weighted blend.

> Note: Dirichlet weights vary between runs (~±0.2) due to GPU non-determinism, but ensemble LB stays within ±0.0001.

### 6. Submission
Best weights applied to test predictions → argmax → inverse label encoding.

```
Per fold (~37s):
  TE:     1.6s  (category_encoders, C-optimized)
  Cat:    7.4s
  XGB:    3.7s
  LGB:   18.4s
  Val:    2.2s
  Test:   4.8s

Total: ~3 min for full 5-fold pipeline
```

## Results

| Model | CV BA | Public LB |
|-------|-------|-----------|
| CatBoost | 0.94918 | — |
| XGBoost | 0.94899 | — |
| LightGBM | 0.94948 | — |
| Uniform blend | 0.94943 | — |
| **Dirichlet ensemble** | **0.94959** | **0.95044** |

## Key Experiments Log

| Experiment | CV BA | LB | Note |
|-----------|-------|-----|------|
| CatBoost baseline | 0.94942 | 0.94938 | Single model |
| + XGBoost (Dirichlet) | 0.94952 | 0.94976 | 2-model ensemble |
| + LightGBM | 0.94963 | 0.94980 | 3-model ensemble |
| + Target Encoding (custom) | 0.94965 | 0.95016 | Own TE with M=10 + count |
| + min_delta=0.002 (3× faster) | 0.94966 | 0.95013 | ES tuned, speed improved |
| + All OneHot (no Ordinal) | 0.94956 | 0.95011 | Simpler encoding |
| − OHE (TE only, 25 feats) | 0.94955 | 0.95027 | TE alone sufficient |
| + RF imputation | 0.94251 | 0.94550 | Worse |
| + NaN passthrough | 0.94920 | 0.94976 | Worse |
| Native cat (no TE, 13 feats) | 0.94955 | 0.94977 | Less generalization |
| Raw cat + TE (31 feats) | 0.94959 | **0.95044** | 🏆 **Best** |
| + Global seeds (`random`/`np`/`PYTHONHASHSEED`) | 0.94954 | 0.95037 | More reproducible |
| + Multiclass TE (K per cat) | 0.94956 | 0.95037 | Like-for-like TE replacement |
| TabNet (1 fold, 80K subset) | 0.87439 | — | Underperforms GBDT |
| + FT-Transformer | OOM | — | Too large for GPU (11GB) |

## Insights

1. **Raw categorical columns + TE beats TE alone** — 31 features (13 raw + 18 TE) achieves LB 0.95044 vs 0.95027 for TE-only. The raw categories give tree models extra split options beyond TE probability features.
2. **TE vs native categorical**: TE generalizes better (0.95027 vs 0.94977) because smoothed probabilities reduce overfitting to rare categories.
3. **min_delta=0.002** is critical for early stopping — without it, XGBoost/LightGBM run full 3000 iterations overfitting on logloss while accuracy plateaus.
4. **Per-fold target encoding** must be computed inside CV to prevent leakage; global TE inflates CV by 0.002-0.003.
5. **RF imputation hurts** — median fill outperforms both NaN passthrough and RF-based imputation.
6. **Training speed** improved 5× (213s → 37s per fold) by removing train set predict_proba, adding min_delta early stopping, and using `category_encoders.TargetEncoder`.
7. **Multiclass TE (K per cat) vs single-column TE**: Results are nearly identical (LB 0.95037 vs 0.95044). The additional per-class probability columns don't add value beyond the single-column smoothed encoding.
8. **GPU non-determinism**: Even with `random.seed`, `np.random.seed`, and `PYTHONHASHSEED`, tree models on GPU produce slightly different OOF predictions between runs (±0.0001 LB). Run multiple times and pick the best submission.
9. **Neural networks (TabNet, FT-Transformer)** underperform GBDT on this tabular dataset — TabNet at 0.874 CV vs GBDT ensemble at 0.949+.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script (3-model ensemble, 31 features) |
| `input/` | Competition data (train.csv, test.csv, sample_submission.csv) |
| `output/submission.csv` | Generated submission |
| `.gitignore` | Ignores data files, cache, and training logs |

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for CatBoost/XGBoost/LightGBM GPU training)
- `pip install pandas numpy scikit-learn catboost xgboost lightgbm category_encoders`

## Usage

```bash
pip install pandas numpy scikit-learn catboost xgboost lightgbm category_encoders
python train.py
```

## License

MIT
