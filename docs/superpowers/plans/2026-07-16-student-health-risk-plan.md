# Student Health Risk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Single Python script that trains a CatBoost model with 5-fold CV and generates a submission CSV.

**Architecture:** One standalone script (`train.py`) handling data loading, preprocessing, encoding, model training with 5-fold CV, and submission generation.

**Tech Stack:** Python 3, pandas, scikit-learn, catboost

**Global Constraints:**
- Single file: `train.py`
- CatBoost with default params, `task_type='GPU'`, `class_weights='Balanced'`, early stopping
- 5-Fold CV averaging predictions on test set
- Ordinal encoding for ordered categoricals, OneHot for unordered
- Median imputation for numeric missing values, mode for categorical

---

### Task 1: Write train.py

**Files:**
- Create: `/home/ivi/zqx/Predicting-Student-Health-Risk/train.py`

**Interfaces:**
- Consumes: `train.csv`, `test.csv` in same directory
- Produces: `submission.csv`

Steps:

- [ ] **Write the complete script**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = train['health_condition']
test_ids = test['id']

numeric_features = [
    'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
    'step_count', 'exercise_duration', 'water_intake'
]

ordinal_features = ['stress_level', 'sleep_quality', 'physical_activity_level']
ordinal_categories = [
    ['low', 'medium', 'high'],
    ['poor', 'average', 'good'],
    ['sedentary', 'moderate', 'active']
]

onehot_features = ['diet_type', 'smoking_alcohol', 'gender']

for df in [train, test]:
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())
    for col in ordinal_features + onehot_features:
        df[col] = df[col].fillna(df[col].mode()[0])

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), onehot_features),
    ('num', 'passthrough', numeric_features)
])

X = preprocessor.fit_transform(train)
X_test = preprocessor.transform(test)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
test_preds = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, target)):
    print(f'Fold {fold + 1}/{N_FOLDS}')
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = target.iloc[train_idx], target.iloc[val_idx]

    model = CatBoostClassifier(
        task_type='GPU',
        class_weights='Balanced',
        early_stopping_rounds=100,
        eval_metric='Accuracy',
        random_seed=RANDOM_STATE,
        verbose=100
    )
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        use_best_model=True
    )
    test_preds += model.predict_proba(X_test) / N_FOLDS

pred_classes = model.classes_[test_preds.argmax(axis=1)]
submission = pd.DataFrame({'id': test_ids, 'health_condition': pred_classes})
submission.to_csv('submission.csv', index=False)
print('Done! submission.csv saved.')
```

- [ ] **Verify the script runs**

Run: `cd /home/ivi/zqx/Predicting-Student-Health-Risk && python3 train.py`
Expected: Script completes without errors, `submission.csv` is generated.

- [ ] **Verify submission format**

```bash
python3 -c "
import pandas as pd
sub = pd.read_csv('submission.csv')
print(sub.shape)
print(sub['health_condition'].value_counts())
print(sub.head())
"
```

Expected: 295800 rows, 3 columns (id, health_condition), valid class labels.
