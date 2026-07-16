import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
target = train['health_condition']
test_ids = test['id']

le = LabelEncoder()
y = le.fit_transform(target)
class_names = le.classes_

numeric_features = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure', 'step_count', 'exercise_duration', 'water_intake']
ordinal_features = ['stress_level', 'sleep_quality', 'physical_activity_level']
onehot_features = ['diet_type', 'smoking_alcohol', 'gender']
ordinal_categories = [
    ['low', 'medium', 'high', 'missing'],
    ['poor', 'average', 'good', 'missing'],
    ['sedentary', 'moderate', 'active', 'missing']
]

for df in [train, test]:
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())
    for col in ordinal_features + onehot_features:
        df[col] = df[col].fillna('missing')

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), onehot_features),
    ('num', 'passthrough', numeric_features)
])

X = preprocessor.fit_transform(train)
X_test = preprocessor.transform(test)

class_weights = (1 / target.value_counts(normalize=True)).to_dict()
sample_weights = np.array([class_weights[c] for c in target])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = [
    ('CatBoost', lambda: CatBoostClassifier(
        task_type='GPU', auto_class_weights='Balanced',
        early_stopping_rounds=100, eval_metric='Accuracy',
        random_seed=RANDOM_STATE, verbose=0
    )),
    ('XGBoost', lambda: xgb.XGBClassifier(
        objective='multi:softprob', num_class=3,
        device='cuda', tree_method='hist', eval_metric='mlogloss',
        early_stopping_rounds=100, random_state=RANDOM_STATE,
        verbosity=0, max_depth=4, learning_rate=0.0195,
        min_child_weight=1.407, subsample=0.73,
        colsample_bytree=0.755, gamma=1.357
    )),
    ('LightGBM', lambda: lgb.LGBMClassifier(
        objective='multiclass', num_class=3, device='gpu',
        class_weight='balanced', early_stopping_round=100,
        metric='multi_logloss', random_state=RANDOM_STATE, verbose=-1
    )),
]

model_test = {name: np.zeros((len(test), 3)) for name, _ in models}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f'Fold {fold + 1}/{N_FOLDS}')
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    for name, clf_fn in models:
        model = clf_fn()
        if name == 'CatBoost':
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=0)
        elif name == 'XGBoost':
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], sample_weight=sample_weights[tr_idx], verbose=0)
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100)])
        model_test[name] += model.predict_proba(X_test) / N_FOLDS

weights = {'CatBoost': 0.7, 'XGBoost': 0.1, 'LightGBM': 0.2}
ensemble = sum(weights[name] * model_test[name] for name, _ in models)

submission = pd.DataFrame({'id': test_ids, 'health_condition': class_names[ensemble.argmax(axis=1)]})
submission.to_csv('output/submission.csv', index=False)
print('Done! output/submission.csv saved.')
