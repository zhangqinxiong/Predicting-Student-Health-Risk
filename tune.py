import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_TRIALS = 20
N_FOLDS = 3

train = pd.read_csv('input/train.csv')
target = train['health_condition']

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

numeric_features = [
    'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
    'step_count', 'exercise_duration', 'water_intake'
]
ordinal_features = ['stress_level', 'sleep_quality', 'physical_activity_level']
ordinal_categories = [
    ['low', 'medium', 'high', 'missing'],
    ['poor', 'average', 'good', 'missing'],
    ['sedentary', 'moderate', 'active', 'missing']
]
onehot_features = ['diet_type', 'smoking_alcohol', 'gender']

for col in numeric_features:
    train[col] = train[col].fillna(train[col].median())
for col in ordinal_features + onehot_features:
    train[col] = train[col].fillna('missing')

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), onehot_features),
    ('num', 'passthrough', numeric_features)
])

X = preprocessor.fit_transform(train)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

class_weights = (1 / target.value_counts(normalize=True)).to_dict()
sample_weights = np.array([class_weights[c] for c in target])

def objective_catboost(trial):
    params = {
        'task_type': 'GPU',
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 100,
        'eval_metric': 'Accuracy',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0, 5),
    }
    scores = []
    for train_idx, val_idx in skf.split(X, target_encoded):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = target_encoded[train_idx], target_encoded[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=0)
        preds = model.predict(X_val)
        scores.append(balanced_accuracy_score(y_val, preds))
    return np.mean(scores)

def objective_xgboost(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'device': 'cuda',
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'random_state': RANDOM_STATE,
        'verbosity': 0,
        'n_estimators': 2000,
        'early_stopping_rounds': 100,
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
    }
    scores = []
    for train_idx, val_idx in skf.split(X, target_encoded):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = target_encoded[train_idx], target_encoded[val_idx]
        sw = sample_weights[train_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=sw, verbose=0)
        preds = model.predict(X_val)
        scores.append(balanced_accuracy_score(y_val, preds))
    return np.mean(scores)

def objective_lightgbm(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'device': 'gpu',
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_estimators': 500,
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
    }
    scores = []
    for train_idx, val_idx in skf.split(X, target_encoded):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = target_encoded[train_idx], target_encoded[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
        preds = model.predict(X_val)
        scores.append(balanced_accuracy_score(y_val, preds))
    return np.mean(scores)

models_to_tune = [
    ('XGBoost', objective_xgboost),
]

best_params = {}

for name, objective_fn in models_to_tune:
    print(f'\n{"=" * 60}')
    print(f'  Tuning {name}')
    print(f'{"=" * 60}')
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True)
    print(f'\nBest {name} score: {study.best_value:.6f}')
    print(f'Best {name} params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    best_params[name] = study.best_params

print(f'\n{"=" * 60}')
print('  ALL TUNING RESULTS')
print(f'{"=" * 60}')
for name, params in best_params.items():
    print(f'\n{name}:')
    for k, v in params.items():
        print(f'  {k}: {v}')
