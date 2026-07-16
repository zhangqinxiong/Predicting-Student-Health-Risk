import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5
N_CLASSES = 3

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
target = train['health_condition']
test_ids = test['id']

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
class_names = label_encoder.classes_

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

base_models = [
    ('CatBoost', lambda: CatBoostClassifier(
        task_type='GPU', auto_class_weights='Balanced',
        early_stopping_rounds=100, eval_metric='Accuracy',
        random_seed=RANDOM_STATE, verbose=0
    ), {'use_best_model': True}, None),
    ('XGBoost', lambda: xgb.XGBClassifier(
        objective='multi:softprob', num_class=N_CLASSES,
        device='cuda', tree_method='hist', eval_metric='mlogloss',
        early_stopping_rounds=100, random_state=RANDOM_STATE,
        verbosity=0, n_estimators=2000,
        max_depth=4, learning_rate=0.0195, min_child_weight=1.41,
        subsample=0.73, colsample_bytree=0.76, gamma=1.36,
        reg_alpha=0.2875, reg_lambda=1.6e-5
    ), {}, 'sample_weight'),
    ('LightGBM', lambda: lgb.LGBMClassifier(
        objective='multiclass', num_class=N_CLASSES,
        device='gpu', class_weight='balanced',
        early_stopping_round=100, metric='multi_logloss',
        random_state=RANDOM_STATE, verbose=-1
    ), {}, None),
]

oof_preds = np.zeros((len(train), N_CLASSES * len(base_models)))
test_preds = np.zeros((len(test), N_CLASSES * len(base_models)))
fold_scores = {name: [] for name, _, _, _ in base_models}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_encoded)):
    print(f'\n{"=" * 50}')
    print(f'  Fold {fold + 1}/{N_FOLDS}')
    print(f'{"=" * 50}')
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = target_encoded[train_idx], target_encoded[val_idx]

    for m, (name, clf_fn, fit_params, sw_key) in enumerate(base_models):
        model = clf_fn()
        fp = fit_params.copy()

        eval_set = [(X_val_fold, y_val_fold)]
        if name == 'CatBoost':
            model.fit(X_train_fold, y_train_fold, eval_set=eval_set[0], **fp)
        elif name == 'XGBoost':
            model.fit(X_train_fold, y_train_fold, eval_set=eval_set,
                      sample_weight=sample_weights[train_idx], **fp)
        else:
            model.fit(X_train_fold, y_train_fold, eval_set=eval_set,
                      callbacks=[lgb.early_stopping(100)], **fp)

        val_proba = model.predict_proba(X_val_fold)
        oof_preds[val_idx, m * N_CLASSES:(m + 1) * N_CLASSES] = val_proba
        test_proba = model.predict_proba(X_test)
        test_preds[:, m * N_CLASSES:(m + 1) * N_CLASSES] += test_proba / N_FOLDS

        val_preds = val_proba.argmax(axis=1)
        ba = balanced_accuracy_score(y_val_fold, val_preds)
        fold_scores[name].append(ba)
        print(f'  {name:12s}  Balanced Accuracy: {ba:.6f}')

print(f'\n{"=" * 50}')
print('  BASE MODEL CV RESULTS')
print(f'{"=" * 50}')
for name, scores in fold_scores.items():
    print(f'  {name:12s}  {np.mean(scores):.6f} (+/- {np.std(scores):.6f})')

meta_model = RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE)
meta_model.fit(oof_preds, target_encoded)
meta_train_preds = meta_model.predict(oof_preds)
meta_train_ba = balanced_accuracy_score(target_encoded, meta_train_preds)
print(f'\n  Meta (Ridge) OOF Balanced Accuracy: {meta_train_ba:.6f}')

final_preds = meta_model.predict(test_preds)
submission = pd.DataFrame({'id': test_ids, 'health_condition': class_names[final_preds]})
submission.to_csv('output/submission.csv', index=False)
print('\nDone! output/submission.csv saved (not submitted).')
