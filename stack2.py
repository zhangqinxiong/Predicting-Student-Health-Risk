import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

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

# StandardScaler for LogisticRegression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

class_weights = (1 / target.value_counts(normalize=True)).to_dict()
sample_weights = np.array([class_weights[c] for c in target])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

base_models = [
    ('CatBoost', lambda: CatBoostClassifier(
        task_type='GPU', auto_class_weights='Balanced',
        early_stopping_rounds=100, eval_metric='Accuracy',
        random_seed=RANDOM_STATE, verbose=0
    ), 'raw'),
    ('KNN', lambda: KNeighborsClassifier(
        n_neighbors=7, weights='distance',
        algorithm='kd_tree', leaf_size=100, n_jobs=-1
    ), 'scaled'),
    ('LogisticRegression', lambda: LogisticRegression(
        multi_class='multinomial', penalty='l2', C=1.0,
        solver='lbfgs', max_iter=1000, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    ), 'scaled'),
]

oof_preds = np.zeros((len(train), 3 * len(base_models)))
test_preds = np.zeros((len(test), 3 * len(base_models)))
fold_scores = {name: [] for name, _, _ in base_models}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_encoded)):
    print(f'\nFold {fold + 1}/{N_FOLDS}')
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = target_encoded[train_idx], target_encoded[val_idx]

    for m, (name, clf_fn, data_type) in enumerate(base_models):
        if data_type == 'scaled':
            X_tr_m = X_scaled[train_idx]
            X_va_m = X_scaled[val_idx]
            X_te_m = X_test_scaled
        else:
            X_tr_m = X_tr
            X_va_m = X_va
            X_te_m = X_test

        model = clf_fn()
        print(f'  -> training {name}...', end=' ', flush=True)
        if name == 'CatBoost':
            model.fit(X_tr_m, y_tr, eval_set=(X_va_m, y_va), use_best_model=True, verbose=0)
        else:
            model.fit(X_tr_m, y_tr)
        print('predicting...', end=' ', flush=True)

        val_proba = model.predict_proba(X_va_m)
        oof_preds[val_idx, m * 3:(m + 1) * 3] = val_proba
        test_preds[:, m * 3:(m + 1) * 3] += model.predict_proba(X_te_m) / N_FOLDS

        ba = balanced_accuracy_score(y_va, val_proba.argmax(axis=1))
        fold_scores[name].append(ba)
        print(f'  {name:22s}  Balanced Accuracy: {ba:.6f}')

print(f'\n{"=" * 50}')
print('  BASE MODEL CV RESULTS')
print(f'{"=" * 50}')
for name, scores in fold_scores.items():
    print(f'  {name:22s}  {np.mean(scores):.6f} (+/- {np.std(scores):.6f})')

meta = RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE)
meta.fit(oof_preds, target_encoded)
meta_ba = balanced_accuracy_score(target_encoded, meta.predict(oof_preds))
print(f'\n  Meta (Ridge) OOF Balanced Accuracy: {meta_ba:.6f}')

# Ridge coefficients
col_names = []
for n, _, _ in base_models:
    for c in ['at-risk', 'fit', 'unhealthy']:
        col_names.append(f'{n[:8]}_{c}')
print(f'\nRidge coefficients:')
for j, name in enumerate(col_names):
    print(f'  {name:25s}  {meta.coef_[0,j]:+8.4f}  {meta.coef_[1,j]:+8.4f}  {meta.coef_[2,j]:+8.4f}')

final_preds = meta.predict(test_preds)
submission = pd.DataFrame({'id': test_ids, 'health_condition': class_names[final_preds]})
submission.to_csv('output/submission.csv', index=False)
print('\nDone! output/submission.csv saved (not submitted).')
