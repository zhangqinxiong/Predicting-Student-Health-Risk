import os
import random
import time
import io
import contextlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping as XGBEarlyStopping
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_eval
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
N_FOLDS = 5
N_DIRICHLET = 2000

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
target = train['health_condition']
test_ids = test['id']

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
class_names = label_encoder.classes_
n_classes = len(class_names)

numeric_features = [
    'sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
    'step_count', 'exercise_duration', 'water_intake'
]

cat_features = ['stress_level', 'sleep_quality', 'physical_activity_level', 'diet_type', 'smoking_alcohol', 'gender']

for df in [train, test]:
    for col in numeric_features:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_features:
        df[col] = df[col].fillna('missing')

all_features = numeric_features + cat_features
X_base = train[all_features].copy()
X_test_base = test[all_features].copy()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_catb = np.zeros((len(train), 3))
oof_xgb = np.zeros((len(train), 3))
oof_lgb = np.zeros((len(train), 3))
test_catb = np.zeros((len(test), 3))
test_xgb = np.zeros((len(test), 3))
test_lgb = np.zeros((len(test), 3))

catb_scores = []
xgb_scores = []
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, target_encoded)):
    fold_start = time.time()
    te_start = time.time()
    print(f'\n{"="*60}')
    print(f'Fold {fold + 1}/{N_FOLDS}')
    print(f'{"="*60}')
    print(f'  Train size: {len(train_idx)}  |  Val size: {len(val_idx)}')

    # --- Multiclass Target Encoding (K=3 columns per cat feature) ---
    SMOOTH_M = 10.0
    cat_codes = lambda df: np.column_stack([df[col].astype('category').cat.codes for col in cat_features]).astype(np.float32)

    te_train_list, te_val_list, te_test_list = [], [], []
    for col in cat_features:
        train_raw = train[col].iloc[train_idx].values
        prior = np.bincount(target_encoded[train_idx], minlength=n_classes) / len(train_idx)
        categories, inv = np.unique(train_raw, return_inverse=True)
        cat_probas = np.zeros((len(categories), n_classes))
        for i in range(len(categories)):
            counts = np.bincount(target_encoded[train_idx][inv == i], minlength=n_classes)
            cat_probas[i] = (counts + SMOOTH_M * prior) / (counts.sum() + SMOOTH_M)
        proba_map = dict(zip(categories, cat_probas))
        encode = lambda vals: np.array([proba_map.get(v, prior) for v in vals])
        te_train_list.append(encode(train_raw))
        te_val_list.append(encode(train[col].iloc[val_idx].values))
        te_test_list.append(encode(test[col].values))

    te_train = np.hstack(te_train_list)
    te_val = np.hstack(te_val_list)
    te_test = np.hstack(te_test_list)

    X_train_fold = np.column_stack([X_base.iloc[train_idx][numeric_features].values, cat_codes(X_base.iloc[train_idx]), te_train])
    X_val_fold = np.column_stack([X_base.iloc[val_idx][numeric_features].values, cat_codes(X_base.iloc[val_idx]), te_val])
    X_test_fold = np.column_stack([X_test_base[numeric_features].values, cat_codes(X_test_base), te_test])

    y_train_fold, y_val_fold = target_encoded[train_idx], target_encoded[val_idx]
    te_time = time.time() - te_start

    # --- CatBoost ---
    t0 = time.time()
    catb = CatBoostClassifier(
        task_type='GPU',
        auto_class_weights='Balanced',
        early_stopping_rounds=100,
        eval_metric='MultiClass',
        random_seed=RANDOM_STATE,
        learning_rate=0.03,
        n_estimators=3000,
        verbose=0
    )
    catb.fit(Pool(X_train_fold, y_train_fold), eval_set=Pool(X_val_fold, y_val_fold), use_best_model=True)
    catb_train_t = time.time() - t0

    catb_val_proba = catb.predict_proba(X_val_fold)
    oof_catb[val_idx] = catb_val_proba
    catb_val_ba = balanced_accuracy_score(y_val_fold, catb_val_proba.argmax(axis=1))
    catb_scores.append(catb_val_ba)
    print(f'  CatBoost  | Val BA: {catb_val_ba:.6f} | iter: {catb.get_best_iteration()} | {catb_train_t:.1f}s')

    # --- XGBoost ---
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        eval_metric='mlogloss',
        device='cuda',
        random_state=RANDOM_STATE,
        verbosity=0,
        callbacks=[XGBEarlyStopping(rounds=100, min_delta=0.002, save_best=True)]
    )
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_fold)
    xgb.fit(X_train_fold, y_train_fold, sample_weight=sample_weights,
            eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    xgb_train_t = time.time() - t0

    xgb_val_proba = xgb.predict_proba(X_val_fold)
    oof_xgb[val_idx] = xgb_val_proba
    xgb_val_ba = balanced_accuracy_score(y_val_fold, xgb_val_proba.argmax(axis=1))
    xgb_scores.append(xgb_val_ba)
    print(f'  XGBoost   | Val BA: {xgb_val_ba:.6f} | iter: {xgb.best_iteration} | {xgb_train_t:.1f}s')

    # --- LightGBM ---
    t0 = time.time()
    lgb = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        class_weight='balanced',
        device='gpu',
        random_state=RANDOM_STATE,
        verbose=-1
    )
    with contextlib.redirect_stdout(io.StringIO()):
        lgb.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='multi_logloss',
                callbacks=[lgb_early_stopping(100, min_delta=0.002, verbose=False), lgb_log_eval(0)])
    lgb_train_t = time.time() - t0

    best_iter = lgb.best_iteration_
    lgb_val_proba = lgb.predict_proba(X_val_fold, num_iteration=best_iter)
    oof_lgb[val_idx] = lgb_val_proba
    lgb_val_ba = balanced_accuracy_score(y_val_fold, lgb_val_proba.argmax(axis=1))
    lgb_scores.append(lgb_val_ba)
    print(f'  LightGBM  | Val BA: {lgb_val_ba:.6f} | iter: {lgb.best_iteration_} | {lgb_train_t:.1f}s')

    # --- Test predictions (accumulate) ---
    t0 = time.time()
    catb_test_proba = catb.predict_proba(X_test_fold)
    test_catb += catb_test_proba / N_FOLDS
    xgb_test_proba = xgb.predict_proba(X_test_fold)
    test_xgb += xgb_test_proba / N_FOLDS
    lgb_test_proba = lgb.predict_proba(X_test_fold, num_iteration=best_iter)
    test_lgb += lgb_test_proba / N_FOLDS
    test_inf_time = time.time() - t0

    fold_time = time.time() - fold_start
    inference_time = fold_time - te_time - catb_train_t - xgb_train_t - lgb_train_t - test_inf_time
    print(f'  ─{"─"*57}')
    print(f'  Fold {fold+1} total: {fold_time:.1f}s')
    print(f'    TE: {te_time:.1f}s  |  Cat: {catb_train_t:.1f}s  |  XGB: {xgb_train_t:.1f}s  |  '
          f'LGB: {lgb_train_t:.1f}s  |  Val: {inference_time:.1f}s  |  Test: {test_inf_time:.1f}s')

print(f'\n{"="*60}')
print(f'CatBoost CV: {np.mean(catb_scores):.6f} +/- {np.std(catb_scores):.6f}')
print(f'XGBoost  CV: {np.mean(xgb_scores):.6f} +/- {np.std(xgb_scores):.6f}')
print(f'LightGBM CV: {np.mean(lgb_scores):.6f} +/- {np.std(lgb_scores):.6f}')

print(f'\n{"="*60}')
print('Dirichlet weight search...')
rng = np.random.default_rng(RANDOM_STATE)

uniform = np.ones(3) / 3
blend_uniform = uniform[0] * oof_catb + uniform[1] * oof_xgb + uniform[2] * oof_lgb
ba_uniform = balanced_accuracy_score(target_encoded, blend_uniform.argmax(axis=1))

ba_catb_only = balanced_accuracy_score(target_encoded, oof_catb.argmax(axis=1))
ba_xgb_only = balanced_accuracy_score(target_encoded, oof_xgb.argmax(axis=1))
ba_lgb_only = balanced_accuracy_score(target_encoded, oof_lgb.argmax(axis=1))

best_weight = None
best_ba = 0
results = []

for i in range(N_DIRICHLET):
    w = rng.dirichlet(np.ones(3))
    blended = w[0] * oof_catb + w[1] * oof_xgb + w[2] * oof_lgb
    ba = balanced_accuracy_score(target_encoded, blended.argmax(axis=1))
    results.append((ba, w))
    if ba > best_ba:
        best_ba = ba
        best_weight = w

top5 = sorted(results, key=lambda x: -x[0])[:5]
model_names = ['CatBoost', 'XGBoost', 'LightGBM']

print(f'  CatBoost only:       {ba_catb_only:.6f}')
print(f'  XGBoost only:        {ba_xgb_only:.6f}')
print(f'  LightGBM only:       {ba_lgb_only:.6f}')
print(f'  Uniform (1/3 each):  {ba_uniform:.6f}')
print(f'  Best Dirichlet:      {best_ba:.6f}  (', end='')
for i, name in enumerate(model_names):
    print(f'w_{name}={best_weight[i]:.4f}', end='  ' if i < 2 else '')
print(')')
print(f'  Top-5 weights:')
for ba, w in top5:
    parts = [f'{model_names[i]}={w[i]:.4f}' for i in range(3)]
    print(f'    BA: {ba:.6f}  |  {"  ".join(parts)}')

test_blend = best_weight[0] * test_catb + best_weight[1] * test_xgb + best_weight[2] * test_lgb
test_preds = test_blend.argmax(axis=1)

submission = pd.DataFrame({'id': test_ids, 'health_condition': class_names[test_preds]})
submission.to_csv('output/submission.csv', index=False)
print(f'\nDone! output/submission.csv saved.')
