import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
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

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
test_preds = np.zeros((len(test), 3))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_encoded)):
    print(f'\nFold {fold + 1}/{N_FOLDS}')
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = target_encoded[train_idx], target_encoded[val_idx]

    model = CatBoostClassifier(
        task_type='GPU',
        auto_class_weights='Balanced',
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

    val_preds = model.predict(X_val_fold)
    ba = balanced_accuracy_score(y_val_fold, val_preds)
    fold_scores.append(ba)
    print(f'  Balanced Accuracy: {ba:.6f}')
    test_preds += model.predict_proba(X_test) / N_FOLDS

mean_ba = np.mean(fold_scores)
print(f'\nCatBoost CV Balanced Accuracy: {mean_ba:.6f} (+/- {np.std(fold_scores):.6f})')

submission = pd.DataFrame({'id': test_ids, 'health_condition': class_names[test_preds.argmax(axis=1)]})
submission.to_csv('output/submission.csv', index=False)
print('Done! output/submission.csv saved.')
