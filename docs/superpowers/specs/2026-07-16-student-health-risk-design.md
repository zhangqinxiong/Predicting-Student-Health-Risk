# Predicting Student Health Risk — Design

## Problem
Multi-class classification (3 classes: at-risk, unhealthy, fit) on synthetically generated student health behavior data. Evaluation metric: balanced accuracy.

## Data
- Train: 690,088 rows, 14 features + id + target
- Test: 295,800 rows
- Severe class imbalance: at-risk ~85.9%, unhealthy ~8.4%, fit ~5.8%
- All numerical features have <50% missing values (range 1-12%), all kept and median-imputed

## Feature Engineering

| Feature | Type | Encoding |
|---------|------|----------|
| sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake | Numeric | Median imputation |
| stress_level (low<medium<high) | Ordinal | OrdinalEncoder |
| sleep_quality (poor<average<good) | Ordinal | OrdinalEncoder |
| physical_activity_level (sedentary<moderate<active) | Ordinal | OrdinalEncoder |
| diet_type (veg/non-veg/balanced) | Nominal | OneHotEncoder |
| smoking_alcohol (yes/no/occasional) | Nominal | OneHotEncoder |
| gender (male/female/other) | Nominal | OneHotEncoder |

## Model
- **Algorithm:** CatBoost
- **Parameters:** Default, `task_type='GPU'`, `class_weights='Balanced'`, `early_stopping_rounds`
- **Validation:** 5-Fold Cross Validation, average predictions on test set
- **Output:** Single submission CSV

## Implementation
Single Python script: `train.py`
- Load data → Preprocess → Encode → CV train → Predict → Save submission
