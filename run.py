# ============================================================
# Predicting Student Health Risk
# 学生健康风险预测 — 三分类（fit / at-risk / unhealthy）
# 
# 策略总结:
#   13个原始特征（7数值 + 6类别），仅做序数编码 + 中位数/众数填充
#   序数编码在填充之前进行，保留 NaN 供 CatBoost 原生利用
#   无衍生特征、无分箱、无独热编码
# ============================================================

import pandas as pd
import numpy as np
import logging
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# 配置日志系统
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 全局随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
DATA_DIR = Path('input')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1: 加载原始数据
# ============================================================
logger.info("=" * 60)
logger.info("STEP 1: LOADING DATA")
logger.info("=" * 60)

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

logger.info(f"Train shape: {train.shape}")   # 690088 行 × 15 列（含 id + health_condition）
logger.info(f"Test shape: {test.shape}")     # 295753 行 × 14 列（无 health_condition）
logger.info(f"Train columns: {list(train.columns)}")
logger.info(f"Test columns: {list(test.columns)}")

# ============================================================
# STEP 2: 探索性数据分析（EDA）
# 目的：了解数据分布、缺失情况、特征与目标的关系
# ============================================================
logger.info("=" * 60)
logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
logger.info("=" * 60)

# ---- 2a. 目标变量分布 ----
# 严重不平衡: at-risk 85.9%, unhealthy 8.4%, fit 5.8%
logger.info(f"\nTarget distribution:\n{train['health_condition'].value_counts()}")
logger.info(f"\nTarget distribution (%):\n{train['health_condition'].value_counts(normalize=True).mul(100).round(2)}")

# ---- 2b. 缺失值分析 ----
# EDA 结论：缺失率在各类间一致（MCAR），无需特殊处理
logger.info("\n--- Missing Values ---")
for name, df in [('Train', train), ('Test', test)]:
    miss = df.isnull().sum()
    miss_df = pd.DataFrame({'count': miss, 'pct': df.isnull().mean().mul(100).round(2)})
    miss_df = miss_df[miss_df['count'] > 0].sort_values('count', ascending=False)
    logger.info(f"{name} missing:\n{miss_df}")

# ---- 2c. 定义数值列和类别列 ----
# 7个数值特征，6个类别特征
num_cols = ['sleep_duration', 'heart_rate', 'bmi', 'calorie_expenditure',
            'step_count', 'exercise_duration', 'water_intake']
cat_cols = ['diet_type', 'stress_level', 'sleep_quality',
            'physical_activity_level', 'smoking_alcohol', 'gender']

# CSV 中所有列均为 object 类型，需将数值列转为 float
for col in num_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

logger.info(f"\nNumerical stats:\n{train[num_cols].describe().round(2)}")

# ---- 2d. 不同健康状态下的数值特征均值 ----
# 关键发现: sleep_duration 区分度最大，fit(7.95h) > at-risk(7.09h) > unhealthy(5.37h)
target_mean = train.groupby('health_condition')[num_cols].mean().round(2)
logger.info(f"\nTarget vs Numerical (mean):\n{target_mean}")

# ---- 2e. 类别特征分布 ----
logger.info("\n--- Categorical Feature Distributions ---")
for col in cat_cols:
    vc = train[col].value_counts(dropna=False)
    logger.info(f"\n{col}:\n{vc}")

# ---- 2f. 类别特征与目标的交叉分析 ----
# 关键发现: stress_level 几乎确定健康状态，high→97.5% unhealthy
logger.info("\n--- Target vs Categorical (% by row) ---")
for col in cat_cols:
    ct = pd.crosstab(train[col], train['health_condition'], normalize='index').mul(100).round(1)
    logger.info(f"\n{col}:\n{ct}")

# ============================================================
# STEP 3: 特征关系分析
# 目的：量化原始特征之间的关联强度，作为特征工程的依据
# ============================================================
logger.info("=" * 60)
logger.info("STEP 3: FEATURE RELATIONSHIP ANALYSIS")
logger.info("=" * 60)

# ---- 3a. 数值-数值相关性（Pearson r）----
# EDA 发现大部分数值特征间近乎独立（corr ~ 0）
# 仅少数存在中等相关性：
#   calorie ↔ step: r=0.400
#   step ↔ exercise_duration: r=0.438
#   calorie ↔ exercise_duration: r=0.394
#   bmi ↔ calorie: r=0.118
logger.info("Numeric-Numeric correlations (|r| > 0.1):")
num_corrs = train[num_cols].corr()
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        r = num_corrs.iloc[i, j]
        if abs(r) > 0.1:
            logger.info(f"  {num_cols[i]} ↔ {num_cols[j]}: r = {r:.3f}")

# ---- 3b. 类别-数值关系（分组均值）----
# 关键发现：
#   stress_level → sleep_duration: 高压力组睡眠更少
#   physical_activity_level → step_count/exercise_duration: 活跃组运动更多
#   sleep_quality → sleep_duration: 高质量睡眠组时长更长
logger.info("\nCategorical → Numeric relationships (group means differ):")
for cat in cat_cols:
    for num in num_cols:
        means = train.groupby(cat)[num].mean()
        if means.std() > 0.1 * train[num].std():
            logger.info(f"  {cat} → {num}: {means.round(2).to_dict()}")

# ============================================================
# STEP 4: 特征工程
# 流程：序数编码（保留NaN）→ 填充原始列 → 丢弃原始类别字符串
# 
# 关键设计决策:
#   1. 序数编码在填充之前执行 → 序数编码列保留原始NaN
#   2. CatBoost 原生支持 NaN → 缺失本身是有信息量的模式
#   3. 实验证明"先编码再填充"（0.9489）远优于"先填充再编码"（0.9084）
#   4. 不添加衍生特征、不分箱、不独热编码——实验证明这些对性能贡献 < 0.04%
# ============================================================
logger.info("")
logger.info("=" * 60)
logger.info("STEP 4: FEATURE ENGINEERING")
logger.info("=" * 60)

def engineer_features(df, is_train=True):
    """
    特征工程主函数。
    
    参数:
        df: 原始 DataFrame（含 id、原始数值列、原始类别列）
        is_train: 是否为训练集（控制填充值的存储与复用）
    
    返回:
        处理后的 DataFrame（13列: 7数值 + 6序数编码）
    """
    df = df.copy()

    # ================================================================
    # 4a. 序数编码（保留 NaN 供 CatBoost 原生利用）
    #
    # 为什么要先编码再填充？
    #   如果先填充再编码，序数编码列全为有效数值，CatBoost 看不到任何缺失信号
    #   实验对比：
    #     先编码→再填充：0.9489
    #     先填充→再编码：0.9084
    #   差距 4 个百分点，说明 CatBoost 的 NaN 原生处理对序数编码非常有效
    #
    # 映射规则：
    #   每个有序类别映射为 0/1/2，保留顺序关系
    #   例如 stress_level: low < medium < high → 0 < 1 < 2
    #   CatBoost 可以基于序数值做 "feature >= 1" 这样的高效切分
    # ================================================================
    ord_maps = {
        'stress_level':           {'low': 0, 'medium': 1, 'high': 2},
        'sleep_quality':          {'poor': 0, 'average': 1, 'good': 2},
        'physical_activity_level': {'sedentary': 0, 'moderate': 1, 'active': 2},
        'smoking_alcohol':        {'no': 0, 'occasional': 1, 'yes': 2},
        'diet_type':              {'veg': 0, 'balanced': 1, 'non-veg': 2},
        'gender':                 {'female': 0, 'male': 1, 'other': 2},
    }
    for col, mapping in ord_maps.items():
        # map 会将无法映射的 NaN 保留为 NaN——这正是我们需要的
        df[f'{col}_ord'] = df[col].map(mapping)
    logger.info("  [4a] Ordinal encoding: 6 categorical columns → 0/1/2 (NaN preserved for CatBoost)")

    # ================================================================
    # 4b. 缺失值填充（仅填充原始列，不影响序数编码列）
    #
    # 策略：
    #   数值列 → 中位数（对异常值鲁棒）
    #   类别列 → 众数（最频繁出现的类别）
    #
    # 训练/测试一致性：
    #   is_train=True:  从训练集计算填充值并存储为函数属性
    #   is_train=False: 从函数属性读取训练集的填充值
    #   保证测试集使用与训练集完全相同的填充值，无数据泄露
    # ================================================================
    for col in num_cols:
        if is_train:
            fill_val = df[col].median()
            setattr(engineer_features, f'{col}_fill', fill_val)
        else:
            fill_val = getattr(engineer_features, f'{col}_fill', df[col].median())
        df[col] = df[col].fillna(fill_val)

    for col in cat_cols:
        if is_train:
            fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
            setattr(engineer_features, f'{col}_fill', fill_val)
        else:
            fill_val = getattr(engineer_features, f'{col}_fill', 'unknown')
        df[col] = df[col].fillna(fill_val)

    logger.info("  [4b] Imputation: median for numeric, mode for categorical")

    # ================================================================
    # 4c. 丢弃原始类别字符串
    #
    # 原始类别列（如 stress_level 的 'low'/'medium'/'high'）已被
    # 序数编码列（stress_level_ord 的 0/1/2）替代
    # 保留序数编码列，丢弃原始字符串列
    # ================================================================
    for col in cat_cols:
        df = df.drop(columns=[col])

    # 丢弃无预测意义的 id 列
    df = df.drop(columns=['id'], errors='ignore')

    # 最终特征数: 7 数值 + 6 序数编码 = 13
    n_feat = len(df.columns) - (0 if 'health_condition' not in df.columns else 1)
    logger.info(f"  [4c] Final features: {n_feat} (7 numeric + 6 ordinal)")

    return df

# 分别处理训练集和测试集
train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# ============================================================
# 训练/测试集对齐与最终数据准备
# ============================================================

# 分离目标变量（health_condition 仅存在于训练集）
y = train_fe['health_condition'].values
train_fe = train_fe.drop(columns=['health_condition'])
test_fe = test_fe.drop(columns=['health_condition'], errors='ignore')

# 对齐列顺序和集合（确保 OHE 列一致，本项目中无 OHE 但仍保持安全处理）
train_fe, test_fe = train_fe.align(test_fe, join='inner', axis=1)

logger.info(f"\nFinal features — Train: {train_fe.shape}, Test: {test_fe.shape}")
logger.info(f"Feature list:\n{list(train_fe.columns)}")

# 转换为 numpy 数组供 CatBoost 使用
X = train_fe.values
X_test = test_fe.values
ids_test = test['id'].values

logger.info(f"\nX shape: {X.shape}")
logger.info(f"X_test shape: {X_test.shape}")
logger.info(f"Target classes: {np.unique(y)}")

# ============================================================
# STEP 5: CatBoost 5-折分层交叉验证
#
# 模型选择理由:
#   CatBoost 原生支持类别特征和缺失值，无需手动处理
#   对称树结构 vs 非对称树的讨论在此场景中无显著差异
#
# 参数说明:
#   loss_function=MultiClass    — 三分类任务
#   auto_class_weights=Balanced — 处理 85.9%/8.4%/5.8% 严重不平衡
#   early_stopping_rounds=50     — 验证集 50 轮无改善则停止
#   use_best_model=True          — 回滚到验证集最佳迭代
#   depth=6, lr=auto, iterations=1000 — CatBoost 默认值
#
# 评估指标: Balanced Accuracy（Macro Recall）
#   对不平衡分类更公平，各类别召回率的算术平均
# ============================================================
logger.info("=" * 60)
logger.info("STEP 5: CATBOOST 5-FOLD CROSS-VALIDATION")
logger.info("=" * 60)

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 标签编码: health_condition → 0/1/2
le = LabelEncoder()
y_enc = le.fit_transform(y)
logger.info(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 分层 5 折交叉验证（每折保持类别比例与全集一致）
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# 初始化结果容器
cv_scores = []          # 每折的 Balanced Accuracy
oof_preds = np.zeros(len(y_enc))   # Out-of-Fold 预测
test_preds = np.zeros((len(X_test), len(le.classes_)))  # 测试集预测（各折平均）

# CatBoost 参数
cb_params = {
    'loss_function': 'MultiClass',       # 多分类对数损失
    'auto_class_weights': 'Balanced',     # 类别权重自动平衡
    'random_seed': SEED,                  # 随机种子
    'task_type': 'GPU',                   # GPU 加速
    'thread_count': -1,                   # 使用所有 GPU 线程
    'verbose': 50,                        # 每 50 轮输出一次日志
    'early_stopping_rounds': 50,          # 早停轮数
    # depth=6, lr=auto, iterations=1000 使用 CatBoost 默认值
}

logger.info(f"CatBoost params: {cb_params}")

# ---- 5 折交叉验证循环 ----
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_enc)):
    logger.info(f"\n{'='*40}")
    logger.info(f"FOLD {fold + 1}/{n_splits}")
    logger.info(f"{'='*40}")

    # 划分训练集和验证集
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

    logger.info(f"Train size: {X_tr.shape[0]}, Val size: {X_val.shape[0]}")
    logger.info(f"Train class dist: {np.bincount(y_tr)}")
    logger.info(f"Val class dist:   {np.bincount(y_val)}")

    # 初始化并训练模型
    model = CatBoostClassifier(**cb_params)

    start_time = datetime.now()
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,       # 自动回滚到最佳迭代
        logging_level='Verbose',   # 输出训练日志
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'
    logger.info(f"Training time: {elapsed:.1f}s (best iteration: {best_iter})")

    # 验证集预测
    val_prob = model.predict_proba(X_val)
    val_pred = val_prob.argmax(axis=1)
    oof_preds[val_idx] = val_pred

    # 计算并记录本折 Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_val, val_pred)
    cv_scores.append(bal_acc)
    logger.info(f"Fold {fold + 1} Balanced Accuracy: {bal_acc:.6f}")

    # 测试集预测（各折预测概率平均，提高稳定性）
    test_prob = model.predict_proba(X_test)
    test_preds += test_prob / n_splits

    # 输出本折 Top 10 特征重要性
    imp = pd.DataFrame({
        'feature': train_fe.columns,
        f'importance_fold{fold+1}': model.feature_importances_
    }).sort_values(f'importance_fold{fold+1}', ascending=False)
    logger.info(f"\nTop 10 features fold {fold+1}:\n{imp.head(10).to_string(index=False)}")

# ============================================================
# 交叉验证结果汇总
# ============================================================
logger.info(f"\n{'='*60}")
logger.info(f"CV RESULTS — {n_splits}-FOLD STRATIFIED")
logger.info(f"{'='*60}")
logger.info(f"Per-fold balanced accuracies: {[f'{s:.6f}' for s in cv_scores]}")
logger.info(f"Mean balanced accuracy: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

# Out-of-Fold 综合评估
oof_bal_acc = balanced_accuracy_score(y_enc, oof_preds.astype(int))
logger.info(f"OOF Balanced Accuracy: {oof_bal_acc:.6f}")

# 混淆矩阵 + 详细分类报告
cm = confusion_matrix(y_enc, oof_preds.astype(int))
logger.info(f"\nOOF Confusion Matrix:\n{cm}")
logger.info(f"\nOOF Classification Report:\n{classification_report(y_enc, oof_preds.astype(int), target_names=le.classes_)}")

# ============================================================
# STEP 6: 生成提交文件
# ============================================================
logger.info("=" * 60)
logger.info("STEP 6: GENERATING SUBMISSION")
logger.info("=" * 60)

# 将测试集预测概率转换为类别标签
test_labels = le.inverse_transform(test_preds.argmax(axis=1))

# 构建提交 DataFrame
sub = pd.DataFrame({'id': ids_test, 'health_condition': test_labels})
sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

logger.info(f"\nSubmission saved to {OUTPUT_DIR / 'submission.csv'}")
logger.info(f"Shape: {sub.shape}")
logger.info(f"Submission head:\n{sub.head(10)}")
logger.info(f"Submission target distribution:\n{sub['health_condition'].value_counts()}")

logger.info("\n" + "=" * 60)
logger.info("DONE")
logger.info("=" * 60)
