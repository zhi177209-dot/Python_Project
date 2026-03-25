# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 读取数据
df = pd.read_csv('Financial Distress.csv')

# =========================
# 0. 数据基本信息分析
# =========================

print('数据维度:', df.shape)
print('\n数据类型:\n', df.dtypes)

print('\n缺失值统计:\n', df.isnull().sum()[df.isnull().sum() > 0])

print('\n重复值数量:', df.duplicated().sum())

print('\n数据描述性统计:\n', df.describe())
# 保存数据描述性统计为CSV文件
description = df.describe()
description.to_csv('data_description.csv', encoding='utf-8')
# 样本时序分布（时间列分析）
if 'Time' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Time', data=df)
    plt.title('Sample Distribution Over Time')
    plt.xlabel('Time Period')
    plt.ylabel('Sample Count')
    plt.savefig('sample_distribution_over_time.png')
    plt.show()

# =========================
# 1. 数据预处理
# =========================

# 去重
df = df.drop_duplicates()

# 缺失值处理（均值填补）
imputer = SimpleImputer(strategy='mean')
df.iloc[:, 3:] = imputer.fit_transform(df.iloc[:, 3:])

# 处理 x80 类别特征
if df['x80'].dtype == 'object':
    le = LabelEncoder()
    df['x80'] = le.fit_transform(df['x80'])

# 创建标签列：Financial Distress <= -0.5 为 1，其他为 0
df['Target'] = df['Financial Distress'].apply(lambda x: 1 if x <= -0.5 else 0)

# =========================
# 2. 数据分析与可视化
# =========================

# 类别分布
plt.figure(figsize=(6, 4))
sns.countplot(x='Target', data=df)
plt.title('Financial Distress Distribution')
plt.xlabel('Financial Distress (0=Healthy, 1=Distressed)')
plt.ylabel('Count')
plt.savefig('financial_distress_distribution.png')
plt.show()

# 计算相关系数矩阵
corr = df.iloc[:, 3:-1].copy()
corr['Target'] = df['Target']
correlation = corr.corr()

# 取与 Target 相关性绝对值最高的前 20 个特征
top_features = correlation['Target'].abs().sort_values(ascending=False).index[1:21]

# 只画这20个特征的相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation.loc[top_features, top_features], cmap='coolwarm', annot=True, center=0)
plt.title('Top 20 Feature Correlation Heatmap')
plt.savefig('top_20_feature_correlation.png')
plt.show()

# =========================
# 3. 特征选择与模型训练
# =========================

# 划分训练集与测试集
X = df.iloc[:, 3:-1]  # 特征（去掉前3列和标签列）
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 4. 处理类别不平衡（SMOTE）
# =========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print('原始数据标签分布:\n', y_train.value_counts())
print('SMOTE 采样后标签分布:\n', pd.Series(y_train_res).value_counts())

# =========================
# 5. 使用 XGBoost 训练模型
# =========================
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1, random_state=42)
model.fit(X_train_res, y_train_res)

# 预测
y_pred = model.predict(X_test)

# 模型评估
print("Classification Report:\n", classification_report(y_test, y_pred))

# 特征重要性可视化
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(20), importances[indices[:20]], align="center")
plt.xticks(range(20), X.columns[indices[:20]], rotation=90)
plt.savefig('feature_importances.png')
plt.show()
