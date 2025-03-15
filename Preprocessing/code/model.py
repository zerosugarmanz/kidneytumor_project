import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier


# 读取预处理后的数据
file_path = "preprocessed_data_fixed.csv"
print("Loading data...")
df = pd.read_csv(file_path)
print("Data loaded successfully.")

# 确保数据类型正确
df = df.apply(pd.to_numeric, errors='coerce')

# 分离特征和目标变量
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# XGBoost 类别（确保类别从 0 开始）
y = y - y.min()
print("Updated class labels:", np.unique(y)) 

print("Data preprocessing completed.")

# 划分训练集和测试集 (80% 训练, 20% 测试)
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split completed.")

# 标准化特征数据
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling completed.")

# 定义模型
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),

    "Neural Network (MLP)": MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 层神经元
    learning_rate_init=0.01,  
    max_iter=1000, 
    early_stopping=True,  # 如果 10 轮没有提升，就停止训练
    random_state=42
)

}

# try hybrid models
base_models = [('lr', models['Logistic Regression']), ('random forest', models['Random Forest'])]
meta_model = models['Logistic Regression']
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
models['Stacking Model'] = stacking_model
# try voting classifier
VotingClassifier(estimators=base_models, voting='soft')


# 训练并评估模型
results = []
print("Training models...")
for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    end_time = time.time()

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else np.nan
    training_time = end_time - start_time

    # 存储结果
    results.append([name, accuracy, f1, auc, training_time])
    print(f"{name} training completed. Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, AUC-ROC: {auc:.4f}, Time: {training_time:.2f}s")

# 转换为 DataFrame 进行可视化
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-score", "AUC-ROC", "Training Time"])
print("Model training completed. Here are the results:")
print(results_df)

# 绘制模型表现比较图
plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="F1-score", data=results_df)
plt.title("Model F1-score Comparison")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="AUC-ROC", data=results_df)
plt.title("Model AUC-ROC Comparison")
plt.ylabel("AUC-ROC Score")
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="Training Time", data=results_df)
plt.title("Model Training Time Comparison")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=20)
plt.show()
