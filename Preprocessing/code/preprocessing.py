import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings('ignore')


files = [
    "AML_output_3.0.xlsx",
    "ccRCC_output_3.0.xlsx",
    "chRCC_output_3.0.xlsx",
    "pRCC_output_3.0.xlsx",
    "RO_output_3.0.xlsx"
]


raw_dataframes = [pd.read_excel(file) for file in files]

# Standardize Column Names
synonym_dict = {
    '丙肝抗体(C)': '丙肝抗体',
    '乙肝E抗体': '乙肝e抗体(YP)',
    '乙肝E抗原': '乙肝e抗原(YP)',
    '乙肝核心抗体': '乙肝核心抗体(YP)',
    '乙肝表面抗体': '乙肝表面抗体(YP)',
    '乙肝表面抗原': '乙肝表面抗原(YP)',
    '梅毒螺旋体抗体': '梅毒确诊试验',
    '梅毒甲苯胺红不加热血清试验(TRUST)': '梅毒确诊试验',
    'Th淋巴细胞(CD3+CD4+)': 'Th淋巴细胞CD4',
    'Th淋巴细胞（CD3+CD4+）': 'Th淋巴细胞CD4',
    'Ts淋巴细胞  CD8': 'Ts淋巴细胞CD8',
    'Ts淋巴细胞(CD3+CD8+)': 'Ts淋巴细胞CD8',
    'CD8+CD38+': 'Ts淋巴细胞CD8',
    'T淋巴细胞（CD3+）': 'T淋巴细胞(CD3+)',
    'T淋巴细胞CD3': 'T淋巴细胞(CD3+)',
    'B淋巴细胞（CD3-CD19+）': 'B淋巴细胞(CD3-CD19+)',
    'B淋巴细胞CD19': 'B淋巴细胞(CD3-CD19+)',
    '淋巴细胞绝对值（CD45+）': '淋巴细胞绝对值',
    '淋巴细胞绝对值（CD45）': '淋巴细胞绝对值',
    '自然杀伤细胞(CD3-CD16+CD56+)': '自然杀伤细胞CD56+CD16',
    'CD3+HLA-DR+': 'CD3+HLA-DR+/CD3+(%)',
    'CD8+HLA-DR+': 'CD8+HLA-DR+/CD8+(%)',
    'INF-r': 'γ干扰素',
    'IFN-γ': 'γ干扰素',
    'IFN-α': 'α干扰素',
    'IL-6': '白细胞介素-6',
    '白介素-6(IL-6)': '白细胞介素-6',
    'IL-12P70': '白细胞介素-12p70',
    'IL-1β': '白细胞介素-1β',
    'IL-5': '白细胞介素-5',
    'IL-8': '白细胞介素-8',
    '细胞角蛋白19片段(CYFRA21-1)': 'CYFRA(21-1)',
    '神经元特异性烯醇化酶(NSE)': '神经烯醇化酶(NSE)',
    '糖类抗原(CA72-4)': '糖类抗原(CA724)',
    '糖类抗原(CA242)': '糖类抗原242',
    '糖类抗原(CA50)': '糖类抗原CA50',
    '鳞癌抗原(SCC)': '鳞癌抗原',
    'C反应蛋白': 'C-反应蛋白',
    '肌酸激酶同工酶(质量法)': '肌酸激酶同工酶',
    '肌酸激酶': '肌酸激酶(CK)',
    '尿糖': '尿葡萄糖',
    '尿蛋白': '尿蛋白质',
    '真菌': '酵母菌',
    '小圆上皮细胞数': '小圆上皮细胞',
    '尿沉渣上皮细胞': '尿上皮细胞计数',
    '尿沉渣白细胞': '尿沉渣白细胞计数',
    '尿沉渣红细胞': '尿沉渣红细胞计数',
    '病理管型': '病理性管型',
    '上皮细胞': '镜检上皮细胞',
    '白细胞': '镜检白细胞',
    '管型': '镜检管型',
    '颗粒管型': '镜检管型',
    '红细胞': '镜检红细胞',
    '谷氨酰转肽酶(GGT)': 'γ谷氨酰基转移酶(GGT)',
    'γ谷氨酰基转移酶': 'γ谷氨酰基转移酶(GGT)',
    '丙氨酸氨基转移酶': '丙氨酸氨基转移酶(ALT)',
    '总胆红素': '总胆红素(TBIL)',
    '总蛋白': '总蛋白(TP)',
    '白球比例': '白球比例(A:G)',
    '白蛋白': '白蛋白(ALB)',
    '碱性磷酸酶': '碱性磷酸酶(ALP)',
    '尿素': '尿素(UREA)',
    '尿酸': '尿酸(URIC)',
    '肌酐': '肌酐(CREA)',
    '血肌酐': '肌酐(CREA)',
    'eGFR-EPIcysc': '胱抑素C(CysC)',
    '尿白蛋白肌酐比': 'ACR比值',
    '尿素氮': '尿素氮(BUN)',
    '血氯': '氯(CL)',
    '氯（全血）': '氯(CL)',
    '氯(全血)': '氯(CL)',
    '钙(全血)': '钙(CA)',
    '血钠': '钠(Na)',
    '钠（全血）': '钠(Na)',
    '钠(全血)': '钠(Na)',
    '钠(NA)': '钠(Na)',
    '血钾': '钾(K)',
    '钾（全血）': '钾(K)',
    '钾(全血)': '钾(K)',
    '镁(MG)': '镁(Mg)',
    '葡萄糖': '空腹血糖(GLU)',
    '葡萄糖(GLU)': '空腹血糖(GLU)',
    '糖(全血)': '空腹血糖(GLU)',
    '标准碳酸氢氢根浓度': '标准碳酸氢根浓度',
    '小而密低密度脂蛋白胆固醇': '小而密低密度脂蛋白(sd-LDL)',
    '甘油三酯(TG)': '甘油三脂(TG)',
    '载脂蛋白A1(APOA1)': '载脂蛋白A1(APOA)',
    '血小板最大聚集率': '血小板最大聚集率(AA)',
    '血小板粘附率': '血小板粘附率(AA)',
    '异常红细胞形态检测': '异常红细胞形态检测(AA)',
    '异常血小板形态检测': '异常血小板形态检测(AA)',
    '血小板计数初始值': '血小板计数初始值(AA)',
    '红细胞平均体积初始': '红细胞计数初始值(AA)',
    '红细胞计数初始值': '红细胞计数初始值(ADP)',
}

standardized_dfs = [df.rename(columns=synonym_dict) for df in raw_dataframes]

# Remove duplicate columns
for i, df in enumerate(standardized_dfs):
    df = df.loc[:, ~df.columns.duplicated()]
    standardized_dfs[i] = df

# Get common columns across all datasets
common_columns = set(standardized_dfs[0].columns)
for df in standardized_dfs[1:]:
    common_columns = common_columns.intersection(set(df.columns))

ordered_common_columns = [col for col in standardized_dfs[0].columns if col in common_columns]

# Retain only common columns
filtered_dataframes = [df.loc[:, ordered_common_columns].copy() for df in standardized_dfs]

# Assign Outcome labels
outcome_mapping = {0: 3, 1: 1, 2: 2, 3: 2, 4: 3}
for i, df in enumerate(filtered_dataframes):
    df['Outcome'] = outcome_mapping[i]

# Merge datasets
merged_df = pd.concat(filtered_dataframes, axis=0, ignore_index=True)

# Drop columns with more than 50% missing values
threshold = 0.5
filtered_df = merged_df.loc[:, merged_df.isnull().mean() <= threshold]

# Drop unnecessary index column
if 'Unnamed: 0' in filtered_df.columns:
    filtered_df = filtered_df.drop(columns=['Unnamed: 0'])

# Identify numeric and categorical columns
numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()

# Ensure 'Outcome' is not treated as numeric
if 'Outcome' in numeric_cols:
    numeric_cols.remove('Outcome')

# Missing value imputation
numeric_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
filtered_df[numeric_cols] = numeric_imputer.fit_transform(filtered_df[numeric_cols])

# Encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    filtered_df[col] = le.fit_transform(filtered_df[col].astype(str))

# Feature scaling
scaler = StandardScaler()
filtered_df[numeric_cols] = scaler.fit_transform(filtered_df[numeric_cols])

# Ensure Outcome is integer type
filtered_df['Outcome'] = filtered_df['Outcome'].astype(int)

# Save preprocessed data
filtered_df.to_csv('preprocessed_data_fixed2.csv', index=False)


