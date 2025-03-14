# Define the synonym dictionary and key features set
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
    'C反应蛋白': 'C-反应蛋白',
    '肌酸激酶同工酶(质量法)': '肌酸激酶同工酶',
    '肌酸激酶': '肌酸激酶(CK)',
    '真菌': '酵母菌',
    '尿沉渣上皮细胞': '尿上皮细胞计数',
    '尿沉渣白细胞': '尿沉渣白细胞计数',
    '尿沉渣红细胞': '尿沉渣红细胞计数',
    '病理管型': '病理性管型',
    '白细胞': '镜检白细胞',
    '管型': '镜检管型',
    '红细胞': '镜检红细胞',
    '总蛋白': '总蛋白(TP)',
    '白球比例': '白球比例(A:G)',
    'eGFR-EPIcysc': '胱抑素C(CysC)',
    '尿白蛋白肌酐比': 'ACR比值',
    '尿素氮': '尿素氮(BUN)',
    '葡萄糖(GLU)': '空腹血糖(GLU)',
    '血小板最大聚集率': '血小板最大聚集率(AA)',
    '血小板粘附率': '血小板粘附率(AA)',
    '异常红细胞形态检测': '异常红细胞形态检测(AA)',
    '异常血小板形态检测': '异常血小板形态检测(AA)',
    '血小板计数初始值': '血小板计数初始值(AA)',
    '红细胞平均体积初始': '红细胞计数初始值(AA)',
    '红细胞计数初始值': '红细胞计数初始值(ADP)',
}

key_feature = {
    '维生素B12', '叶酸', '肝素残留', 'IFN-α', '白介素-6(IL-6)', 'IL-12P70', 'IL-1β',
    'IL-5', 'IL-8', '总前列腺特异性抗原', '游离PSA/总PSA', '游离前列腺特异性抗原',
    'C反应蛋白', '血清淀粉样蛋白A', '肌酸激酶同工酶(质量法)', '肌酸激酶', 'NT-proBNP',
    '中荧光网织红细胞', '低荧光网织红细胞', '高荧光网织红细胞', '未成熟RET指数', '网织红细胞比率',
    '网织红细胞计数', '真菌', '尿沉渣上皮细胞', '尿沉渣白细胞', '尿沉渣红细胞', '病理管型',
    '白细胞', '管型', '红细胞', '异常红细胞', '正常红细胞', '总蛋白', '白球比例',
    'eGFR-EPIcysc', 'eGFR-EPIcr+cysc', '尿白蛋白肌酐比', '尿素氮', '葡萄糖(GLU)',
    '空腹C肽(CPE)', '空腹胰岛素', '糖化白蛋白%(GA)', '不饱和铁结合力(UIBC)', '总铁结合力(TIBC)',
    '血清铁(Fe)', '小而密低密度脂蛋白胆固醇', '载脂蛋白A1(APOA1)', '促甲状腺素受体抗体',
    '抗甲状腺过氧化物酶抗体', '甲状旁腺素', '甲状腺球蛋白抗体', '总T3', '总T4',
    '免疫球蛋白A', '免疫球蛋白G', '免疫球蛋白IgG4', '免疫球蛋白M', '补体C1Q', '补体C3',
    '补体C4', '轻链kap', '轻链lam', '血Kap:Lam', '血小板最大聚集率', '血小板粘附率',
    '异常红细胞形态检测', '异常血小板形态检测', '血小板计数初始值', '红细胞平均体积初始', '红细胞计数初始值'
}

# Standardize key_feature names
standardized_features = set(synonym_dict.get(feature, feature) for feature in key_feature)

# Find common features between synonym_dict values and standardized key_feature
common_features = standardized_features.intersection(set(synonym_dict.values()))

import ace_tools as tools; tools.display_dataframe_to_user(name="Common Standardized Features", dataframe=pd.DataFrame(list(common_features), columns=['Standardized Feature']))

