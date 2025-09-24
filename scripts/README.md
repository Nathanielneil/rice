# 近红外光谱水稻品质分析脚本（集成PyNIR）

本目录包含基于**PyNIR原生包**和机器学习方法的水稻光谱数据处理和品质预测脚本。

## 脚本概览

### 1. `01_nir_data_preprocessing.py` - 光谱数据预处理（PyNIR集成版）
**功能：**
- **PyNIR MSC**: 多元散射校正
- **PyNIR SNV**: 标准正态变量变换  
- **PyNIR 导数**: 一阶和二阶导数计算
- **PyNIR SG滤波**: Savitzky-Golay滤波
- **PyNIR 平滑**: 平滑处理
- **PyNIR 中心化**: 数据中心化
- **备用方法**: 自定义实现作为后备

**使用方法：**
```bash
cd /home/daniel/NGW/data/daughter/scripts
python 01_nir_data_preprocessing.py
```

### 2. `02_nir_data_visualization.py` - 数据可视化
**功能：**
- 原始光谱图
- 预处理方法对比图
- 主成分分析(PCA)可视化
- 光谱相关性热图
- 统计摘要图

**使用方法：**
```bash
python 02_nir_data_visualization.py
```

### 3. `03_nir_spectral_analysis.py` - 光谱分析（PyNIR集成版）
**功能：**
- 光谱峰检测
- **PyNIR异常值检测**: 基于PLS的异常值检测
- 传统异常值检测(Isolation Forest, 马氏距离)
- 聚类分析(K-means, DBSCAN)
- 光谱特征提取(统计、形状、峰、区域、比值、导数特征)

**使用方法：**
```bash
python 03_nir_spectral_analysis.py
```

### 4. `04_rice_quality_prediction.py` - 品质预测建模（PyNIR集成版）
**功能：**
- **PyNIR PLS回归**: 蛋白质、水分、直链淀粉含量预测
- **PyNIR PLSDA**: 品质等级分类预测
- 传统机器学习模型比较(线性回归、随机森林、SVM、神经网络)
- 模型性能评估和对比分析

**使用方法：**
```bash
python 04_rice_quality_prediction.py
```

### 5. `05_pynir_integration_demo.py` - PyNIR功能完整演示
**功能：**
- **PyNIR预处理**: MSC, SNV, 导数, SG滤波, 平滑, 中心化, 小波变换
- **PyNIR异常值检测**: PLS异常值检测演示
- **PyNIR建模**: PLS回归和PLSDA分类演示
- **功能对比**: PyNIR vs 传统方法性能对比
- **完整流程**: 从数据预处理到建模的完整演示

**使用方法：**
```bash
python 05_pynir_integration_demo.py
```

## 数据要求

### 输入数据
- **光谱数据**: `/home/daniel/NGW/data/daughter/data/不同品种优质稻光谱扫描.xlsx`
  - Sheet "第一次取样 原始光谱": 第一列为波长，其余列为光谱数据
  - 其他预处理sheet: 直接的光谱矩阵数据

### 输出结果
所有结果保存在 `../results/` 目录下：
- `../results/`: 预处理数据CSV文件
- `../results/figures/`: 可视化图表
- `../results/analysis/`: 光谱分析结果
- `../results/prediction/`: 预测模型结果

## 依赖包安装

### 核心依赖
```bash
# PyNIR核心包
pip install pynir

# 基础科学计算包
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

### PyNIR包说明
- **版本**: 0.7.11
- **核心功能**: NIR光谱预处理、异常值检测、PLS建模
- **依赖**: scipy, scikit-learn, pywavelets
- **文档**: https://pypi.org/project/pynir/

## 运行顺序建议

### 快速上手（推荐）
1. **PyNIR功能演示** → `05_pynir_integration_demo.py` 🌟

### 详细分析流程
1. **数据预处理** → `01_nir_data_preprocessing.py`
2. **数据可视化** → `02_nir_data_visualization.py`  
3. **光谱分析** → `03_nir_spectral_analysis.py`
4. **品质预测** → `04_rice_quality_prediction.py`
5. **完整演示** → `05_pynir_integration_demo.py`

## 主要分析方法

### PyNIR预处理方法
- **PyNIR MSC**: 多元散射校正，消除散射效应
- **PyNIR SNV**: 标准正态变量变换，标准化样本
- **PyNIR 导数**: 一阶/二阶导数，消除基线漂移
- **PyNIR SG滤波**: Savitzky-Golay平滑滤波
- **PyNIR 平滑**: 移动窗口平滑
- **PyNIR 中心化**: 数据中心化处理
- **PyNIR CWT**: 连续小波变换

### 建模方法
- **PyNIR方法**:
  - **PLS回归**: 偏最小二乘回归
  - **PLSDA**: PLS判别分析
  - **异常值检测**: 基于PLS的异常值检测
- **传统机器学习**:
  - **回归**: Linear/Ridge/Lasso回归, 随机森林, SVR, 神经网络
  - **分类**: 随机森林, SVM, 神经网络

### 评价指标
- **回归**: R², RMSE, 交叉验证
- **分类**: 准确率, 精确率, 召回率, F1得分

## 技术特点

### PyNIR集成优势
- ✅ **原生PyNIR方法**: 使用PyNIR包的专业NIR处理方法
- ✅ **容错机制**: PyNIR失败时自动切换到自定义方法
- ✅ **方法对比**: PyNIR vs 传统方法性能对比
- ✅ **专业建模**: PLS/PLSDA专门用于光谱分析
- ✅ **完整流程**: 从预处理到建模的一体化解决方案

### 通用特点
- 支持多种NIR光谱预处理方法
- 集成机器学习建模流程
- 完整的可视化分析
- 自动化的模型评估和比较
- 结果输出和保存

## 注意事项

### PyNIR相关
1. **PyNIR安装**: 确保正确安装pynir包及其依赖
2. **兼容性**: PyNIR方法失败时会自动使用备用方法
3. **参数优化**: PyNIR方法的参数已预设优化值

### 通用注意事项
1. 确保数据文件路径正确
2. 如果没有实际品质数据，脚本会自动生成模拟数据用于演示
3. 图表需要中文字体支持，如遇显示问题请安装相应字体
4. 建议先运行`05_pynir_integration_demo.py`了解PyNIR功能
5. 建议按顺序运行脚本以获得完整的分析结果

## PyNIR vs 传统方法对比

| 功能 | PyNIR方法 | 传统方法 | 优势 |
|------|-----------|----------|---------|
| MSC校正 | `Preprocessing.msc()` | 自定义实现 | 专业优化，性能更好 |
| SNV变换 | `Preprocessing.snv()` | 自定义实现 | 标准化实现，结果稳定 |
| 导数计算 | `Preprocessing.derivate()` | `numpy.gradient()` | 专门针对光谱优化 |
| 异常值检测 | `OutlierDetection.outlierDetection_PLS()` | Isolation Forest | 基于光谱特性设计 |
| 回归建模 | `Calibration.pls()` | sklearn回归器 | 专门用于光谱回归 |
| 分类建模 | `Calibration.plsda()` | sklearn分类器 | 光谱分类专用算法 |

## 扩展功能

可根据实际需求扩展：
- 集成更多PyNIR功能（校准转移等）
- 添加深度学习模型对比
- 实现在线预测功能
- 添加模型解释和特征重要性分析
- 多仪器校准转移