# Rice - 近红外光谱水稻品质分析工具包

一个专业的近红外光谱(NIR)水稻品质评估工具包，集成PyNIR库和机器学习技术，提供完整的光谱分析解决方案。

## 🎯 核心特性

- **🔬 PyNIR原生集成** - 支持PyNIR预处理方法(MSC、SNV、导数等)
- **📊 全面分析能力** - 光谱峰检测、异常值分析、聚类分析、品质预测
- **🤖 机器学习建模** - PLS回归、PLSDA分类、传统机器学习模型对比
- **📈 专业可视化** - 高质量英文标签图表，无字体显示问题
- **🧩 模块化设计** - 清晰的代码结构，可独立使用或组合使用
- **⚡ 一键分析** - 提供即用即显的完整分析流程

## 📁 项目结构

```
rice/
├── README.md                          # 英文说明文档
├── README_CN.md                       # 中文说明文档(本文件)
├── OVERVIEW.md                        # 项目概览
├── data/                              # 输入数据
│   ├── rice_spectra.xlsx             # NIR光谱数据(主要数据文件)
│   └── [其他原始数据文件]
├── src/                               # 核心源代码
│   ├── preprocessing.py               # 数据预处理模块
│   ├── analysis.py                    # 光谱分析模块
│   └── modeling.py                    # 品质预测建模模块
├── examples/                          # 使用示例
│   ├── quick_start.py                 # 快速开始演示
│   └── run_full_analysis.py           # 完整分析流程
├── outputs/                           # 统一输出目录
│   ├── figures/                       # 生成的图表
│   ├── models/                        # 模型结果文件
│   ├── processed_data/                # 预处理后的数据
│   └── reports/                       # 分析报告
├── scripts/                           # 原始脚本(向后兼容)
└── tools/                             # 辅助工具
```

## 🚀 快速开始

### 环境要求

```bash
# Python 3.7+
# 必需的Python包:
pip install pynir scikit-learn pandas numpy matplotlib seaborn scipy pathlib
```

### 使用方式

#### 1. 快速演示 (2分钟体验)
```bash
cd /home/daniel/NGW/data/rice
python3 examples/quick_start.py
```
**输出内容:**
- 数据预处理演示(MSC、SNV)
- 光谱分析演示(峰检测、异常值检测、聚类)
- 功能验证确认

#### 2. 完整分析流程 (5-10分钟)
```bash
python3 examples/run_full_analysis.py
```
**完整流程包括:**
- 🔄 **步骤1**: 数据预处理 → 生成对比图表和处理后数据
- 📊 **步骤2**: 光谱分析 → 峰检测、异常值分析、聚类分析
- 🎯 **步骤3**: 品质预测建模 → 回归和分类模型构建及验证

#### 3. 独立模块使用
```bash
# 仅进行数据预处理
python3 src/preprocessing.py

# 仅进行光谱分析
python3 src/analysis.py

# 仅进行品质建模
python3 src/modeling.py
```

## 📊 详细分析流程

### 第一阶段: 数据预处理 🔄

**功能描述:**
- 从Excel文件加载NIR光谱数据
- 应用PyNIR预处理方法消除光谱干扰
- 生成预处理效果对比图表
- 导出处理后的光谱数据

**PyNIR预处理方法:**
- **MSC (多元散射校正)**: 消除散射效应，提高光谱一致性
- **SNV (标准正态变量变换)**: 标准化每个样本，消除基线漂移
- **一阶导数**: 突出光谱特征，消除基线偏移
- **二阶导数**: 进一步增强光谱分辨率
- **Savitzky-Golay滤波**: 平滑光谱噪声
- **中心化处理**: 数据标准化预处理

**输出文件:**
- `outputs/figures/preprocessing_comparison.png` - 预处理方法对比图
- `outputs/processed_data/` - 各种预处理后的CSV数据文件

### 第二阶段: 光谱分析 📈

**功能描述:**
- 光谱特征峰自动检测和标注
- 多方法异常值检测(PyNIR PLS + 传统方法)
- 聚类分析识别样本模式
- 生成综合分析报告

**分析方法:**

1. **峰检测算法**
   - 基于scipy.signal.find_peaks
   - 自动识别特征峰位置和强度
   - 标注重要的光谱吸收峰

2. **异常值检测**
   - **PyNIR PLS方法**: 基于PLS模型的异常值检测
   - **Isolation Forest**: 基于孤立森林的异常值检测  
   - **马氏距离法**: 基于统计距离的异常值检测
   - 多方法结果对比和一致性分析

3. **聚类分析**
   - **K-means聚类**: 基于欧几里得距离的聚类
   - **DBSCAN聚类**: 基于密度的聚类算法
   - 轮廓系数评估聚类效果
   - 聚类结果可视化

**输出文件:**
- `outputs/reports/analysis_summary.png` - 综合分析总览图
- `outputs/reports/analysis_results.csv` - 分析结果统计表

### 第三阶段: 品质预测建模 🎯

**功能描述:**
- 基于光谱数据预测水稻品质参数
- PyNIR PLS建模与传统机器学习模型对比
- 模型性能评估和交叉验证
- 生成模型结果报告

**建模方法:**

1. **回归建模** (连续型品质指标)
   - **PyNIR PLS回归**: 偏最小二乘回归，专门用于光谱数据
   - **随机森林回归**: 集成学习方法，作为对比基准
   - **预测目标**: 蛋白质含量、水分含量、直链淀粉含量等

2. **分类建模** (品质等级分类)
   - **PyNIR PLSDA**: 偏最小二乘判别分析
   - **随机森林分类**: 传统机器学习分类方法
   - **品质等级**: 优质、标准、一般三个等级

**模型评估指标:**
- **回归模型**: R²决定系数、RMSE均方根误差、交叉验证得分
- **分类模型**: 准确率、精确率、召回率、F1分数

**输出文件:**
- `outputs/models/model_performance.png` - 模型性能对比图
- `outputs/models/model_summary.csv` - 详细的模型评估结果

## 📈 输出文件详细说明

### 图表文件 (`outputs/figures/`)

| 文件名 | 描述 | 内容 |
|--------|------|------|
| `preprocessing_comparison.png` | 预处理方法对比 | 4个子图展示不同预处理效果 |
| `analysis_summary.png` | 光谱分析总览 | 峰检测、PCA、异常值、统计信息 |
| `model_performance.png` | 模型性能对比 | 回归和分类模型性能柱状图 |

### 数据文件 (`outputs/processed_data/`)

| 文件名 | 描述 | 格式 |
|--------|------|------|
| `original_data.csv` | 原始光谱数据 | 样本×波长矩阵 |
| `msc_data.csv` | MSC校正后数据 | 样本×波长矩阵 |
| `snv_data.csv` | SNV变换后数据 | 样本×波长矩阵 |
| `first_derivative_data.csv` | 一阶导数数据 | 样本×波长矩阵 |
| `second_derivative_data.csv` | 二阶导数数据 | 样本×波长矩阵 |

### 模型结果 (`outputs/models/`)

| 文件名 | 描述 | 内容 |
|--------|------|------|
| `model_summary.csv` | 模型性能汇总 | 所有模型的评估指标 |
| `model_performance.png` | 性能对比图表 | 直观的性能比较 |

### 分析报告 (`outputs/reports/`)

| 文件名 | 描述 | 内容 |
|--------|------|------|
| `analysis_results.csv` | 分析结果统计 | 峰数量、异常值数量等 |
| `analysis_summary.png` | 综合分析图表 | 4个子图的综合展示 |

## 🔧 技术实现细节

### PyNIR集成特性

1. **预处理方法调用**
   ```python
   # 使用PyNIR的sklearn风格API
   msc_obj = Preprocessing.msc()
   msc_data = msc_obj.fit_transform(raw_data)
   
   snv_obj = Preprocessing.snv()  
   snv_data = snv_obj.fit_transform(raw_data)
   ```

2. **异常值检测**
   ```python
   # PyNIR PLS异常值检测
   outlier_result = OutlierDetection.outlierDetection_PLS(data, target)
   ```

3. **建模方法**
   ```python
   # PLS回归
   pls_obj = Calibration.pls(n_components=10)
   pls_obj.fit(X_train, y_train)
   y_pred = pls_obj.predict(X_test)
   
   # PLSDA分类
   plsda_obj = Calibration.plsda(n_components=5)
   plsda_obj.fit(X_train, y_train)
   y_pred = plsda_obj.predict(X_test)
   ```

### 传统机器学习方法

- **聚类分析**: K-means、DBSCAN与轮廓系数评估
- **异常值检测**: Isolation Forest、马氏距离法
- **回归建模**: 随机森林、支持向量机、神经网络
- **分类建模**: 随机森林、SVM、多层感知机

### 可视化特性

- **英文标签设计**: 所有图表使用英文标签，避免中文字体显示问题
- **高质量输出**: 300 DPI分辨率，适合学术论文和报告
- **多子图布局**: 科学合理的图表布局，信息密度高
- **专业配色**: 使用科学可视化标准配色方案

## 💡 使用技巧和建议

### 数据准备

1. **文件格式要求**
   - Excel格式(.xlsx)，第一列为波长，后续列为光谱数据
   - 确保数据为数值型，避免文本和空值
   - 推荐样本数量: 50-500个样本

2. **光谱质量检查**
   - 运行快速演示检查数据加载是否正常
   - 观察预处理效果选择合适的预处理方法
   - 注意异常值检测结果，考虑是否需要剔除异常样本

### 参数调优

1. **预处理参数**
   ```python
   # 可以在src/preprocessing.py中调整预处理方法组合
   methods = ['msc', 'snv', 'first_deriv']  # 根据需要选择
   ```

2. **建模参数**
   ```python
   # PLS成分数可根据数据特点调整
   pls_obj = Calibration.pls(n_components=10)  # 通常5-20之间
   ```

3. **异常值检测敏感度**
   ```python
   # Isolation Forest污染率可调整
   iso_forest = IsolationForest(contamination=0.1)  # 0.05-0.15
   ```

### 结果解读

1. **预处理效果评估**
   - 观察预处理前后光谱的平滑度和基线
   - MSC和SNV主要消除散射，导数突出细节特征
   - 选择最适合后续分析的预处理方法

2. **异常值分析**
   - 多种方法检测到的异常值取交集更可靠
   - 结合PCA散点图观察异常值的空间分布
   - 考虑异常值的实际意义，不要盲目剔除

3. **模型性能评估**
   - R² > 0.8 表示回归模型效果良好
   - 准确率 > 0.85 表示分类模型性能满意
   - 关注PyNIR模型与传统方法的对比结果

## 🛠️ 故障排除

### 常见问题及解决方案

1. **数据加载失败**
   ```
   问题: 无法读取Excel文件或数据格式错误
   解决: 检查文件路径和格式，确保第一列为波长数据
   ```

2. **PyNIR方法报错**
   ```
   问题: PyNIR相关函数调用失败
   解决: 检查PyNIR包安装，确保使用sklearn风格的API调用方式
   ```

3. **内存不足**
   ```
   问题: 大数据集处理时内存溢出
   解决: 减少样本数量或降低光谱分辨率，使用数据分批处理
   ```

4. **图表显示问题**
   ```
   问题: 生成的图表无法正常显示
   解决: 确保matplotlib后端配置正确，所有图表已使用英文标签
   ```

### 性能优化建议

1. **数据预处理优化**
   - 仅选择必要的预处理方法
   - 对大数据集考虑降采样
   - 使用并行处理加速计算

2. **建模效率提升**
   - 进行特征选择减少变量数量
   - 调整交叉验证折数
   - 使用网格搜索优化超参数

## 📚 技术特点总结

### 创新特性
- ✅ **PyNIR原生集成**: 业界首个完整集成PyNIR的水稻品质分析工具包
- ✅ **模块化架构**: 清晰的三层架构设计，便于维护和扩展
- ✅ **多方法融合**: 结合PyNIR专业方法与传统机器学习的优势
- ✅ **即用即显**: 提供完整的示例和一键运行功能
- ✅ **专业可视化**: 高质量英文标签图表，适合学术发表

### 应用价值
- 🎯 **科研应用**: 支持NIR光谱学研究和方法开发
- 🏭 **工业应用**: 可用于粮食品质检测和质量控制
- 📖 **教学应用**: 优秀的光谱分析和机器学习教学案例
- 🔬 **技术验证**: PyNIR方法与传统方法的性能对比平台

### 技术优势
- 🚀 **高效性**: 优化的算法实现，快速处理大规模光谱数据
- 🎖️ **专业性**: 基于PyNIR的专业光谱分析方法
- 🔧 **易用性**: 简洁的接口设计，降低使用门槛
- 📊 **完整性**: 从数据预处理到模型验证的完整工作流
- 🌍 **兼容性**: 跨平台支持，标准Python环境即可运行

## 🎉 开始使用

立即体验Rice工具包的强大功能：

```bash
# 进入项目目录
cd /home/daniel/NGW/data/rice

# 运行快速演示
python3 examples/quick_start.py

# 查看完整分析结果
python3 examples/run_full_analysis.py

# 检查输出结果
ls outputs/
```

所有生成的图表、数据和报告都保存在 `outputs/` 目录中，随时可以查看和使用！

---

**Rice工具包** - 专业的NIR水稻品质分析解决方案，让光谱分析变得简单高效！ 🌾✨

*如有问题或建议，欢迎查看项目文档或提交反馈。*