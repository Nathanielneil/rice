#!/usr/bin/env python3
"""
基于NIR光谱的水稻品质预测建模脚本
使用PyNIR包和机器学习方法建立光谱-品质关系模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pynir import Calibration, Preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Set English font configuration to avoid display issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class RiceQualityPredictor:
    def __init__(self, spectral_data_file, quality_data_file=None):
        """
        初始化水稻品质预测器
        
        Args:
            spectral_data_file: 光谱数据文件路径
            quality_data_file: 品质数据文件路径（如果有的话）
        """
        self.spectral_data_file = spectral_data_file
        self.quality_data_file = quality_data_file
        self.spectral_data = {}
        self.quality_data = None
        self.wavelengths = None
        self.models = {}
        self.results_dir = Path("../results/prediction")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_spectral_data(self):
        """加载光谱数据"""
        print("正在加载光谱数据...")
        xl = pd.ExcelFile(self.spectral_data_file)
        
        for sheet_name in xl.sheet_names:
            try:
                df = pd.read_excel(self.spectral_data_file, sheet_name=sheet_name)
                
                if sheet_name == '第一次取样 原始光谱':
                    self.wavelengths = df.iloc[:, 0].values
                    spectra_raw = df.iloc[:, 1:]
                    
                    # 过滤非数值列
                    numeric_cols = []
                    for col in spectra_raw.columns:
                        try:
                            pd.to_numeric(spectra_raw[col], errors='raise')
                            numeric_cols.append(col)
                        except:
                            continue
                    
                    if numeric_cols:
                        spectra = spectra_raw[numeric_cols].T
                        spectra = spectra.apply(pd.to_numeric, errors='coerce')
                        spectra = spectra.dropna()
                        spectra.columns = [f'wl_{w:.2f}' for w in self.wavelengths[:len(spectra.columns)]]
                        self.spectral_data[sheet_name] = spectra
                    else:
                        continue
                else:
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        self.spectral_data[sheet_name] = numeric_df
                    else:
                        continue
                    
                print(f"已加载光谱数据 {sheet_name}: {df.shape}")
                
            except Exception as e:
                print(f"加载sheet {sheet_name} 时出错: {e}")
    
    def load_quality_data(self):
        """加载品质数据"""
        if self.quality_data_file and Path(self.quality_data_file).exists():
            try:
                self.quality_data = pd.read_csv(self.quality_data_file, encoding='utf-8-sig')
                print(f"已加载品质数据: {self.quality_data.shape}")
            except Exception as e:
                print(f"加载品质数据时出错: {e}")
        else:
            print("未提供品质数据文件，将使用模拟数据进行演示")
            self.generate_simulated_quality_data()
    
    def generate_simulated_quality_data(self):
        """生成模拟品质数据用于演示"""
        print("生成模拟品质数据...")
        
        # 获取样本数量
        if '第一次取样 原始光谱' in self.spectral_data:
            n_samples = len(self.spectral_data['第一次取样 原始光谱'])
        else:
            n_samples = 100  # 默认样本数
        
        # 创建模拟品质指标
        np.random.seed(42)  # 设置随机种子以确保结果可重现
        
        # 模拟品质数据
        quality_data = {
            'sample_id': range(1, n_samples + 1),
            'protein_content': np.random.normal(7.5, 1.2, n_samples),  # 蛋白质含量 (%)
            'moisture_content': np.random.normal(14.0, 1.5, n_samples),  # 水分含量 (%)
            'amylose_content': np.random.normal(18.0, 3.0, n_samples),  # 直链淀粉含量 (%)
            'hardness': np.random.normal(50, 10, n_samples),  # 硬度
            'stickiness': np.random.normal(30, 8, n_samples),  # 粘性
            'elasticity': np.random.normal(25, 6, n_samples),  # 弹性
        }
        
        # 确保数值在合理范围内
        quality_data['protein_content'] = np.clip(quality_data['protein_content'], 5, 12)
        quality_data['moisture_content'] = np.clip(quality_data['moisture_content'], 10, 18)
        quality_data['amylose_content'] = np.clip(quality_data['amylose_content'], 10, 30)
        quality_data['hardness'] = np.clip(quality_data['hardness'], 20, 80)
        quality_data['stickiness'] = np.clip(quality_data['stickiness'], 10, 50)
        quality_data['elasticity'] = np.clip(quality_data['elasticity'], 10, 40)
        
        # 创建品质等级（基于综合得分）
        composite_score = (
            quality_data['protein_content'] * 0.2 +
            (20 - quality_data['moisture_content']) * 0.1 +  # 水分越低越好
            quality_data['amylose_content'] * 0.15 +
            quality_data['hardness'] * 0.2 +
            quality_data['stickiness'] * 0.15 +
            quality_data['elasticity'] * 0.2
        )
        
        # 根据综合得分分级
        quality_grades = []
        for score in composite_score:
            if score >= np.percentile(composite_score, 75):
                quality_grades.append('优质')
            elif score >= np.percentile(composite_score, 25):
                quality_grades.append('中等')
            else:
                quality_grades.append('一般')
        
        quality_data['quality_grade'] = quality_grades
        
        self.quality_data = pd.DataFrame(quality_data)
        
        # 保存模拟数据
        self.quality_data.to_csv(self.results_dir / "simulated_quality_data.csv", 
                                index=False, encoding='utf-8-sig')
        
        print(f"已生成 {n_samples} 个样本的模拟品质数据")
    
    def build_regression_models(self, target_column='protein_content'):
        """
        建立回归模型预测连续型品质指标
        
        Args:
            target_column: 目标品质指标列名
        """
        if self.quality_data is None or target_column not in self.quality_data.columns:
            print(f"品质数据中不存在 {target_column} 列")
            return
            
        # 准备数据
        X = self.spectral_data['第一次取样 原始光谱'].values
        y = self.quality_data[target_column].values
        
        # 数据预处理
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # 特征选择 - 选择最重要的特征
        selector = SelectKBest(score_func=f_regression, k=min(100, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42
        )
        
        # 定义模型（包拮PyNIR模型）
        models = {
            'PyNIR PLS': None,  # 将在下面特殊处理
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', gamma='scale'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 训练和评估模型
        results = {}
        predictions = {}
        
        print(f"\n正在训练 {target_column} 预测模型...")
        
        for name, model in models.items():
            try:
                if name == 'PyNIR PLS':
                    # 使用PyNIR的PLS回归
                    try:
                        print(f"正在训练PyNIR PLS模型...")
                        
                        # PyNIR PLS回归 - 使用正确的sklearn-style API
                        pls_obj = Calibration.pls(n_components=10)
                        pls_obj.fit(X_train, y_train)
                        y_pred_test = pls_obj.predict(X_test)
                        
                        # 计算评估指标
                        test_r2 = r2_score(y_test, y_pred_test) if len(y_pred_test) > 0 else 0
                        train_r2 = 0  # PyNIR可能不返回训练R2
                        
                        if len(y_pred_test) == len(y_test):
                            train_rmse = 0  # PyNIR可能不返回训练RMSE
                            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                            
                            results[name] = {
                                'train_r2': train_r2,
                                'test_r2': test_r2,
                                'train_rmse': train_rmse,
                                'test_rmse': test_rmse,
                                'cv_r2_mean': test_r2,  # 使用测试R2作为代替
                                'cv_r2_std': 0
                            }
                            
                            predictions[name] = {
                                'y_test': y_test,
                                'y_pred': y_pred_test
                            }
                            
                            print(f"PyNIR PLS: Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.3f}")
                        else:
                            print(f"PyNIR PLS预测结果维度不匹配，跳过")
                            
                    except Exception as e:
                        print(f"PyNIR PLS建模失败: {e}")
                        continue
                        
                else:
                    # 传统机器学习模型
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 预测
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # 评估
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    
                    # 交叉验证
                    cv_scores = cross_val_score(model, X_selected, y, cv=5, 
                                              scoring='r2', n_jobs=-1)
                    
                    results[name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'cv_r2_mean': cv_scores.mean(),
                        'cv_r2_std': cv_scores.std()
                    }
                    
                    predictions[name] = {
                        'y_test': y_test,
                        'y_pred': y_pred_test
                    }
                    
                    print(f"{name}: Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.3f}")
                
            except Exception as e:
                print(f"训练 {name} 模型时出错: {e}")
        
        # 保存模型结果
        self.models[f'{target_column}_regression'] = {
            'results': results,
            'predictions': predictions,
            'best_model': max(results.keys(), key=lambda k: results[k]['test_r2'])
        }
        
        # 可视化结果
        self._plot_regression_results(results, predictions, target_column)
        
        return results
    
    def build_classification_models(self, target_column='quality_grade'):
        """
        建立分类模型预测品质等级
        
        Args:
            target_column: 目标品质等级列名
        """
        if self.quality_data is None or target_column not in self.quality_data.columns:
            print(f"品质数据中不存在 {target_column} 列")
            return
            
        # 准备数据
        X = self.spectral_data['第一次取样 原始光谱'].values
        y = self.quality_data[target_column].values
        
        # 编码标签
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # 数据预处理
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # 特征选择
        selector = SelectKBest(score_func=f_classif, k=min(100, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y_encoded)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # 定义模型（包拮PyNIR模型）
        models = {
            'PyNIR PLSDA': None,  # PyNIR的PLS判别分析
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 训练和评估模型
        results = {}
        predictions = {}
        
        print(f"\n正在训练 {target_column} 分类模型...")
        
        for name, model in models.items():
            try:
                if name == 'PyNIR PLSDA':
                    # 使用PyNIR的PLSDA分类
                    try:
                        print(f"正在训练PyNIR PLSDA模型...")
                        
                        # PyNIR PLSDA - 使用正确的sklearn-style API
                        plsda_obj = Calibration.plsda(n_components=5)
                        plsda_obj.fit(X_train, y_train)
                        y_pred_test = plsda_obj.predict(X_test)
                        
                        # 计算评估指标
                        test_acc = np.mean(y_test == y_pred_test) if len(y_pred_test) > 0 else 0
                        train_acc = 0  # PyNIR可能不返回训练准确率
                        
                        if len(y_pred_test) == len(y_test):
                            results[name] = {
                                'train_accuracy': train_acc,
                                'test_accuracy': test_acc,
                                'cv_accuracy_mean': test_acc,  # 使用测试准确率作为代替
                                'cv_accuracy_std': 0
                            }
                            
                            predictions[name] = {
                                'y_test': y_test,
                                'y_pred': y_pred_test,
                                'labels': label_encoder.classes_
                            }
                            
                            print(f"PyNIR PLSDA: Test Accuracy = {test_acc:.3f}")
                        else:
                            print(f"PyNIR PLSDA预测结果维度不匹配，跳过")
                            
                    except Exception as e:
                        print(f"PyNIR PLSDA建模失败: {e}")
                        continue
                        
                else:
                    # 传统机器学习模型
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 预测
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # 评估
                    train_acc = model.score(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    
                    # 交叉验证
                    cv_scores = cross_val_score(model, X_selected, y_encoded, cv=5, 
                                              scoring='accuracy', n_jobs=-1)
                    
                    results[name] = {
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'cv_accuracy_mean': cv_scores.mean(),
                        'cv_accuracy_std': cv_scores.std()
                    }
                    
                    predictions[name] = {
                        'y_test': y_test,
                        'y_pred': y_pred_test,
                        'labels': label_encoder.classes_
                    }
                    
                    print(f"{name}: Test Accuracy = {test_acc:.3f}")
                
            except Exception as e:
                print(f"训练 {name} 模型时出错: {e}")
        
        # 保存模型结果
        self.models[f'{target_column}_classification'] = {
            'results': results,
            'predictions': predictions,
            'label_encoder': label_encoder,
            'best_model': max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        }
        
        # 可视化结果
        self._plot_classification_results(results, predictions, target_column)
        
        return results
    
    def _plot_regression_results(self, results, predictions, target_column):
        """绘制回归模型结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model performance comparison
        models = list(results.keys())
        test_r2 = [results[m]['test_r2'] for m in models]
        test_rmse = [results[m]['test_rmse'] for m in models]
        
        x_pos = np.arange(len(models))
        axes[0,0].bar(x_pos, test_r2, alpha=0.7)
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('Model Performance Comparison (R²)')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(models, rotation=45)
        for i, v in enumerate(test_r2):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. RMSE comparison
        axes[0,1].bar(x_pos, test_rmse, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Model')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].set_title('Model Performance Comparison (RMSE)')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(models, rotation=45)
        for i, v in enumerate(test_rmse):
            axes[0,1].text(i, v + max(test_rmse)*0.01, f'{v:.3f}', ha='center')
        
        # 3. Best model predictions vs actual values
        best_model = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_pred = predictions[best_model]
        
        axes[1,0].scatter(best_pred['y_test'], best_pred['y_pred'], alpha=0.7)
        min_val = min(best_pred['y_test'].min(), best_pred['y_pred'].min())
        max_val = max(best_pred['y_test'].max(), best_pred['y_pred'].max())
        axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Values')
        axes[1,0].set_ylabel('Predicted Values')
        axes[1,0].set_title(f'Best Model Prediction Performance ({best_model})')
        axes[1,0].grid(True, alpha=0.3)
        
        # Display R² score
        r2 = results[best_model]['test_r2']
        axes[1,0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1,0].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Cross-validation results
        cv_means = [results[m]['cv_r2_mean'] for m in models]
        cv_stds = [results[m]['cv_r2_std'] for m in models]
        
        axes[1,1].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', capsize=5)
        axes[1,1].set_xlabel('Model')
        axes[1,1].set_ylabel('Cross-validation R² Score')
        axes[1,1].set_title('Cross-validation Results')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(models, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{target_column} Regression Model Results', fontsize=16)
        plt.tight_layout()
        filename = f"regression_results_{target_column}.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_classification_results(self, results, predictions, target_column):
        """绘制分类模型结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model accuracy comparison
        models = list(results.keys())
        test_acc = [results[m]['test_accuracy'] for m in models]
        
        x_pos = np.arange(len(models))
        axes[0,0].bar(x_pos, test_acc, alpha=0.7)
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(models, rotation=45)
        for i, v in enumerate(test_acc):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. Cross-validation results
        cv_means = [results[m]['cv_accuracy_mean'] for m in models]
        cv_stds = [results[m]['cv_accuracy_std'] for m in models]
        
        axes[0,1].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', capsize=5)
        axes[0,1].set_xlabel('Model')
        axes[0,1].set_ylabel('Cross-validation Accuracy')
        axes[0,1].set_title('Cross-validation Results')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(models, rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Best model confusion matrix
        best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_pred = predictions[best_model]
        
        cm = confusion_matrix(best_pred['y_test'], best_pred['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0], 
                   xticklabels=best_pred['labels'],
                   yticklabels=best_pred['labels'])
        axes[1,0].set_xlabel('Predicted Labels')
        axes[1,0].set_ylabel('True Labels')
        axes[1,0].set_title(f'Confusion Matrix ({best_model})')
        
        # 4. Classification report (as table)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            best_pred['y_test'], best_pred['y_pred'], average=None
        )
        
        metrics_data = {
            'Class': best_pred['labels'],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        table_data = []
        for i, label in enumerate(best_pred['labels']):
            table_data.append([label, f'{precision[i]:.3f}', 
                              f'{recall[i]:.3f}', f'{f1[i]:.3f}'])
        
        table = axes[1,1].table(cellText=table_data,
                               colLabels=['Class', 'Precision', 'Recall', 'F1-Score'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1,1].set_title(f'Classification Report ({best_model})')
        
        plt.suptitle(f'{target_column} Classification Model Results', fontsize=16)
        plt.tight_layout()
        filename = f"classification_results_{target_column}.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_quality_prediction_analysis(self):
        """运行完整的品质预测分析"""
        print("开始水稻品质预测分析...")
        
        # 加载数据
        self.load_spectral_data()
        self.load_quality_data()
        
        if self.quality_data is None:
            print("无法加载品质数据，分析终止")
            return
        
        # 显示品质数据概览
        print("\n品质数据概览:")
        print(self.quality_data.describe())
        
        # 建立回归模型
        continuous_targets = ['protein_content', 'moisture_content', 'amylose_content']
        for target in continuous_targets:
            if target in self.quality_data.columns:
                print(f"\n=== {target} 回归建模 ===")
                self.build_regression_models(target)
        
        # 建立分类模型
        categorical_targets = ['quality_grade']
        for target in categorical_targets:
            if target in self.quality_data.columns:
                print(f"\n=== {target} 分类建模 ===")
                self.build_classification_models(target)
        
        # 保存模型摘要
        self._save_model_summary()
        
        print(f"\n品质预测分析完成！所有结果已保存到: {self.results_dir}")
    
    def _save_model_summary(self):
        """保存模型摘要"""
        summary_data = []
        
        for model_key, model_info in self.models.items():
            if 'regression' in model_key:
                for model_name, metrics in model_info['results'].items():
                    summary_data.append({
                        '任务': model_key,
                        '模型': model_name,
                        '类型': '回归',
                        '测试R²': metrics['test_r2'],
                        '测试RMSE': metrics['test_rmse'],
                        'CV均值': metrics['cv_r2_mean'],
                        'CV标准差': metrics['cv_r2_std']
                    })
            elif 'classification' in model_key:
                for model_name, metrics in model_info['results'].items():
                    summary_data.append({
                        '任务': model_key,
                        '模型': model_name,
                        '类型': '分类',
                        '测试准确率': metrics['test_accuracy'],
                        'CV均值': metrics['cv_accuracy_mean'],
                        'CV标准差': metrics['cv_accuracy_std']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.results_dir / "model_summary.csv", 
                         index=False, encoding='utf-8-sig')
        
        print(f"模型摘要已保存: {self.results_dir / 'model_summary.csv'}")

def main():
    """主函数"""
    # 数据文件路径
    spectral_data_file = "/home/daniel/NGW/data/daughter/data/不同品种优质稻光谱扫描.xlsx"
    
    # 创建预测器（这里使用模拟品质数据）
    predictor = RiceQualityPredictor(spectral_data_file)
    
    # 运行完整的品质预测分析
    predictor.run_quality_prediction_analysis()
    
    print("\n水稻品质预测建模完成！")
    print("分析内容包括:")
    print("- 蛋白质含量回归预测")
    print("- 水分含量回归预测") 
    print("- 直链淀粉含量回归预测")
    print("- 品质等级分类预测")
    print("- 模型性能评估和比较")

if __name__ == "__main__":
    main()