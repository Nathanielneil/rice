#!/usr/bin/env python3
"""
PyNIR功能集成演示脚本
展示如何使用PyNIR的各种功能进行近红外光谱分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pynir import Preprocessing, OutlierDetection, Calibration, CalibrationTransfer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set English font configuration to avoid display issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class PyNIRDemo:
    def __init__(self, data_file):
        """
        初始化PyNIR演示类
        
        Args:
            data_file: Excel文件路径
        """
        self.data_file = data_file
        self.spectral_data = None
        self.wavelengths = None
        self.results_dir = Path("../results/pynir_demo")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """加载光谱数据"""
        print("正在加载光谱数据...")
        
        df = pd.read_excel(self.data_file, sheet_name='第一次取样 原始光谱')
        self.wavelengths = df.iloc[:, 0].values
        self.spectral_data = df.iloc[:, 1:].T.values
        
        print(f"已加载光谱数据: {self.spectral_data.shape[0]} 个样本, {self.spectral_data.shape[1]} 个波长点")
        print(f"波长范围: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")
        
    def demonstrate_preprocessing(self):
        """演示PyNIR预处理功能"""
        print("\n=== PyNIR预处理方法演示 ===")
        
        # 创建子图
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.ravel()
        
        # Original spectra
        axes[0].plot(self.wavelengths, self.spectral_data[:10].T, alpha=0.7)
        axes[0].set_title('Original Spectra')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Absorbance')
        axes[0].grid(True, alpha=0.3)
        
        # 1. MSC
        try:
            msc_data = Preprocessing.msc(self.spectral_data)
            axes[1].plot(self.wavelengths, msc_data[:10].T, alpha=0.7)
            axes[1].set_title('PyNIR MSC Correction')
            axes[1].set_xlabel('Wavelength (nm)')
            axes[1].set_ylabel('Absorbance')
            axes[1].grid(True, alpha=0.3)
            print("✓ MSC校正成功")
        except Exception as e:
            print(f"✗ MSC校正失败: {e}")
            axes[1].text(0.5, 0.5, 'MSC校正失败', ha='center', va='center', transform=axes[1].transAxes)
        
        # 2. SNV
        try:
            snv_data = Preprocessing.snv(self.spectral_data)
            axes[2].plot(self.wavelengths, snv_data[:10].T, alpha=0.7)
            axes[2].set_title('PyNIR SNV Transform')
            axes[2].set_xlabel('Wavelength (nm)')
            axes[2].set_ylabel('Absorbance')
            axes[2].grid(True, alpha=0.3)
            print("✓ SNV变换成功")
        except Exception as e:
            print(f"✗ SNV变换失败: {e}")
            axes[2].text(0.5, 0.5, 'SNV变换失败', ha='center', va='center', transform=axes[2].transAxes)
        
        # 3. 一阶导数
        try:
            deriv1_data = Preprocessing.derivate(self.spectral_data, order=1)
            axes[3].plot(self.wavelengths, deriv1_data[:10].T, alpha=0.7)
            axes[3].set_title('PyNIR 1st Derivative')
            axes[3].set_xlabel('Wavelength (nm)')
            axes[3].set_ylabel('1st Derivative')
            axes[3].grid(True, alpha=0.3)
            print("✓ 一阶导数计算成功")
        except Exception as e:
            print(f"✗ 一阶导数计算失败: {e}")
            axes[3].text(0.5, 0.5, '一阶导数失败', ha='center', va='center', transform=axes[3].transAxes)
        
        # 4. 二阶导数
        try:
            deriv2_data = Preprocessing.derivate(self.spectral_data, order=2)
            axes[4].plot(self.wavelengths, deriv2_data[:10].T, alpha=0.7)
            axes[4].set_title('PyNIR 2nd Derivative')
            axes[4].set_xlabel('Wavelength (nm)')
            axes[4].set_ylabel('2nd Derivative')
            axes[4].grid(True, alpha=0.3)
            print("✓ 二阶导数计算成功")
        except Exception as e:
            print(f"✗ 二阶导数计算失败: {e}")
            axes[4].text(0.5, 0.5, '二阶导数失败', ha='center', va='center', transform=axes[4].transAxes)
        
        # 5. SG滤波
        try:
            sg_data = Preprocessing.SG_filtering(self.spectral_data, 15, 3)
            axes[5].plot(self.wavelengths, sg_data[:10].T, alpha=0.7)
            axes[5].set_title('PyNIR SG Filtering')
            axes[5].set_xlabel('Wavelength (nm)')
            axes[5].set_ylabel('Absorbance')
            axes[5].grid(True, alpha=0.3)
            print("✓ SG滤波成功")
        except Exception as e:
            print(f"✗ SG滤波失败: {e}")
            axes[5].text(0.5, 0.5, 'SG滤波失败', ha='center', va='center', transform=axes[5].transAxes)
        
        # 6. 平滑处理
        try:
            smooth_data = Preprocessing.smooth(self.spectral_data, window_size=5)
            axes[6].plot(self.wavelengths, smooth_data[:10].T, alpha=0.7)
            axes[6].set_title('PyNIR Smoothing')
            axes[6].set_xlabel('Wavelength (nm)')
            axes[6].set_ylabel('Absorbance')
            axes[6].grid(True, alpha=0.3)
            print("✓ 平滑处理成功")
        except Exception as e:
            print(f"✗ 平滑处理失败: {e}")
            axes[6].text(0.5, 0.5, '平滑处理失败', ha='center', va='center', transform=axes[6].transAxes)
        
        # 7. 中心化
        try:
            centered_data = Preprocessing.centralization(self.spectral_data)
            axes[7].plot(self.wavelengths, centered_data[:10].T, alpha=0.7)
            axes[7].set_title('PyNIR Centralization')
            axes[7].set_xlabel('Wavelength (nm)')
            axes[7].set_ylabel('Absorbance')
            axes[7].grid(True, alpha=0.3)
            print("✓ 中心化成功")
        except Exception as e:
            print(f"✗ 中心化失败: {e}")
            axes[7].text(0.5, 0.5, '中心化失败', ha='center', va='center', transform=axes[7].transAxes)
        
        # 8. 连续小波变换
        try:
            cwt_data = Preprocessing.cwt(self.spectral_data[:, ::10], scales=np.arange(1, 11))  # 下采样减少计算量
            axes[8].imshow(cwt_data[:5], aspect='auto', cmap='viridis')
            axes[8].set_title('PyNIR Continuous Wavelet Transform')
            axes[8].set_xlabel('Wavelength Points')
            axes[8].set_ylabel('Scale')
            print("✓ 连续小波变换成功")
        except Exception as e:
            print(f"✗ 连续小波变换失败: {e}")
            axes[8].text(0.5, 0.5, 'CWT失败', ha='center', va='center', transform=axes[8].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "pynir_preprocessing_demo.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("PyNIR预处理演示完成！")
    
    def demonstrate_outlier_detection(self):
        """演示PyNIR异常值检测功能"""
        print("\n=== PyNIR异常值检测演示 ===")
        
        try:
            # 创建模拟目标变量用于PLS异常值检测
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(self.spectral_data)
            pca = PCA(n_components=1)
            target = pca.fit_transform(data_scaled).ravel()
            
            # PyNIR PLS异常值检测
            outliers = OutlierDetection.outlierDetection_PLS(self.spectral_data, target, n_components=5)
            
            outlier_indices = np.where(outliers)[0]
            normal_indices = np.where(~outliers)[0]
            
            # 可视化结果
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 光谱对比
            if len(normal_indices) > 0:
                normal_mean = np.mean(self.spectral_data[normal_indices], axis=0)
                axes[0].plot(self.wavelengths, normal_mean, 'g-', linewidth=2, 
                           label=f'正常样本均值 (n={len(normal_indices)})')
            
            if len(outlier_indices) > 0:
                for i, idx in enumerate(outlier_indices[:5]):  # 最多显示5个异常样本
                    axes[0].plot(self.wavelengths, self.spectral_data[idx], 'r-', 
                               alpha=0.7, linewidth=1, 
                               label='异常样本' if i == 0 else "")
            
            axes[0].set_xlabel('Wavelength (nm)')
            axes[0].set_ylabel('Absorbance')
            axes[0].set_title('PyNIR Outlier Detection Results')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # PCA散点图标记异常值
            pca_2d = PCA(n_components=2)
            pca_data = pca_2d.fit_transform(data_scaled)
            
            colors = ['red' if outliers[i] else 'blue' for i in range(len(outliers))]
            axes[1].scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.7)
            axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
            axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
            axes[1].set_title('PCA Scatter Plot - PyNIR Outliers (Red)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "pynir_outlier_detection.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ PyNIR异常值检测成功")
            print(f"  检测到 {np.sum(outliers)}/{len(outliers)} 个异常样本")
            
        except Exception as e:
            print(f"✗ PyNIR异常值检测失败: {e}")
    
    def demonstrate_calibration(self):
        """演示PyNIR建模功能"""
        print("\n=== PyNIR建模功能演示 ===")
        
        # 生成模拟目标变量
        np.random.seed(42)
        n_samples = self.spectral_data.shape[0]
        
        # 回归目标（模拟蛋白质含量）
        y_regression = np.random.normal(7.5, 1.2, n_samples)
        y_regression = np.clip(y_regression, 5, 12)
        
        # 分类目标（模拟品质等级）
        y_classification = []
        for val in y_regression:
            if val >= 8.5:
                y_classification.append(2)  # 优质
            elif val >= 7.0:
                y_classification.append(1)  # 中等
            else:
                y_classification.append(0)  # 一般
        y_classification = np.array(y_classification)
        
        # 数据分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            self.spectral_data, y_regression, test_size=0.3, random_state=42
        )
        _, _, y_cls_train, y_cls_test = train_test_split(
            self.spectral_data, y_classification, test_size=0.3, random_state=42
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PLS回归
        try:
            print("正在进行PyNIR PLS回归...")
            pls_result = Calibration.pls(X_train, y_reg_train, X_test, y_reg_test, n_components=8)
            
            if hasattr(pls_result, '__len__') and len(pls_result) == len(y_reg_test):
                y_pred = pls_result
                from sklearn.metrics import r2_score
                r2 = r2_score(y_reg_test, y_pred)
                
                axes[0,0].scatter(y_reg_test, y_pred, alpha=0.7)
                min_val = min(y_reg_test.min(), y_pred.min())
                max_val = max(y_reg_test.max(), y_pred.max())
                axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                axes[0,0].set_xlabel('Actual Values')
                axes[0,0].set_ylabel('Predicted Values')
                axes[0,0].set_title(f'PyNIR PLS Regression (R² = {r2:.3f})')
                axes[0,0].grid(True, alpha=0.3)
                
                print(f"✓ PLS回归成功, R² = {r2:.3f}")
            else:
                axes[0,0].text(0.5, 0.5, 'PLS回归结果格式异常', ha='center', va='center', transform=axes[0,0].transAxes)
                print("✗ PLS回归结果格式异常")
                
        except Exception as e:
            print(f"✗ PLS回归失败: {e}")
            axes[0,0].text(0.5, 0.5, f'PLS回归失败\n{str(e)[:30]}...', ha='center', va='center', transform=axes[0,0].transAxes)
        
        # 2. PLSDA分类
        try:
            print("正在进行PyNIR PLSDA分类...")
            plsda_result = Calibration.plsda(X_train, y_cls_train, X_test, y_cls_test, n_components=5)
            
            if hasattr(plsda_result, '__len__') and len(plsda_result) == len(y_cls_test):
                y_pred_cls = plsda_result
                accuracy = np.mean(y_cls_test == y_pred_cls)
                
                # 混淆矩阵
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_cls_test, y_pred_cls)
                
                im = axes[0,1].imshow(cm, cmap='Blues', alpha=0.7)
                axes[0,1].set_xlabel('Predicted Labels')
                axes[0,1].set_ylabel('True Labels')
                axes[0,1].set_title(f'PyNIR PLSDA Classification (Acc = {accuracy:.3f})')
                
                # 添加数值标注
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        axes[0,1].text(j, i, str(cm[i, j]), ha='center', va='center')
                
                plt.colorbar(im, ax=axes[0,1])
                print(f"✓ PLSDA分类成功, 准确率 = {accuracy:.3f}")
            else:
                axes[0,1].text(0.5, 0.5, 'PLSDA分类结果格式异常', ha='center', va='center', transform=axes[0,1].transAxes)
                print("✗ PLSDA分类结果格式异常")
                
        except Exception as e:
            print(f"✗ PLSDA分类失败: {e}")
            axes[0,1].text(0.5, 0.5, f'PLSDA分类失败\n{str(e)[:30]}...', ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. 随机森林（作为对比）
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train, y_reg_train)
            y_pred_rf = rf_reg.predict(X_test)
            
            from sklearn.metrics import r2_score
            r2_rf = r2_score(y_reg_test, y_pred_rf)
            
            axes[1,0].scatter(y_reg_test, y_pred_rf, alpha=0.7, color='orange')
            min_val = min(y_reg_test.min(), y_pred_rf.min())
            max_val = max(y_reg_test.max(), y_pred_rf.max())
            axes[1,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[1,0].set_xlabel('Actual Values')
            axes[1,0].set_ylabel('Predicted Values')
            axes[1,0].set_title(f'Random Forest Regression (R² = {r2_rf:.3f})')
            axes[1,0].grid(True, alpha=0.3)
            
            print(f"✓ 随机森林回归对比成功, R² = {r2_rf:.3f}")
            
        except Exception as e:
            print(f"✗ 随机森林回归失败: {e}")
            axes[1,0].text(0.5, 0.5, '随机森林回归失败', ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 4. 建模结果汇总
        axes[1,1].axis('off')
        summary_text = """
PyNIR建模功能演示总结:

✓ PLS回归: 偏最小二乘回归
  - 适用于高维数据回归
  - 可处理多重共线性
  
✓ PLSDA: PLS判别分析
  - 基于PLS的分类方法
  - 适用于光谱分类任务
  
✓ 自动交叉验证和优化
✓ 内置模型评估指标
        """
        
        axes[1,1].text(0.1, 0.9, summary_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "pynir_calibration_demo.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("PyNIR建模演示完成！")
    
    def run_complete_demo(self):
        """运行完整的PyNIR功能演示"""
        print("=== PyNIR功能完整演示开始 ===")
        
        # 加载数据
        self.load_data()
        
        # 预处理演示
        self.demonstrate_preprocessing()
        
        # 异常值检测演示
        self.demonstrate_outlier_detection()
        
        # 建模演示
        self.demonstrate_calibration()
        
        print(f"\n=== PyNIR功能演示完成 ===")
        print(f"所有结果已保存到: {self.results_dir}")
        print("\nPyNIR主要功能总结:")
        print("1. 预处理: MSC, SNV, 导数, SG滤波, 平滑, 中心化, 小波变换")
        print("2. 异常值检测: 基于PLS的异常值检测")
        print("3. 建模: PLS回归, PLSDA分类")
        print("4. 校准转移: 不同仪器间的校准转移")

def main():
    """主函数"""
    # 数据文件路径
    data_file = "/home/daniel/NGW/data/daughter/data/不同品种优质稻光谱扫描.xlsx"
    
    # 创建演示对象
    demo = PyNIRDemo(data_file)
    
    # 运行完整演示
    demo.run_complete_demo()

if __name__ == "__main__":
    main()