#!/usr/bin/env python3
"""
近红外光谱数据预处理脚本
使用PyNIR包进行各种预处理方法：MSC、SNV、导数、Savitzky-Golay滤波等
"""

import pandas as pd
import numpy as np
from pynir import Preprocessing
from pynir import OutlierDetection
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set English font configuration to avoid display issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class NIRPreprocessor:
    def __init__(self, data_file):
        """
        初始化NIR预处理器
        
        Args:
            data_file: Excel文件路径
        """
        self.data_file = data_file
        self.data = {}
        self.wavelengths = None
        
    def load_data(self):
        """加载Excel数据的所有sheet"""
        print("正在加载光谱数据...")
        xl = pd.ExcelFile(self.data_file)
        
        for sheet_name in xl.sheet_names:
            try:
                df = pd.read_excel(self.data_file, sheet_name=sheet_name)
                
                # 处理第一个sheet的特殊格式（包含波长信息）
                if sheet_name == '第一次取样 原始光谱':
                    # 第一列是波长
                    self.wavelengths = df.iloc[:, 0].values
                    # 其余列是光谱数据（转置使每行为一个样本）
                    spectra_raw = df.iloc[:, 1:]
                    
                    # 过滤非数值列
                    numeric_cols = []
                    for col in spectra_raw.columns:
                        try:
                            pd.to_numeric(spectra_raw[col], errors='raise')
                            numeric_cols.append(col)
                        except:
                            print(f"跳过非数值列: {col}")
                    
                    if numeric_cols:
                        spectra = spectra_raw[numeric_cols].T
                        # 确保数据类型是数值
                        spectra = spectra.apply(pd.to_numeric, errors='coerce')
                        # 删除包含NaN的行
                        spectra = spectra.dropna()
                        spectra.columns = [f'wl_{w:.2f}' for w in self.wavelengths[:len(spectra.columns)]]
                        self.data[sheet_name] = spectra
                        print(f"数值列数: {len(numeric_cols)}, 最终光谱数据形状: {spectra.shape}")
                    else:
                        print(f"警告: {sheet_name} 中没有找到数值列")
                        continue
                else:
                    # 其他sheet直接使用，但需要过滤非数值数据
                    # 过滤所有列为数值的数据
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        self.data[sheet_name] = numeric_df
                    else:
                        print(f"警告: {sheet_name} 中没有数值数据")
                        continue
                    
                print(f"已加载 {sheet_name}: {df.shape}")
                
            except Exception as e:
                print(f"加载sheet {sheet_name} 时出错: {e}")
        
        print(f"共加载 {len(self.data)} 个数据表")
        
    def apply_msc(self, spectra):
        """
        使用PyNIR的多元散射校正 (MSC)
        """
        try:
            msc_obj = Preprocessing.msc()
            msc_spectra = msc_obj.fit_transform(spectra)
            return msc_spectra
        except Exception as e:
            print(f"PyNIR MSC处理失败，使用自定义方法: {e}")
            return self._custom_msc(spectra)
    
    def apply_snv(self, spectra):
        """
        使用PyNIR的标准正态变量变换 (SNV)
        """
        try:
            snv_obj = Preprocessing.snv()
            snv_spectra = snv_obj.fit_transform(spectra)
            return snv_spectra
        except Exception as e:
            print(f"PyNIR SNV处理失败，使用自定义方法: {e}")
            return self._custom_snv(spectra)
    
    def apply_derivative(self, spectra, order=1):
        """
        使用PyNIR的导数计算
        
        Args:
            order: 导数阶数 (1 或 2)
        """
        try:
            deriv_obj = Preprocessing.derivate(order=order)
            # PyNIR的derivate可能使用transform而不是fit_transform
            if hasattr(deriv_obj, 'transform'):
                deriv_spectra = deriv_obj.transform(spectra)
            elif hasattr(deriv_obj, 'fit_transform'):
                deriv_spectra = deriv_obj.fit_transform(spectra)
            else:
                # 如果都没有，使用自定义方法
                raise AttributeError("No transform method found")
            return deriv_spectra
        except Exception as e:
            print(f"PyNIR导数计算失败，使用自定义方法: {e}")
            return self._custom_derivative(spectra, order)
    
    def apply_sg_filter(self, spectra, window_length=15, polyorder=3):
        """
        使用PyNIR的Savitzky-Golay滤波
        
        Args:
            window_length: 窗口长度
            polyorder: 多项式阶数
        """
        try:
            # PyNIR的SG滤波
            sg_obj = Preprocessing.SG_filtering(window_length=window_length, polyorder=polyorder)
            sg_spectra = sg_obj.fit_transform(spectra)
            return sg_spectra
        except Exception as e:
            print(f"PyNIR SG滤波失败，使用自定义方法: {e}")
            return self._custom_sg_filter(spectra, window_length, polyorder)
    
    def apply_smoothing(self, spectra, window_size=5):
        """
        使用PyNIR的平滑处理
        """
        try:
            smooth_obj = Preprocessing.smooth(window_size=window_size)
            smooth_spectra = smooth_obj.fit_transform(spectra)
            return smooth_spectra
        except Exception as e:
            print(f"PyNIR平滑处理失败: {e}")
            return spectra
    
    def apply_centralization(self, spectra):
        """
        使用PyNIR的中心化处理
        """
        try:
            center_obj = Preprocessing.centralization()
            centered_spectra = center_obj.fit_transform(spectra)
            return centered_spectra
        except Exception as e:
            print(f"PyNIR中心化处理失败: {e}")
            return spectra - np.mean(spectra, axis=1, keepdims=True)
    
    # 备用的自定义方法
    def _custom_msc(self, spectra):
        """自定义MSC方法（备用）"""
        mean_spectrum = np.mean(spectra, axis=0)
        msc_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            coef = np.polyfit(mean_spectrum, spectra[i], 1)
            msc_spectra[i] = (spectra[i] - coef[1]) / coef[0]
        return msc_spectra
    
    def _custom_snv(self, spectra):
        """自定义SNV方法（备用）"""
        snv_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            snv_spectra[i] = (spectra[i] - np.mean(spectra[i])) / np.std(spectra[i])
        return snv_spectra
    
    def _custom_derivative(self, spectra, order=1):
        """自定义导数方法（备用）"""
        if order == 1:
            return np.gradient(spectra, axis=1)
        elif order == 2:
            first_deriv = np.gradient(spectra, axis=1)
            return np.gradient(first_deriv, axis=1)
        else:
            return spectra
    
    def _custom_sg_filter(self, spectra, window_length=15, polyorder=3):
        """自定义SG滤波方法（备用）"""
        from scipy.signal import savgol_filter
        if window_length % 2 == 0:
            window_length += 1
        smoothed_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            smoothed_spectra[i] = savgol_filter(spectra[i], window_length, polyorder)
        return smoothed_spectra
    
    def preprocess_data(self, sheet_name='第一次取样 原始光谱', methods=['msc', 'snv', 'first_deriv']):
        """
        对指定数据表应用预处理方法
        
        Args:
            sheet_name: 要处理的数据表名称
            methods: 预处理方法列表
        """
        if sheet_name not in self.data:
            print(f"数据表 {sheet_name} 不存在")
            return
            
        original_data = self.data[sheet_name].values
        results = {'original': original_data}
        
        print(f"\n正在对 {sheet_name} 应用预处理方法...")
        
        # 应用PyNIR预处理方法
        if 'msc' in methods:
            print("应用PyNIR多元散射校正 (MSC)...")
            results['msc'] = self.apply_msc(original_data)
            
        if 'snv' in methods:
            print("应用PyNIR标准正态变量变换 (SNV)...")
            results['snv'] = self.apply_snv(original_data)
            
        if 'first_deriv' in methods:
            print("使用PyNIR计算一阶导数...")
            results['first_derivative'] = self.apply_derivative(original_data, order=1)
            
        if 'second_deriv' in methods:
            print("使用PyNIR计算二阶导数...")
            results['second_derivative'] = self.apply_derivative(original_data, order=2)
            
        if 'savgol' in methods:
            print("应用PyNIR的Savitzky-Golay滤波...")
            results['savgol'] = self.apply_sg_filter(original_data)
            
        if 'smooth' in methods:
            print("应用PyNIR平滑处理...")
            results['smooth'] = self.apply_smoothing(original_data)
            
        if 'centralize' in methods:
            print("应用PyNIR中心化处理...")
            results['centralized'] = self.apply_centralization(original_data)
        
        return results
    
    def save_preprocessed_data(self, results, output_dir='../results'):
        """保存预处理结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for method, data in results.items():
            filename = f"preprocessed_{method}.csv"
            filepath = output_path / filename
            
            # 创建DataFrame并保存
            if self.wavelengths is not None:
                columns = [f'wl_{w:.2f}' for w in self.wavelengths]
            else:
                columns = [f'var_{i}' for i in range(data.shape[1])]
                
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"已保存: {filepath}")
    
    def plot_preprocessing_comparison(self, results, output_dir='../results/figures'):
        """Create English-labeled preprocessing comparison plots"""
        if not results or self.wavelengths is None:
            print("No preprocessing results or wavelengths available for plotting")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot preprocessing comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NIR Spectral Preprocessing Comparison', fontsize=16)
        
        # Select first spectrum for visualization
        sample_idx = 0
        
        # Plot 1: Original vs MSC
        if 'original' in results and 'msc' in results:
            axes[0,0].plot(self.wavelengths, results['original'][sample_idx], 'b-', 
                          linewidth=2, label='Original')
            axes[0,0].plot(self.wavelengths, results['msc'][sample_idx], 'r-', 
                          linewidth=2, label='MSC')
            axes[0,0].set_xlabel('Wavelength (nm)')
            axes[0,0].set_ylabel('Absorbance')
            axes[0,0].set_title('Original vs MSC')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Original vs SNV
        if 'original' in results and 'snv' in results:
            axes[0,1].plot(self.wavelengths, results['original'][sample_idx], 'b-', 
                          linewidth=2, label='Original')
            axes[0,1].plot(self.wavelengths, results['snv'][sample_idx], 'g-', 
                          linewidth=2, label='SNV')
            axes[0,1].set_xlabel('Wavelength (nm)')
            axes[0,1].set_ylabel('Absorbance')
            axes[0,1].set_title('Original vs SNV')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: First Derivative
        if 'first_derivative' in results:
            axes[1,0].plot(self.wavelengths, results['first_derivative'][sample_idx], 
                          'purple', linewidth=2, label='1st Derivative')
            axes[1,0].set_xlabel('Wavelength (nm)')
            axes[1,0].set_ylabel('Derivative Value')
            axes[1,0].set_title('First Derivative')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Second Derivative
        if 'second_derivative' in results:
            axes[1,1].plot(self.wavelengths, results['second_derivative'][sample_idx], 
                          'orange', linewidth=2, label='2nd Derivative')
            axes[1,1].set_xlabel('Wavelength (nm)')
            axes[1,1].set_ylabel('Derivative Value')
            axes[1,1].set_title('Second Derivative')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = "preprocessing_comparison_english.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved preprocessing comparison: {filepath}")
        
        # Additional comparison plot with all methods
        if len(results) > 2:
            plt.figure(figsize=(14, 10))
            
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
            method_names = {
                'original': 'Original',
                'msc': 'MSC',
                'snv': 'SNV', 
                'first_derivative': '1st Derivative',
                'second_derivative': '2nd Derivative',
                'savgol': 'Savitzky-Golay',
                'smooth': 'Smoothed',
                'centralized': 'Centralized'
            }
            
            for i, (method, data) in enumerate(results.items()):
                color = colors[i % len(colors)]
                method_label = method_names.get(method, method.capitalize())
                plt.plot(self.wavelengths, data[sample_idx], 
                        color=color, linewidth=2, label=method_label, alpha=0.8)
            
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Absorbance / Derivative Value')
            plt.title('All Preprocessing Methods Comparison')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            filename = "all_preprocessing_methods_english.png"
            filepath = output_path / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved all methods comparison: {filepath}")

def main():
    """主函数"""
    # 数据文件路径
    data_file = "/home/daniel/NGW/data/daughter/data/不同品种优质稻光谱扫描.xlsx"
    
    # 创建预处理器
    processor = NIRPreprocessor(data_file)
    
    # 加载数据
    processor.load_data()
    
    # 应用PyNIR预处理方法
    methods = ['msc', 'snv', 'first_deriv', 'second_deriv', 'savgol', 'smooth', 'centralize']
    results = processor.preprocess_data(methods=methods)
    
    if results:
        # 保存预处理结果
        processor.save_preprocessed_data(results)
        
        # 生成英文版对比图表
        processor.plot_preprocessing_comparison(results)
        
        print("\n预处理完成！")
        print("PyNIR预处理方法说明:")
        print("- MSC: PyNIR多元散射校正，消除散射效应")
        print("- SNV: PyNIR标准正态变量变换，标准化每个样本")
        print("- 一阶导数: PyNIR一阶导数，消除基线漂移")
        print("- 二阶导数: PyNIR二阶导数，进一步消除基线和斜率影响")
        print("- Savgol: PyNIR的Savitzky-Golay平滑滤波")
        print("- Smooth: PyNIR平滑处理")
        print("- Centralize: PyNIR中心化处理")

if __name__ == "__main__":
    main()