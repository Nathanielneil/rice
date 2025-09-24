#!/usr/bin/env python3
"""
使用英文标签重新生成图表以避免中文字体问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
sns.set_palette("husl")

class EnglishNIRVisualizer:
    def __init__(self, data_file):
        """Initialize NIR visualizer with English labels"""
        self.data_file = data_file
        self.data = {}
        self.wavelengths = None
        self.figures_dir = Path("../results/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load spectral data"""
        print("Loading spectral data...")
        xl = pd.ExcelFile(self.data_file)
        
        for sheet_name in xl.sheet_names:
            try:
                df = pd.read_excel(self.data_file, sheet_name=sheet_name)
                
                if sheet_name == '第一次取样 原始光谱':
                    self.wavelengths = df.iloc[:, 0].values
                    spectra_raw = df.iloc[:, 1:]
                    
                    # Filter numeric columns
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
                        self.data[sheet_name] = spectra
                    else:
                        continue
                else:
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        self.data[sheet_name] = numeric_df
                    else:
                        continue
                        
                print(f"Loaded {sheet_name}: {df.shape}")
                
            except Exception as e:
                print(f"Error loading {sheet_name}: {e}")
    
    def plot_raw_spectra_english(self, sheet_name='第一次取样 原始光谱', sample_size=20):
        """Plot raw spectra with English labels"""
        if sheet_name not in self.data:
            return
            
        data = self.data[sheet_name]
        n_samples = min(sample_size, len(data))
        
        plt.figure(figsize=(12, 8))
        
        # Plot selected sample spectra
        for i in range(n_samples):
            plt.plot(self.wavelengths, data.iloc[i], alpha=0.7, linewidth=1)
        
        # Plot mean spectrum
        mean_spectrum = data.mean(axis=0)
        plt.plot(self.wavelengths, mean_spectrum, 'r-', linewidth=2, label='Mean Spectrum')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.title(f'Raw NIR Spectra - Rice Quality Analysis ({n_samples} samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = "raw_spectra_english.png"
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def plot_pca_analysis_english(self, sheet_name='第一次取样 原始光谱', n_components=5):
        """PCA analysis with English labels"""
        if sheet_name not in self.data:
            return
            
        data = self.data[sheet_name].values
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, data.shape[1]))
        pca_result = pca.fit_transform(data_scaled)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Score plot (PC1 vs PC2)
        axes[0,0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,0].set_title('PCA Score Plot (PC1 vs PC2)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        x_pos = range(1, len(explained_variance) + 1)
        axes[0,1].bar(x_pos, explained_variance, alpha=0.7, label='Individual')
        axes[0,1].plot(x_pos, cumulative_variance, 'ro-', label='Cumulative')
        axes[0,1].set_xlabel('Principal Component')
        axes[0,1].set_ylabel('Explained Variance Ratio')
        axes[0,1].set_title('PCA Explained Variance')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Loading plot
        if self.wavelengths is not None:
            axes[1,0].plot(self.wavelengths, pca.components_[0], label='PC1')
            axes[1,0].plot(self.wavelengths, pca.components_[1], label='PC2')
            axes[1,0].set_xlabel('Wavelength (nm)')
        else:
            axes[1,0].plot(pca.components_[0], label='PC1')
            axes[1,0].plot(pca.components_[1], label='PC2')
            axes[1,0].set_xlabel('Variable Index')
        axes[1,0].set_ylabel('Loading Value')
        axes[1,0].set_title('PCA Loading Plot')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 3D plot if enough components
        if pca_result.shape[1] >= 3:
            ax_3d = fig.add_subplot(224, projection='3d')
            scatter = ax_3d.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], 
                                  c=range(len(pca_result)), cmap='viridis', alpha=0.7)
            ax_3d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax_3d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax_3d.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax_3d.set_title('3D PCA Plot')
            plt.colorbar(scatter, ax=ax_3d)
        else:
            axes[1,1].text(0.5, 0.5, 'Need at least 3 PCs\nfor 3D plot', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('3D PCA Plot')
        
        plt.tight_layout()
        filename = "pca_analysis_english.png"
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        # Output PCA results
        print(f"\nPCA Analysis Results:")
        print(f"Explained variance by first {len(explained_variance)} components:")
        for i, var in enumerate(explained_variance):
            print(f"  PC{i+1}: {var:.3f} ({var:.1%})")
        print(f"Cumulative explained variance: {cumulative_variance[-1]:.3f} ({cumulative_variance[-1]:.1%})")
    
    def plot_spectral_analysis_english(self, sheet_name='第一次取样 原始光谱'):
        """Spectral analysis with English labels"""
        if sheet_name not in self.data:
            return
            
        data = self.data[sheet_name].values
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Peak detection
        mean_spectrum = np.mean(data, axis=0)
        peaks, properties = find_peaks(mean_spectrum, 
                                     height=np.mean(mean_spectrum),
                                     distance=20,
                                     prominence=0.01)
        
        axes[0,0].plot(self.wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean Spectrum')
        axes[0,0].plot(self.wavelengths[peaks], mean_spectrum[peaks], 'ro', 
                      markersize=8, label=f'Peaks ({len(peaks)})')
        axes[0,0].set_xlabel('Wavelength (nm)')
        axes[0,0].set_ylabel('Absorbance')
        axes[0,0].set_title('Spectral Peak Detection')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Outlier detection
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = iso_forest.fit_predict(data_scaled)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data_scaled)
        
        colors = ['red' if x == -1 else 'blue' for x in outliers_iso]
        axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.7)
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,1].set_title('Outlier Detection (Red=Outliers)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_data)
        
        scatter = axes[0,2].scatter(pca_data[:, 0], pca_data[:, 1], 
                                   c=cluster_labels, cmap='viridis', alpha=0.7)
        axes[0,2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        silhouette_avg = silhouette_score(pca_data, cluster_labels)
        axes[0,2].set_title(f'K-means Clustering (Silhouette={silhouette_avg:.3f})')
        plt.colorbar(scatter, ax=axes[0,2])
        
        # 4. Cluster spectra
        for cluster_id in range(3):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_mean = np.mean(data[cluster_indices], axis=0)
                axes[1,0].plot(self.wavelengths, cluster_mean, 
                              linewidth=2, label=f'Cluster {cluster_id+1} (n={len(cluster_indices)})')
        
        axes[1,0].set_xlabel('Wavelength (nm)')
        axes[1,0].set_ylabel('Absorbance')
        axes[1,0].set_title('Mean Spectra by Cluster')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Statistical summary
        mean_spec = np.mean(data, axis=0)
        std_spec = np.std(data, axis=0)
        
        axes[1,1].fill_between(self.wavelengths, 
                              mean_spec - 2*std_spec,
                              mean_spec + 2*std_spec,
                              alpha=0.3, label='±2σ interval')
        axes[1,1].plot(self.wavelengths, mean_spec, 'r-', linewidth=2, label='Mean Spectrum')
        axes[1,1].set_xlabel('Wavelength (nm)')
        axes[1,1].set_ylabel('Absorbance')
        axes[1,1].set_title('Spectral Statistics')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Coefficient of variation
        cv = std_spec / (mean_spec + 1e-10) * 100  # CV (%)
        axes[1,2].plot(self.wavelengths, cv)
        axes[1,2].set_xlabel('Wavelength (nm)')
        axes[1,2].set_ylabel('Coefficient of Variation (%)')
        axes[1,2].set_title('Spectral Variability')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = "spectral_analysis_english.png"
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        print(f"Detected {len(peaks)} spectral peaks")
        print(f"Outliers detected: {np.sum(outliers_iso == -1)}/{len(data)} samples")
        print(f"K-means clustering silhouette score: {silhouette_avg:.3f}")

def main():
    """Main function"""
    print("=== Generating English-labeled NIR Analysis Plots ===")
    
    # Data file path
    data_file = "/home/daniel/NGW/data/daughter/data/不同品种优质稻光谱扫描.xlsx"
    
    # Create visualizer
    visualizer = EnglishNIRVisualizer(data_file)
    
    # Load data
    visualizer.load_data()
    
    # Generate plots
    visualizer.plot_raw_spectra_english()
    visualizer.plot_pca_analysis_english()
    visualizer.plot_spectral_analysis_english()
    
    print(f"\nAll English-labeled plots saved to: {visualizer.figures_dir}")
    print("\nGenerated plots:")
    print("- raw_spectra_english.png: Raw NIR spectra")
    print("- pca_analysis_english.png: Principal component analysis") 
    print("- spectral_analysis_english.png: Comprehensive spectral analysis")

if __name__ == "__main__":
    main()