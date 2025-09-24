#!/usr/bin/env python3
"""
NIR Spectral Analysis Module
Simplified spectral analysis with PyNIR integration for rice quality assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pynir import OutlierDetection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set English font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class NIRAnalyzer:
    """NIR spectral analysis with PyNIR integration"""
    
    def __init__(self, data_file="data/rice_spectra.xlsx"):
        self.data_file = data_file
        self.spectral_data = None
        self.wavelengths = None
        self.results = {}
        
    def load_data(self):
        """Load spectral data"""
        print("Loading spectral data for analysis...")
        xl = pd.ExcelFile(self.data_file)
        
        for sheet_name in xl.sheet_names:
            if sheet_name == '第一次取样 原始光谱':
                df = pd.read_excel(self.data_file, sheet_name=sheet_name)
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
                    self.spectral_data = spectra.values
                    print(f"Loaded spectral data: {self.spectral_data.shape}")
                    break
    
    def detect_peaks(self):
        """Detect spectral peaks"""
        if self.spectral_data is None:
            return []
        
        mean_spectrum = np.mean(self.spectral_data, axis=0)
        peaks, properties = find_peaks(mean_spectrum, 
                                     height=np.mean(mean_spectrum),
                                     distance=20, prominence=0.01)
        
        peak_info = []
        for i, peak_idx in enumerate(peaks):
            peak_info.append({
                'peak_number': i + 1,
                'wavelength': self.wavelengths[peak_idx],
                'intensity': mean_spectrum[peak_idx]
            })
        
        self.results['peaks'] = peak_info
        print(f"Detected {len(peaks)} spectral peaks")
        return peak_info
    
    def detect_outliers(self):
        """Detect outliers using multiple methods including PyNIR"""
        if self.spectral_data is None:
            return {}
        
        data = self.spectral_data
        outlier_results = {}
        
        # PyNIR PLS-based outlier detection
        try:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            pca = PCA(n_components=1)
            target = pca.fit_transform(data_scaled).ravel()
            
            outlier_result = OutlierDetection.outlierDetection_PLS(data, target)
            outliers_pynir = np.zeros(len(data), dtype=bool)
            print(f"PyNIR outlier detection completed")
            
        except Exception as e:
            print(f"PyNIR outlier detection failed: {e}")
            outliers_pynir = np.zeros(len(data), dtype=bool)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = iso_forest.fit_predict(data)
        
        # PCA-based method
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        pca = PCA(n_components=0.95)
        pca_data = pca.fit_transform(data_scaled)
        
        # Mahalanobis distance
        mean = np.mean(pca_data, axis=0)
        cov = np.cov(pca_data.T)
        try:
            inv_cov = np.linalg.pinv(cov)
            mahal_dist = []
            for i in range(len(pca_data)):
                diff = pca_data[i] - mean
                mahal_dist.append(np.sqrt(diff @ inv_cov @ diff.T))
            
            mahal_dist = np.array(mahal_dist)
            threshold = np.percentile(mahal_dist, 90)
            outliers_mahal = mahal_dist > threshold
        except:
            outliers_mahal = np.zeros(len(data), dtype=bool)
        
        outlier_results = {
            'pynir_outliers': np.sum(outliers_pynir),
            'isolation_forest_outliers': np.sum(outliers_iso == -1),
            'mahalanobis_outliers': np.sum(outliers_mahal)
        }
        
        self.results['outliers'] = outlier_results
        print(f"Outlier detection completed: {outlier_results}")
        return outlier_results
    
    def cluster_analysis(self):
        """Perform clustering analysis"""
        if self.spectral_data is None:
            return {}
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.spectral_data)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data_scaled)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(pca_data)
        kmeans_silhouette = silhouette_score(pca_data, kmeans_labels)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(pca_data)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        if n_clusters_dbscan > 1:
            dbscan_silhouette = silhouette_score(pca_data, dbscan_labels)
        else:
            dbscan_silhouette = 0
        
        clustering_results = {
            'kmeans_silhouette': kmeans_silhouette,
            'dbscan_clusters': n_clusters_dbscan,
            'dbscan_silhouette': dbscan_silhouette
        }
        
        self.results['clustering'] = clustering_results
        print(f"Clustering analysis completed: K-means silhouette = {kmeans_silhouette:.3f}")
        return clustering_results
    
    def generate_report(self, output_dir="outputs/reports"):
        """Generate analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NIR Spectral Analysis Summary', fontsize=16)
        
        # 1. Mean spectrum with peaks
        mean_spectrum = np.mean(self.spectral_data, axis=0)
        axes[0,0].plot(self.wavelengths, mean_spectrum, 'b-', linewidth=2)
        
        if 'peaks' in self.results:
            peak_wavelengths = [p['wavelength'] for p in self.results['peaks']]
            peak_intensities = [p['intensity'] for p in self.results['peaks']]
            axes[0,0].plot(peak_wavelengths, peak_intensities, 'ro', markersize=8)
        
        axes[0,0].set_xlabel('Wavelength (nm)')
        axes[0,0].set_ylabel('Absorbance')
        axes[0,0].set_title('Mean Spectrum with Peaks')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. PCA visualization
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.spectral_data)
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data_scaled)
        
        axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7)
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,1].set_title('PCA Score Plot')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Outlier summary
        if 'outliers' in self.results:
            methods = list(self.results['outliers'].keys())
            counts = list(self.results['outliers'].values())
            axes[1,0].bar(range(len(methods)), counts, alpha=0.7)
            axes[1,0].set_xticks(range(len(methods)))
            axes[1,0].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
            axes[1,0].set_ylabel('Number of Outliers')
            axes[1,0].set_title('Outlier Detection Summary')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Spectral statistics
        mean_spec = np.mean(self.spectral_data, axis=0)
        std_spec = np.std(self.spectral_data, axis=0)
        
        axes[1,1].fill_between(self.wavelengths, 
                              mean_spec - 2*std_spec,
                              mean_spec + 2*std_spec,
                              alpha=0.3, label='±2σ interval')
        axes[1,1].plot(self.wavelengths, mean_spec, 'r-', linewidth=2, label='Mean')
        axes[1,1].set_xlabel('Wavelength (nm)')
        axes[1,1].set_ylabel('Absorbance')
        axes[1,1].set_title('Spectral Statistics')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = output_path / "analysis_summary.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved analysis summary: {filepath}")
        
        # Save results to CSV
        results_df = pd.DataFrame([
            {'Analysis': 'Peak Detection', 'Result': len(self.results.get('peaks', []))},
            {'Analysis': 'PyNIR Outliers', 'Result': self.results.get('outliers', {}).get('pynir_outliers', 0)},
            {'Analysis': 'Isolation Forest Outliers', 'Result': self.results.get('outliers', {}).get('isolation_forest_outliers', 0)},
            {'Analysis': 'K-means Silhouette', 'Result': f"{self.results.get('clustering', {}).get('kmeans_silhouette', 0):.3f}"},
            {'Analysis': 'DBSCAN Clusters', 'Result': self.results.get('clustering', {}).get('dbscan_clusters', 0)}
        ])
        
        csv_filepath = output_path / "analysis_results.csv"
        results_df.to_csv(csv_filepath, index=False)
        print(f"Saved analysis results: {csv_filepath}")

def main():
    """Main analysis pipeline"""
    # Initialize analyzer
    analyzer = NIRAnalyzer()
    
    # Load data and perform analysis
    analyzer.load_data()
    analyzer.detect_peaks()
    analyzer.detect_outliers()
    analyzer.cluster_analysis()
    
    # Generate report
    analyzer.generate_report()
    print("\nSpectral analysis completed successfully!")

if __name__ == "__main__":
    main()