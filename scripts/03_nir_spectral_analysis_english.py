#!/usr/bin/env python3
"""
NIR Spectral Analysis Script with English Labels
Near-Infrared Spectroscopy analysis for rice quality with PyNIR integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pynir import OutlierDetection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set English font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
sns.set_palette("husl")

class NIRSpectralAnalyzer:
    def __init__(self, data_file):
        """
        Initialize NIR Spectral Analyzer
        
        Args:
            data_file: Excel file path
        """
        self.data_file = data_file
        self.data = {}
        self.wavelengths = None
        self.results_dir = Path("../results/analysis_english")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load Excel data"""
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
                print(f"Error loading sheet {sheet_name}: {e}")
    
    def peak_detection(self, sheet_name='第一次取样 原始光谱'):
        """Spectral peak detection"""
        if sheet_name not in self.data:
            return []
        
        data = self.data[sheet_name].values
        mean_spectrum = np.mean(data, axis=0)
        
        # Find peaks with better parameters
        peaks, properties = find_peaks(mean_spectrum, 
                                     height=np.mean(mean_spectrum),
                                     distance=20,
                                     prominence=0.01)
        
        peak_info = []
        for i, peak_idx in enumerate(peaks):
            wavelength = self.wavelengths[peak_idx]
            intensity = mean_spectrum[peak_idx]
            peak_info.append({
                'peak_number': i + 1,
                'wavelength': wavelength,
                'intensity': intensity
            })
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.plot(self.wavelengths, mean_spectrum, 'b-', linewidth=2, label='Mean Spectrum')
        plt.plot(self.wavelengths[peaks], mean_spectrum[peaks], 'ro', 
                markersize=8, label=f'Peaks ({len(peaks)})')
        
        # Mark peaks with annotations
        for i, peak_idx in enumerate(peaks):
            plt.annotate(f'Peak {i+1}\n{self.wavelengths[peak_idx]:.1f}nm', 
                        xy=(self.wavelengths[peak_idx], mean_spectrum[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
        plt.title('NIR Spectral Peak Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = "peak_detection_english.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        return peak_info
    
    def outlier_detection(self, sheet_name='第一次取样 原始光谱'):
        """Outlier detection using multiple methods"""
        if sheet_name not in self.data:
            return {}
        
        data = self.data[sheet_name].values
        
        # Method 1: PyNIR PLS-based outlier detection
        outliers_pynir = np.zeros(len(data), dtype=bool)
        try:
            print("Using PyNIR PLS outlier detection...")
            # Prepare target for PLS outlier detection
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            pca = PCA(n_components=1)
            target = pca.fit_transform(data_scaled).ravel()
            
            # PyNIR outlier detection - try direct function call approach
            outlier_result = OutlierDetection.outlierDetection_PLS(data, target)
            # Try to extract outlier indices or boolean mask from the result
            if hasattr(outlier_result, 'outlier_indices'):
                outlier_indices = outlier_result.outlier_indices
                outliers_pynir = np.zeros(len(data), dtype=bool)
                outliers_pynir[outlier_indices] = True
            elif hasattr(outlier_result, 'outliers'):
                outliers_pynir = outlier_result.outliers
            else:
                # If we can't extract outliers, fall back to zeros
                outliers_pynir = np.zeros(len(data), dtype=bool)
            print(f"PyNIR detected {np.sum(outliers_pynir)}/{len(data)} outlier samples")
        except Exception as e:
            print(f"PyNIR outlier detection failed: {e}")
            outliers_pynir = np.zeros(len(data), dtype=bool)
        
        # Method 2: PCA-based outlier detection
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        pca = PCA(n_components=0.95)  # Keep 95% variance
        pca_data = pca.fit_transform(data_scaled)
        
        # Method 3: Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers_iso = iso_forest.fit_predict(data_scaled)
        
        # Method 4: Mahalanobis distance
        mean = np.mean(pca_data, axis=0)
        cov = np.cov(pca_data.T)
        
        try:
            mahal_dist = []
            inv_cov = np.linalg.pinv(cov)
            for i in range(len(pca_data)):
                diff = pca_data[i] - mean
                mahal_dist.append(np.sqrt(diff @ inv_cov @ diff.T))
            
            mahal_dist = np.array(mahal_dist)
            threshold = np.percentile(mahal_dist, 90)  # 90th percentile as threshold
            outliers_mahal = mahal_dist > threshold
        except:
            print("Mahalanobis distance calculation failed, skipping this method")
            outliers_mahal = np.zeros(len(data), dtype=bool)
        
        # Visualization of outlier detection results
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. PCA scatter plot with outliers marked
        colors = ['red' if x == -1 else 'blue' for x in outliers_iso]
        axes[0,0].scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.6)
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,0].set_title('PCA Scatter Plot - Isolation Forest Outliers (Red)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Mahalanobis distance distribution
        axes[0,1].hist(mahal_dist, bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
        axes[0,1].set_xlabel('Mahalanobis Distance')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Mahalanobis Distance Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Comparison of outlier detection methods
        methods = ['PyNIR PLS', 'Isolation Forest', 'Mahalanobis']
        outlier_counts = [
            np.sum(outliers_pynir),
            np.sum(outliers_iso == -1),
            np.sum(outliers_mahal)
        ]
        
        axes[0,2].bar(methods, outlier_counts, alpha=0.7)
        axes[0,2].set_ylabel('Number of Outliers')
        axes[0,2].set_title('Outlier Detection Method Comparison')
        axes[0,2].tick_params(axis='x', rotation=45)
        for i, v in enumerate(outlier_counts):
            axes[0,2].text(i, v + 1, str(v), ha='center')
        
        # 4. PCA with Mahalanobis outliers
        mahal_colors = ['red' if x else 'blue' for x in outliers_mahal]
        axes[1,0].scatter(pca_data[:, 0], pca_data[:, 1], c=mahal_colors, alpha=0.6)
        axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1,0].set_title('PCA Scatter Plot - Mahalanobis Outliers (Red)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. PyNIR outliers (if any detected)
        pynir_colors = ['red' if x else 'blue' for x in outliers_pynir]
        axes[1,1].scatter(pca_data[:, 0], pca_data[:, 1], c=pynir_colors, alpha=0.6)
        axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1,1].set_title('PCA Scatter Plot - PyNIR PLS Outliers (Red)')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Outlier agreement analysis
        # Create Venn diagram-like comparison
        iso_outliers = set(np.where(outliers_iso == -1)[0])
        mahal_outliers = set(np.where(outliers_mahal)[0])
        pynir_outliers = set(np.where(outliers_pynir)[0])
        
        all_outliers = iso_outliers.union(mahal_outliers).union(pynir_outliers)
        agreement_data = {
            'All Methods': len(iso_outliers.intersection(mahal_outliers).intersection(pynir_outliers)),
            'ISO + Mahal': len(iso_outliers.intersection(mahal_outliers)) - len(iso_outliers.intersection(mahal_outliers).intersection(pynir_outliers)),
            'ISO + PyNIR': len(iso_outliers.intersection(pynir_outliers)) - len(iso_outliers.intersection(mahal_outliers).intersection(pynir_outliers)),
            'Mahal + PyNIR': len(mahal_outliers.intersection(pynir_outliers)) - len(iso_outliers.intersection(mahal_outliers).intersection(pynir_outliers)),
            'ISO Only': len(iso_outliers - mahal_outliers - pynir_outliers),
            'Mahal Only': len(mahal_outliers - iso_outliers - pynir_outliers),
            'PyNIR Only': len(pynir_outliers - iso_outliers - mahal_outliers)
        }
        
        categories = list(agreement_data.keys())
        values = list(agreement_data.values())
        
        axes[1,2].bar(categories, values, alpha=0.7)
        axes[1,2].set_ylabel('Number of Samples')
        axes[1,2].set_title('Outlier Detection Agreement')
        axes[1,2].tick_params(axis='x', rotation=45)
        for i, v in enumerate(values):
            if v > 0:
                axes[1,2].text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        filename = "outlier_detection_english.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        # Get outlier indices for each method
        pynir_outlier_indices = np.where(outliers_pynir)[0] if np.any(outliers_pynir) else []
        iso_outlier_indices = np.where(outliers_iso == -1)[0]
        mahal_outlier_indices = np.where(outliers_mahal)[0]
        
        results = {
            'pynir_outliers': pynir_outlier_indices,
            'isolation_forest_outliers': iso_outlier_indices,
            'mahalanobis_outliers': mahal_outlier_indices,
            'outlier_counts': outlier_counts,
            'agreement_analysis': agreement_data
        }
        
        return results
    
    def clustering_analysis(self, sheet_name='第一次取样 原始光谱'):
        """Clustering analysis using K-means and DBSCAN"""
        if sheet_name not in self.data:
            return {}
        
        data = self.data[sheet_name].values
        
        # Data preprocessing
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data_scaled)
        
        # K-means clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. K-means clustering result
        scatter1 = axes[0,0].scatter(pca_data[:, 0], pca_data[:, 1], 
                                    c=kmeans_labels, cmap='viridis', alpha=0.7)
        axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,0].set_title(f'K-means Clustering (k={n_clusters})\nSilhouette Score: {kmeans_silhouette:.3f}')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0,0])
        
        # 2. DBSCAN clustering result
        scatter2 = axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], 
                                    c=dbscan_labels, cmap='viridis', alpha=0.7)
        axes[0,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0,1].set_title(f'DBSCAN Clustering ({n_clusters_dbscan} clusters)\nSilhouette Score: {dbscan_silhouette:.3f}')
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0,1])
        
        # 3. Elbow method for optimal K
        K_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        for k in K_range:
            kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_k = kmeans_k.fit_predict(pca_data)
            inertias.append(kmeans_k.inertia_)
            silhouette_scores.append(silhouette_score(pca_data, labels_k))
        
        axes[0,2].plot(K_range, inertias, 'bo-')
        axes[0,2].set_xlabel('Number of Clusters (k)')
        axes[0,2].set_ylabel('Inertia')
        axes[0,2].set_title('Elbow Method for Optimal k')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Silhouette scores for different K
        axes[1,0].plot(K_range, silhouette_scores, 'ro-')
        axes[1,0].set_xlabel('Number of Clusters (k)')
        axes[1,0].set_ylabel('Silhouette Score')
        axes[1,0].set_title('Silhouette Analysis')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Cluster mean spectra (K-means)
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(kmeans_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_mean = np.mean(data[cluster_indices], axis=0)
                axes[1,1].plot(self.wavelengths, cluster_mean, 
                              linewidth=2, label=f'Cluster {cluster_id+1} (n={len(cluster_indices)})')
        
        axes[1,1].set_xlabel('Wavelength (nm)')
        axes[1,1].set_ylabel('Absorbance')
        axes[1,1].set_title('Mean Spectra by K-means Cluster')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Cluster size comparison
        kmeans_counts = np.bincount(kmeans_labels)
        dbscan_counts = np.bincount(dbscan_labels[dbscan_labels >= 0])  # Exclude noise points
        
        # Create grouped bar chart with proper alignment
        max_clusters = max(len(kmeans_counts), len(dbscan_counts))
        x_pos = np.arange(max_clusters)
        width = 0.35
        
        # Pad shorter array with zeros
        kmeans_padded = np.pad(kmeans_counts, (0, max_clusters - len(kmeans_counts)), 'constant')
        dbscan_padded = np.pad(dbscan_counts, (0, max_clusters - len(dbscan_counts)), 'constant')
        
        axes[1,2].bar(x_pos - width/2, kmeans_padded, width, label='K-means', alpha=0.7)
        if len(dbscan_counts) > 0:
            axes[1,2].bar(x_pos + width/2, dbscan_padded, width, label='DBSCAN', alpha=0.7)
        
        axes[1,2].set_xlabel('Cluster ID')
        axes[1,2].set_ylabel('Cluster Size')
        axes[1,2].set_title('Cluster Size Comparison')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        # Add noise points for DBSCAN
        if -1 in dbscan_labels:
            noise_count = np.sum(dbscan_labels == -1)
            max_height = max(max(kmeans_padded), max(dbscan_padded) if len(dbscan_padded) > 0 else 0)
            axes[1,2].text(0.7, max_height * 0.8, 
                          f'DBSCAN Noise: {noise_count}', 
                          bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        
        plt.tight_layout()
        filename = "clustering_analysis_english.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        
        results = {
            'kmeans_labels': kmeans_labels,
            'kmeans_silhouette': kmeans_silhouette,
            'dbscan_labels': dbscan_labels,
            'dbscan_silhouette': dbscan_silhouette,
            'n_clusters_dbscan': n_clusters_dbscan,
            'optimal_k_candidates': K_range[np.argmax(silhouette_scores)],
            'silhouette_scores': silhouette_scores
        }
        
        return results
    
    def feature_extraction(self, sheet_name='第一次取样 原始光谱'):
        """Extract spectral features"""
        if sheet_name not in self.data:
            return pd.DataFrame()
        
        print("Extracting spectral features...")
        data = self.data[sheet_name].values
        features_list = []
        
        for i, spectrum in enumerate(data):
            features = {'sample_id': i + 1}
            
            # 1. Statistical features
            features['mean'] = np.mean(spectrum)
            features['std'] = np.std(spectrum)
            features['var'] = np.var(spectrum)
            features['min'] = np.min(spectrum)
            features['max'] = np.max(spectrum)
            features['range'] = features['max'] - features['min']
            features['median'] = np.median(spectrum)
            features['q25'] = np.percentile(spectrum, 25)
            features['q75'] = np.percentile(spectrum, 75)
            features['iqr'] = features['q75'] - features['q25']
            features['skewness'] = skew(spectrum)
            features['kurtosis'] = kurtosis(spectrum)
            
            # 2. Shape features
            features['peak_count'] = len(find_peaks(spectrum, height=np.mean(spectrum))[0])
            
            # 3. Peak features
            peaks, peak_properties = find_peaks(spectrum, height=np.mean(spectrum), distance=10)
            if len(peaks) > 0:
                features['max_peak_intensity'] = np.max(spectrum[peaks])
                features['max_peak_wavelength'] = self.wavelengths[peaks[np.argmax(spectrum[peaks])]]
                features['peak_intensity_std'] = np.std(spectrum[peaks])
            else:
                features['max_peak_intensity'] = 0
                features['max_peak_wavelength'] = 0
                features['peak_intensity_std'] = 0
            
            # 4. Regional features (divide spectrum into regions)
            n_regions = 5
            region_size = len(spectrum) // n_regions
            for region in range(n_regions):
                start_idx = region * region_size
                end_idx = (region + 1) * region_size if region < n_regions - 1 else len(spectrum)
                region_data = spectrum[start_idx:end_idx]
                features[f'region_{region+1}_mean'] = np.mean(region_data)
                features[f'region_{region+1}_std'] = np.std(region_data)
            
            # 5. Ratio features
            if len(spectrum) > 4:
                quarter_size = len(spectrum) // 4
                first_quarter = spectrum[:quarter_size]
                last_quarter = spectrum[-quarter_size:]
                features['first_last_quarter_ratio'] = np.mean(first_quarter) / (np.mean(last_quarter) + 1e-10)
            
            # 6. Derivative features
            first_derivative = np.gradient(spectrum)
            second_derivative = np.gradient(first_derivative)
            features['first_derivative_mean'] = np.mean(first_derivative)
            features['first_derivative_std'] = np.std(first_derivative)
            features['second_derivative_mean'] = np.mean(second_derivative)
            features['second_derivative_std'] = np.std(second_derivative)
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Save features
        features_df.to_csv(self.results_dir / "extracted_features_english.csv", 
                          index=False, encoding='utf-8-sig')
        
        print(f"Successfully extracted {len(features_df.columns)-1} spectral features")
        print("Features include: statistical features, shape features, peak features, regional features, ratio features, derivative features")
        
        return features_df
    
    def run_full_analysis(self):
        """Run complete spectral analysis"""
        print("Starting complete near-infrared spectral analysis...")
        
        # Load data
        self.load_data()
        
        if '第一次取样 原始光谱' not in self.data:
            print("Raw spectral data not found!")
            return {}
        
        results = {}
        
        # 1. Peak detection
        print("\n=== 1. Spectral Peak Detection ===")
        peaks = self.peak_detection()
        results['peaks'] = peaks
        
        if peaks:
            print(f"Detected {len(peaks)} spectral peaks:")
            for peak in peaks:
                print(f"  Peak {peak['peak_number']}: {peak['wavelength']:.1f}nm (Intensity: {peak['intensity']:.4f})")
        else:
            print("No significant peaks detected")
        
        # 2. Outlier detection
        print("\n=== 2. Outlier Detection ===")
        outlier_results = self.outlier_detection()
        results['outliers'] = outlier_results
        
        print("Outlier detection results:")
        if outlier_results:
            methods = ['PyNIR PLS', 'Isolation Forest', 'Mahalanobis Distance']
            for i, method in enumerate(methods):
                count = outlier_results['outlier_counts'][i]
                total = len(self.data['第一次取样 原始光谱'])
                print(f"  {method}: {count}/{total} outlier samples")
        
        # 3. Clustering analysis
        print("\n=== 3. Clustering Analysis ===")
        clustering_results = self.clustering_analysis()
        results['clustering'] = clustering_results
        
        if clustering_results:
            print("Clustering analysis results:")
            print(f"  K-means (k=3): Silhouette coefficient = {clustering_results['kmeans_silhouette']:.3f}")
            print(f"  DBSCAN: {clustering_results['n_clusters_dbscan']} clusters, Silhouette coefficient = {clustering_results['dbscan_silhouette']:.3f}")
        
        # 4. Feature extraction
        print("\n=== 4. Spectral Feature Extraction ===")
        features = self.feature_extraction()
        results['features'] = features
        
        print(f"Feature extraction completed!")
        print(f"Features saved to: {self.results_dir / 'extracted_features_english.csv'}")
        
        print(f"\nAnalysis completed! All results saved to: {self.results_dir}")
        
        return results

def main():
    """Main function"""
    # Data file path
    data_file = "/home/daniel/NGW/data/daughter/data/不同品种优质稻光谱扫描.xlsx"
    
    # Create analyzer
    analyzer = NIRSpectralAnalyzer(data_file)
    
    # Run complete analysis
    features = analyzer.run_full_analysis()
    
    print("\nSpectral analysis completed!")
    print("Analysis content includes:")
    print("- Spectral peak detection: Identify characteristic peak positions and intensities")
    print("- Outlier detection: Identify abnormal spectral samples")
    print("- Clustering analysis: Sample grouping and pattern recognition") 
    print("- Feature extraction: Extract spectral features for modeling")

if __name__ == "__main__":
    main()