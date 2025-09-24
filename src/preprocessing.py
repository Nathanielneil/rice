#!/usr/bin/env python3
"""
NIR Spectral Data Preprocessing Module
Simplified and optimized version for rice quality analysis with PyNIR integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pynir import Preprocessing
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set English font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class NIRPreprocessor:
    """NIR spectroscopy data preprocessing with PyNIR integration"""
    
    def __init__(self, data_file="data/rice_spectra.xlsx"):
        self.data_file = data_file
        self.data = {}
        self.wavelengths = None
        self.processed_data = {}
        
    def load_data(self):
        """Load spectral data from Excel file"""
        print("Loading NIR spectral data...")
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
                        self.data['raw_spectra'] = spectra.values
                        print(f"Loaded raw spectra: {spectra.shape}")
                        
            except Exception as e:
                print(f"Error loading {sheet_name}: {e}")
    
    def apply_preprocessing(self, methods=['msc', 'snv', 'first_deriv', 'second_deriv']):
        """Apply PyNIR preprocessing methods"""
        if 'raw_spectra' not in self.data:
            print("No raw spectral data available")
            return
            
        raw_data = self.data['raw_spectra']
        results = {'original': raw_data}
        
        print("Applying PyNIR preprocessing methods...")
        
        # MSC - Multiplicative Scatter Correction
        if 'msc' in methods:
            try:
                msc_obj = Preprocessing.msc()
                msc_data = msc_obj.fit_transform(raw_data)
                results['msc'] = msc_data
                print("✓ MSC correction applied")
            except Exception as e:
                print(f"✗ MSC failed: {e}")
        
        # SNV - Standard Normal Variate
        if 'snv' in methods:
            try:
                snv_obj = Preprocessing.snv()
                snv_data = snv_obj.fit_transform(raw_data)
                results['snv'] = snv_data
                print("✓ SNV transformation applied")
            except Exception as e:
                print(f"✗ SNV failed: {e}")
        
        # First derivative
        if 'first_deriv' in methods:
            try:
                deriv1_obj = Preprocessing.derivate(order=1)
                deriv1_data = deriv1_obj.fit_transform(raw_data)
                results['first_derivative'] = deriv1_data
                print("✓ First derivative calculated")
            except Exception as e:
                print(f"✗ First derivative failed: {e}")
        
        # Second derivative
        if 'second_deriv' in methods:
            try:
                deriv2_obj = Preprocessing.derivate(order=2)
                deriv2_data = deriv2_obj.fit_transform(raw_data)
                results['second_derivative'] = deriv2_data
                print("✓ Second derivative calculated")
            except Exception as e:
                print(f"✗ Second derivative failed: {e}")
        
        self.processed_data = results
        return results
    
    def plot_comparison(self, output_dir="outputs/figures"):
        """Create preprocessing comparison plots"""
        if not self.processed_data:
            print("No processed data available for plotting")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NIR Spectral Preprocessing Comparison', fontsize=16)
        
        sample_idx = 0  # Use first spectrum for visualization
        
        # Original vs MSC
        if 'original' in self.processed_data and 'msc' in self.processed_data:
            axes[0,0].plot(self.wavelengths, self.processed_data['original'][sample_idx], 'b-', 
                          linewidth=2, label='Original')
            axes[0,0].plot(self.wavelengths, self.processed_data['msc'][sample_idx], 'r-', 
                          linewidth=2, label='MSC')
            axes[0,0].set_xlabel('Wavelength (nm)')
            axes[0,0].set_ylabel('Absorbance')
            axes[0,0].set_title('Original vs MSC')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Original vs SNV
        if 'original' in self.processed_data and 'snv' in self.processed_data:
            axes[0,1].plot(self.wavelengths, self.processed_data['original'][sample_idx], 'b-', 
                          linewidth=2, label='Original')
            axes[0,1].plot(self.wavelengths, self.processed_data['snv'][sample_idx], 'g-', 
                          linewidth=2, label='SNV')
            axes[0,1].set_xlabel('Wavelength (nm)')
            axes[0,1].set_ylabel('Absorbance')
            axes[0,1].set_title('Original vs SNV')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # First Derivative
        if 'first_derivative' in self.processed_data:
            axes[1,0].plot(self.wavelengths, self.processed_data['first_derivative'][sample_idx], 
                          'purple', linewidth=2)
            axes[1,0].set_xlabel('Wavelength (nm)')
            axes[1,0].set_ylabel('1st Derivative')
            axes[1,0].set_title('First Derivative')
            axes[1,0].grid(True, alpha=0.3)
        
        # Second Derivative
        if 'second_derivative' in self.processed_data:
            axes[1,1].plot(self.wavelengths, self.processed_data['second_derivative'][sample_idx], 
                          'orange', linewidth=2)
            axes[1,1].set_xlabel('Wavelength (nm)')
            axes[1,1].set_ylabel('2nd Derivative')
            axes[1,1].set_title('Second Derivative')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = output_path / "preprocessing_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved preprocessing comparison: {filepath}")
    
    def save_processed_data(self, output_dir="outputs/processed_data"):
        """Save processed data to CSV files"""
        if not self.processed_data:
            print("No processed data to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for method, data in self.processed_data.items():
            filename = f"{method}_data.csv"
            filepath = output_path / filename
            
            # Create DataFrame
            columns = [f'wl_{w:.2f}' for w in self.wavelengths]
            df = pd.DataFrame(data, columns=columns)
            df.to_csv(filepath, index=False)
            print(f"Saved: {filepath}")

def main():
    """Main preprocessing pipeline"""
    # Initialize preprocessor
    processor = NIRPreprocessor()
    
    # Load and process data
    processor.load_data()
    results = processor.apply_preprocessing()
    
    if results:
        # Generate visualizations and save data
        processor.plot_comparison()
        processor.save_processed_data()
        print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()