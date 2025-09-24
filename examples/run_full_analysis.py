#!/usr/bin/env python3
"""
Rice - Complete NIR Quality Analysis Pipeline
Simplified end-to-end analysis using all modules
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import NIRPreprocessor
from analysis import NIRAnalyzer
from modeling import RiceQualityModeler

def main():
    """Run complete NIR analysis pipeline"""
    print("="*60)
    print("RICE - NIR QUALITY ANALYSIS TOOLKIT")
    print("="*60)
    
    # Step 1: Data Preprocessing
    print("\n🔄 STEP 1: DATA PREPROCESSING")
    print("-" * 40)
    preprocessor = NIRPreprocessor()
    preprocessor.load_data()
    preprocessor.apply_preprocessing()
    preprocessor.plot_comparison()
    preprocessor.save_processed_data()
    
    # Step 2: Spectral Analysis
    print("\n📊 STEP 2: SPECTRAL ANALYSIS")
    print("-" * 40)
    analyzer = NIRAnalyzer()
    analyzer.load_data()
    analyzer.detect_peaks()
    analyzer.detect_outliers()
    analyzer.cluster_analysis()
    analyzer.generate_report()
    
    # Step 3: Quality Prediction Modeling
    print("\n🎯 STEP 3: QUALITY PREDICTION MODELING")
    print("-" * 40)
    modeler = RiceQualityModeler()
    modeler.load_spectral_data()
    modeler.generate_quality_data()
    modeler.build_regression_model('protein_content')
    modeler.build_classification_model('quality_grade')
    modeler.generate_model_summary()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 Check the following directories for results:")
    print("   - outputs/figures/          - Visualization plots")
    print("   - outputs/models/           - Model results")
    print("   - outputs/processed_data/   - Processed spectral data")
    print("   - outputs/reports/          - Analysis reports")
    print("\n🎉 Rice quality analysis pipeline finished!")

if __name__ == "__main__":
    main()