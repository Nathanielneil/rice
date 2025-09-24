#!/usr/bin/env python3
"""
Rice - Quick Start Example
Simplified demo of key functionalities
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import NIRPreprocessor
from analysis import NIRAnalyzer

def main():
    """Quick start demo"""
    print("🚀 RICE - NIR QUALITY ANALYSIS TOOLKIT")
    print("=" * 50)
    
    # Quick preprocessing demo
    print("\n1️⃣  Data Preprocessing Demo")
    preprocessor = NIRPreprocessor()
    preprocessor.load_data()
    results = preprocessor.apply_preprocessing(['msc', 'snv'])
    if results:
        print(f"   ✅ Applied {len(results)} preprocessing methods")
    
    # Quick analysis demo
    print("\n2️⃣  Spectral Analysis Demo")
    analyzer = NIRAnalyzer()
    analyzer.load_data()
    peaks = analyzer.detect_peaks()
    outliers = analyzer.detect_outliers()
    clustering = analyzer.cluster_analysis()
    
    print(f"   ✅ Detected {len(peaks)} spectral peaks")
    print(f"   ✅ Outlier detection completed")
    print(f"   ✅ Clustering analysis completed")
    
    print("\n🎯 Quick demo completed!")
    print("💡 Run 'python examples/run_full_analysis.py' for complete analysis")

if __name__ == "__main__":
    main()