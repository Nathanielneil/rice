# Rice - NIR Quality Analysis Toolkit

A comprehensive Near-Infrared (NIR) spectroscopy analysis toolkit for rice quality assessment, featuring PyNIR integration and machine learning capabilities.

## 🎯 Features

- **PyNIR Integration** - Native support for PyNIR preprocessing methods (MSC, SNV, derivatives)
- **Comprehensive Analysis** - Peak detection, outlier analysis, clustering, and quality prediction
- **Machine Learning** - PLS regression, PLSDA classification, and traditional ML models
- **Professional Visualizations** - High-quality plots with English labels (no font issues)
- **Modular Design** - Clean, maintainable code structure

## 📁 Project Structure

```
├── README.md                          # This file
├── data/                              # Input data
│   └── rice_spectra.xlsx             # NIR spectral data
├── src/                               # Source code
│   ├── preprocessing.py               # Data preprocessing with PyNIR
│   ├── analysis.py                    # Spectral analysis
│   └── modeling.py                    # Quality prediction modeling
├── examples/                          # Usage examples
│   ├── run_full_analysis.py           # Complete analysis pipeline
│   └── quick_start.py                 # Quick demo
└── outputs/                           # Results and outputs
    ├── figures/                       # Visualization plots
    ├── models/                        # Model results
    ├── processed_data/                # Processed spectral data
    └── reports/                       # Analysis reports
```

## 🚀 Quick Start

### 1. Quick Demo (2 minutes)
```bash
cd /home/daniel/NGW/data/rice
python examples/quick_start.py
```

### 2. Full Analysis Pipeline (5-10 minutes)
```bash
python examples/run_full_analysis.py
```

### 3. Individual Modules
```bash
# Data preprocessing only
python src/preprocessing.py

# Spectral analysis only
python src/analysis.py

# Quality modeling only
python src/modeling.py
```

## 📊 Analysis Workflow

1. **Data Preprocessing** 🔄
   - Load NIR spectral data from Excel file
   - Apply PyNIR preprocessing methods (MSC, SNV, derivatives)
   - Generate comparison visualizations
   - Export processed data

2. **Spectral Analysis** 📈
   - Peak detection and characterization
   - Outlier detection using multiple methods (PyNIR PLS, Isolation Forest, Mahalanobis)
   - Clustering analysis (K-means, DBSCAN)
   - Generate comprehensive analysis report

3. **Quality Prediction** 🎯
   - PyNIR PLS regression for continuous parameters (protein, moisture, amylose content)
   - PyNIR PLSDA classification for quality grades
   - Model performance comparison and validation
   - Export model results and predictions

## 🛠️ Technical Details

### PyNIR Integration
- **Preprocessing**: MSC, SNV, derivatives using PyNIR's sklearn-style API
- **Outlier Detection**: PLS-based outlier detection
- **Modeling**: PLS regression and PLSDA classification

### Traditional ML Methods
- **Clustering**: K-means, DBSCAN with silhouette analysis
- **Outlier Detection**: Isolation Forest, Mahalanobis distance
- **Modeling**: Random Forest, SVM, Neural Networks for comparison

### Visualization Features
- **English Labels**: All plots use English labels (no Chinese font issues)
- **High Quality**: 300 DPI publication-ready figures
- **Comprehensive**: Multi-panel plots showing different analysis aspects

## 📈 Output Files

### Figures (`outputs/figures/`)
- `preprocessing_comparison.png` - Preprocessing method comparison
- `analysis_summary.png` - Complete spectral analysis overview
- `model_performance.png` - Model performance comparison

### Data (`outputs/processed_data/`)
- `original_data.csv` - Raw spectral data
- `msc_data.csv` - MSC corrected spectra
- `snv_data.csv` - SNV transformed spectra
- `first_derivative_data.csv` - First derivative spectra
- `second_derivative_data.csv` - Second derivative spectra

### Models (`outputs/models/`)
- `model_summary.csv` - Model performance metrics
- `model_performance.png` - Performance comparison plots

### Reports (`outputs/reports/`)
- `analysis_results.csv` - Analysis summary statistics
- `analysis_summary.png` - Comprehensive analysis visualization

## 🔧 Requirements

- Python 3.7+
- PyNIR package
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## 💡 Usage Tips

1. **Font Issues**: All visualizations use English labels to avoid font display problems
2. **Modular Design**: Each module can be run independently or as part of the full pipeline
3. **Data Format**: Expects Excel file with wavelengths in first column, spectra in subsequent columns
4. **Customization**: Modify parameters in each module for specific analysis needs

## 📚 Key Capabilities

- ✅ PyNIR native preprocessing methods
- ✅ Multi-method outlier detection
- ✅ Advanced clustering analysis
- ✅ PLS regression and classification
- ✅ Model performance validation
- ✅ Professional visualization output
- ✅ Modular and extensible design
- ✅ Complete documentation

## 🎉 Getting Started

Run the quick start example to see the toolkit in action:

```bash
python examples/quick_start.py
```

Then explore the full analysis pipeline:

```bash
python examples/run_full_analysis.py
```

Check the `outputs/` directory for all generated results!

---

*Rice toolkit provides a complete solution for NIR-based rice quality analysis, combining the power of PyNIR with modern machine learning techniques in a clean, professional package.*