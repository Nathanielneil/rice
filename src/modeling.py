#!/usr/bin/env python3
"""
Rice Quality Prediction Modeling Module
Simplified modeling with PyNIR integration for quality parameter prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pynir import Calibration
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set English font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class RiceQualityModeler:
    """Rice quality prediction modeling with PyNIR integration"""
    
    def __init__(self, data_file="data/rice_spectra.xlsx"):
        self.data_file = data_file
        self.spectral_data = None
        self.quality_data = None
        self.wavelengths = None
        self.models = {}
        
    def load_spectral_data(self):
        """Load spectral data"""
        print("Loading spectral data for modeling...")
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
    
    def generate_quality_data(self):
        """Generate simulated quality data for demonstration"""
        if self.spectral_data is None:
            return
        
        n_samples = len(self.spectral_data)
        np.random.seed(42)
        
        # Simulate quality parameters
        quality_data = {
            'sample_id': range(1, n_samples + 1),
            'protein_content': np.random.normal(7.5, 1.2, n_samples),
            'moisture_content': np.random.normal(14.0, 1.5, n_samples),
            'amylose_content': np.random.normal(18.0, 3.0, n_samples),
        }
        
        # Ensure reasonable ranges
        quality_data['protein_content'] = np.clip(quality_data['protein_content'], 5, 12)
        quality_data['moisture_content'] = np.clip(quality_data['moisture_content'], 10, 18)
        quality_data['amylose_content'] = np.clip(quality_data['amylose_content'], 10, 30)
        
        # Create quality grades
        composite_score = (
            quality_data['protein_content'] * 0.3 +
            (20 - quality_data['moisture_content']) * 0.2 +
            quality_data['amylose_content'] * 0.3 +
            np.random.normal(20, 5, n_samples) * 0.2  # Additional quality factor
        )
        
        quality_grades = []
        for score in composite_score:
            if score >= np.percentile(composite_score, 75):
                quality_grades.append('Premium')
            elif score >= np.percentile(composite_score, 25):
                quality_grades.append('Standard')
            else:
                quality_grades.append('Basic')
        
        quality_data['quality_grade'] = quality_grades
        self.quality_data = pd.DataFrame(quality_data)
        print(f"Generated quality data for {n_samples} samples")
    
    def build_regression_model(self, target='protein_content'):
        """Build regression model for continuous quality parameters"""
        if self.spectral_data is None or self.quality_data is None:
            return
        
        X = self.spectral_data
        y = self.quality_data[target].values
        
        # Feature selection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.3, random_state=42
        )
        
        results = {}
        
        # PyNIR PLS Regression
        try:
            pls_obj = Calibration.pls(n_components=10)
            pls_obj.fit(X_train, y_train)
            y_pred_pls = pls_obj.predict(X_test)
            
            pls_r2 = r2_score(y_test, y_pred_pls)
            pls_rmse = np.sqrt(mean_squared_error(y_test, y_pred_pls))
            
            results['PyNIR PLS'] = {
                'r2': pls_r2,
                'rmse': pls_rmse,
                'predictions': y_pred_pls
            }
            print(f"PyNIR PLS - R²: {pls_r2:.3f}, RMSE: {pls_rmse:.3f}")
            
        except Exception as e:
            print(f"PyNIR PLS failed: {e}")
        
        # Random Forest (for comparison)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        rf_r2 = r2_score(y_test, y_pred_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        
        results['Random Forest'] = {
            'r2': rf_r2,
            'rmse': rf_rmse,
            'predictions': y_pred_rf
        }
        print(f"Random Forest - R²: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}")
        
        # Store results
        self.models[f'{target}_regression'] = {
            'results': results,
            'y_test': y_test,
            'best_model': max(results.keys(), key=lambda k: results[k]['r2'])
        }
        
        return results
    
    def build_classification_model(self, target='quality_grade'):
        """Build classification model for categorical quality grades"""
        if self.spectral_data is None or self.quality_data is None:
            return
        
        X = self.spectral_data
        y = self.quality_data[target].values
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Feature selection
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        results = {}
        
        # PyNIR PLSDA Classification
        try:
            plsda_obj = Calibration.plsda(n_components=5)
            plsda_obj.fit(X_train, y_train)
            y_pred_plsda = plsda_obj.predict(X_test)
            
            plsda_acc = accuracy_score(y_test, y_pred_plsda)
            results['PyNIR PLSDA'] = {
                'accuracy': plsda_acc,
                'predictions': y_pred_plsda
            }
            print(f"PyNIR PLSDA - Accuracy: {plsda_acc:.3f}")
            
        except Exception as e:
            print(f"PyNIR PLSDA failed: {e}")
        
        # Random Forest (for comparison)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        rf_acc = accuracy_score(y_test, y_pred_rf)
        results['Random Forest'] = {
            'accuracy': rf_acc,
            'predictions': y_pred_rf
        }
        print(f"Random Forest - Accuracy: {rf_acc:.3f}")
        
        # Store results
        self.models[f'{target}_classification'] = {
            'results': results,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'best_model': max(results.keys(), key=lambda k: results[k]['accuracy'])
        }
        
        return results
    
    def generate_model_summary(self, output_dir="outputs/models"):
        """Generate model performance summary"""
        if not self.models:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create performance comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Regression performance
        reg_models = [k for k in self.models.keys() if 'regression' in k]
        if reg_models:
            model_info = self.models[reg_models[0]]
            models = list(model_info['results'].keys())
            r2_scores = [model_info['results'][m]['r2'] for m in models]
            
            axes[0].bar(models, r2_scores, alpha=0.7)
            axes[0].set_ylabel('R² Score')
            axes[0].set_title('Regression Model Performance')
            axes[0].set_xticklabels(models, rotation=45)
            for i, v in enumerate(r2_scores):
                axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Classification performance
        cls_models = [k for k in self.models.keys() if 'classification' in k]
        if cls_models:
            model_info = self.models[cls_models[0]]
            models = list(model_info['results'].keys())
            acc_scores = [model_info['results'][m]['accuracy'] for m in models]
            
            axes[1].bar(models, acc_scores, alpha=0.7, color='orange')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Classification Model Performance')
            axes[1].set_xticklabels(models, rotation=45)
            for i, v in enumerate(acc_scores):
                axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        filepath = output_path / "model_performance.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved model performance: {filepath}")
        
        # Save summary to CSV
        summary_data = []
        for model_type, model_info in self.models.items():
            for model_name, metrics in model_info['results'].items():
                if 'regression' in model_type:
                    summary_data.append({
                        'Task': model_type,
                        'Model': model_name,
                        'Type': 'Regression',
                        'R²': metrics['r2'],
                        'RMSE': metrics['rmse'],
                        'Accuracy': None
                    })
                else:
                    summary_data.append({
                        'Task': model_type,
                        'Model': model_name,
                        'Type': 'Classification',
                        'R²': None,
                        'RMSE': None,
                        'Accuracy': metrics['accuracy']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        csv_filepath = output_path / "model_summary.csv"
        summary_df.to_csv(csv_filepath, index=False)
        print(f"Saved model summary: {csv_filepath}")

def main():
    """Main modeling pipeline"""
    # Initialize modeler
    modeler = RiceQualityModeler()
    
    # Load data and build models
    modeler.load_spectral_data()
    modeler.generate_quality_data()
    
    # Build regression models
    modeler.build_regression_model('protein_content')
    
    # Build classification models
    modeler.build_classification_model('quality_grade')
    
    # Generate summary
    modeler.generate_model_summary()
    print("\nQuality prediction modeling completed successfully!")

if __name__ == "__main__":
    main()