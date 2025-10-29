"""
Main Pipeline: End-to-End Cancer Classification on Omics Data

Pipeline steps:
1. Data Loading & Preprocessing
2. Feature Selection
3. Train Classical ML Models (Baseline)
4. Train Deep Learning Models
5. Model Evaluation & Comparison
6. Model Interpretation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import *
from data_preprocessing import OmicsPreprocessor, load_omics_data
from feature_selection import FeatureSelector
from classical_models import ClassicalMLModels
from deep_learning_models import DeepLearningTrainer
from model_interpretation import ModelInterpreter


class OmicsCancerClassificationPipeline:
    """
    Complete pipeline for cancer classification on omics data
    """
    
    def __init__(self, config_dict=None):
        """
        Initialize pipeline with configuration
        
        Parameters:
        -----------
        config_dict : dict, optional
            Configuration dictionary. If None, use default from config.py
        """
        if config_dict is None:
            self.dataset_config = DATASET_CONFIG
            self.preprocessing_config = PREPROCESSING_CONFIG
            self.feature_config = FEATURE_SELECTION_CONFIG
            self.classical_config = CLASSICAL_MODELS_CONFIG
            self.dl_config = DL_CONFIG
            self.eval_config = EVALUATION_CONFIG
            self.interp_config = INTERPRETATION_CONFIG
        else:
            self.dataset_config = config_dict.get('dataset', DATASET_CONFIG)
            self.preprocessing_config = config_dict.get('preprocessing', PREPROCESSING_CONFIG)
            self.feature_config = config_dict.get('feature_selection', FEATURE_SELECTION_CONFIG)
            self.classical_config = config_dict.get('classical_models', CLASSICAL_MODELS_CONFIG)
            self.dl_config = config_dict.get('deep_learning', DL_CONFIG)
            self.eval_config = config_dict.get('evaluation', EVALUATION_CONFIG)
            self.interp_config = config_dict.get('interpretation', INTERPRETATION_CONFIG)
        
        # Initialize components
        self.preprocessor = None
        self.feature_selector = None
        self.classical_models = None
        self.dl_trainer = None
        
        # Data storage
        self.X_raw = None
        self.y = None
        self.feature_names = None
        self.sample_ids = None
        self.class_names = None
        
        self.X_processed = None
        self.X_selected = None
        self.selected_feature_indices = None
        
        # Train/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results
        self.results = {}
    
    def load_data(self, data_path=None, X=None, y=None, feature_names=None):
        """
        Load omics data
        
        Parameters:
        -----------
        data_path : str, optional
            Path to data file (CSV/TSV)
        X : array-like, optional
            Feature matrix (n_samples, n_features)
        y : array-like, optional
            Labels (n_samples,)
        feature_names : list, optional
            Feature names (genes/proteins)
        """
        print("\n" + "="*70)
        print(" "*20 + "DATA LOADING")
        print("="*70)
        
        if data_path is not None:
            # Load from file
            self.X_raw, self.y, self.feature_names, self.sample_ids = load_omics_data(
                data_path, self.dataset_config
            )
        elif X is not None and y is not None:
            # Use provided data
            self.X_raw = X
            self.y = y
            self.feature_names = feature_names if feature_names else \
                [f"Feature_{i}" for i in range(X.shape[1])]
            self.sample_ids = [f"Sample_{i}" for i in range(X.shape[0])]
        else:
            raise ValueError("Either data_path or (X, y) must be provided")
        
        # Get class names
        self.class_names = [f"Cancer_Type_{i}" for i in range(len(np.unique(self.y)))]
        
        print(f"✓ Data loaded successfully")
        print(f"  Samples: {self.X_raw.shape[0]}")
        print(f"  Features: {self.X_raw.shape[1]}")
        print(f"  Classes: {len(np.unique(self.y))}")
        print("="*70 + "\n")
    
    def preprocess_data(self):
        """
        Preprocess omics data
        """
        print("\n" + "="*70)
        print(" "*20 + "DATA PREPROCESSING")
        print("="*70)
        
        self.preprocessor = OmicsPreprocessor(self.preprocessing_config)
        self.X_processed = self.preprocessor.preprocess(
            self.X_raw, self.y, is_train=True
        )
        
        print(f"✓ Preprocessing completed")
        print(f"  Input shape: {self.X_raw.shape}")
        print(f"  Output shape: {self.X_processed.shape}")
        print("="*70 + "\n")
    
    def select_features(self):
        """
        Feature selection
        """
        print("\n" + "="*70)
        print(" "*20 + "FEATURE SELECTION")
        print("="*70)
        
        self.feature_selector = FeatureSelector(self.feature_config)
        self.X_selected, self.selected_feature_indices = self.feature_selector.select_features(
            self.X_processed, self.y
        )
        
        # Update feature names
        if self.feature_names is not None:
            original_feature_names = np.array(self.feature_names)
            
            # Get mask from preprocessor (variance filtering)
            if self.preprocessor.get_feature_mask() is not None:
                mask = self.preprocessor.get_feature_mask()
                filtered_names = original_feature_names[mask]
            else:
                filtered_names = original_feature_names
            
            # Get selected feature names
            self.selected_feature_names = filtered_names[self.selected_feature_indices].tolist()
        else:
            self.selected_feature_names = [f"Feature_{i}" for i in self.selected_feature_indices]
        
        print(f"✓ Feature selection completed")
        print(f"  Input features: {self.X_processed.shape[1]}")
        print(f"  Selected features: {self.X_selected.shape[1]}")
        print(f"  Reduction: {(1 - self.X_selected.shape[1]/self.X_processed.shape[1])*100:.2f}%")
        print("="*70 + "\n")
    
    def split_data(self):
        """
        Train-test split
        """
        print("\n" + "="*70)
        print(" "*20 + "TRAIN-TEST SPLIT")
        print("="*70)
        
        test_size = self.dataset_config.get('test_size', 0.2)
        random_state = self.dataset_config.get('random_state', 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_selected, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=random_state
        )
        
        print(f"✓ Data split completed")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        print(f"  Test ratio: {test_size*100:.1f}%")
        print("="*70 + "\n")
    
    def train_classical_models(self):
        """
        Train classical ML models (baseline)
        """
        print("\n" + "="*70)
        print(" "*20 + "CLASSICAL ML MODELS (BASELINE)")
        print("="*70)
        
        self.classical_models = ClassicalMLModels(self.classical_config)
        
        # Train SVM
        print("\n[1/3] Training SVM...")
        self.classical_models.train_svm(self.X_train, self.y_train)
        
        # Train XGBoost
        print("\n[2/3] Training XGBoost...")
        self.classical_models.train_xgboost(self.X_train, self.y_train)
        
        # Train Logistic Regression with L1
        print("\n[3/3] Training Logistic Regression (L1)...")
        self.classical_models.train_logistic_l1(self.X_train, self.y_train)
        
        # Evaluate all models
        print("\n" + "-"*70)
        print("Evaluating Classical Models on Test Set")
        print("-"*70)
        
        comparison = self.classical_models.compare_models(self.X_test, self.y_test)
        self.results['classical_comparison'] = comparison
        
        print("="*70 + "\n")
    
    def train_deep_learning_models(self):
        """
        Train deep learning models
        """
        print("\n" + "="*70)
        print(" "*20 + "DEEP LEARNING MODELS")
        print("="*70)
        
        # Further split train into train/val
        X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
            self.X_train, self.y_train,
            test_size=0.2,
            stratify=self.y_train,
            random_state=42
        )
        
        self.dl_trainer = DeepLearningTrainer(self.dl_config)
        
        # Train MLP
        print("\n[1/2] Training MLP Classifier...")
        mlp_model = self.dl_trainer.train_mlp(
            X_train_dl, y_train_dl, X_val_dl, y_val_dl
        )
        
        # Evaluate MLP
        y_pred_mlp, y_proba_mlp = self.dl_trainer.predict(self.X_test)
        
        from sklearn.metrics import accuracy_score, f1_score
        mlp_acc = accuracy_score(self.y_test, y_pred_mlp)
        mlp_f1 = f1_score(self.y_test, y_pred_mlp, average='macro')
        
        print(f"\nMLP Test Results:")
        print(f"  Accuracy: {mlp_acc:.4f}")
        print(f"  Macro F1: {mlp_f1:.4f}")
        
        self.results['mlp'] = {
            'accuracy': mlp_acc,
            'f1_macro': mlp_f1,
            'predictions': y_pred_mlp,
            'probabilities': y_proba_mlp
        }
        
        # Train Autoencoder + Classifier
        print("\n[2/2] Training Autoencoder + Classifier...")
        ae_model = self.dl_trainer.train_autoencoder_classifier(
            X_train_dl, y_train_dl, X_val_dl, y_val_dl
        )
        
        # Evaluate Autoencoder Classifier
        y_pred_ae, y_proba_ae = self.dl_trainer.predict(self.X_test)
        
        ae_acc = accuracy_score(self.y_test, y_pred_ae)
        ae_f1 = f1_score(self.y_test, y_pred_ae, average='macro')
        
        print(f"\nAutoencoder + Classifier Test Results:")
        print(f"  Accuracy: {ae_acc:.4f}")
        print(f"  Macro F1: {ae_f1:.4f}")
        
        self.results['autoencoder'] = {
            'accuracy': ae_acc,
            'f1_macro': ae_f1,
            'predictions': y_pred_ae,
            'probabilities': y_proba_ae
        }
        
        # Plot training history
        print("\nPlotting training history...")
        self.dl_trainer.plot_training_history()
        
        print("="*70 + "\n")
    
    def compare_all_models(self):
        """
        Compare all models (classical + DL)
        """
        print("\n" + "="*70)
        print(" "*20 + "MODEL COMPARISON (ALL MODELS)")
        print("="*70)
        
        # Compile results
        all_results = []
        
        # Classical models
        if 'classical_comparison' in self.results:
            for _, row in self.results['classical_comparison'].iterrows():
                all_results.append({
                    'Model': row['Model'],
                    'Type': 'Classical ML',
                    'Test Accuracy': row['Test Accuracy'],
                    'Test F1 (Macro)': row['Test F1 (Macro)'],
                    'Precision': row['Precision'],
                    'Recall': row['Recall']
                })
        
        # Deep learning models
        if 'mlp' in self.results:
            all_results.append({
                'Model': 'MLP',
                'Type': 'Deep Learning',
                'Test Accuracy': self.results['mlp']['accuracy'],
                'Test F1 (Macro)': self.results['mlp']['f1_macro'],
                'Precision': '-',
                'Recall': '-'
            })
        
        if 'autoencoder' in self.results:
            all_results.append({
                'Model': 'Autoencoder + Classifier',
                'Type': 'Deep Learning',
                'Test Accuracy': self.results['autoencoder']['accuracy'],
                'Test F1 (Macro)': self.results['autoencoder']['f1_macro'],
                'Precision': '-',
                'Recall': '-'
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        comparison_df = comparison_df.sort_values('Test F1 (Macro)', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
        
        print("\n" + "="*70 + "\n")
        
        return comparison_df
    
    def interpret_best_model(self):
        """
        Interpret best performing model
        """
        print("\n" + "="*70)
        print(" "*20 + "MODEL INTERPRETATION")
        print("="*70)
        
        # Use XGBoost as example (typically best classical model)
        if self.classical_models and 'xgboost' in self.classical_models.models:
            model = self.classical_models.models['xgboost']
            
            interpreter = ModelInterpreter(model, self.selected_feature_names)
            
            # SHAP analysis
            if self.interp_config.get('use_shap', True):
                shap_values, explainer = interpreter.shap_analysis(
                    self.X_train, self.X_test, model_type='tree'
                )
                
                # Plot SHAP summary
                interpreter.plot_shap_summary(
                    self.X_test, 
                    max_display=self.interp_config.get('top_n_features', 50)
                )
            
            # Permutation importance
            perm_importance = interpreter.permutation_importance_analysis(
                model, self.X_test, self.y_test
            )
            
            # Get top genes
            top_genes_df = interpreter.get_top_genes(
                n_top=self.interp_config.get('top_n_features', 50)
            )
            
            print("\nTop 10 Important Genes/Features:")
            print(top_genes_df.head(10).to_string(index=False))
            
            # Save results
            if self.interp_config.get('save_feature_importance', True):
                top_genes_df.to_csv('top_important_features.csv', index=False)
                print("\n✓ Top features saved to 'top_important_features.csv'")
            
            self.results['interpretation'] = {
                'top_genes': top_genes_df,
                'interpreter': interpreter
            }
        
        print("="*70 + "\n")
    
    def _plot_model_comparison(self, comparison_df):
        """
        Plot model comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: F1 scores
        models = comparison_df['Model'].values
        f1_scores = comparison_df['Test F1 (Macro)'].values
        colors = ['skyblue' if t == 'Classical ML' else 'salmon' 
                  for t in comparison_df['Type'].values]
        
        axes[0].barh(models, f1_scores, color=colors)
        axes[0].set_xlabel('Macro F1 Score')
        axes[0].set_title('Model Comparison - F1 Score')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(f1_scores):
            axes[0].text(v + 0.005, i, f'{v:.4f}', va='center')
        
        # Plot 2: Accuracy
        accuracy = comparison_df['Test Accuracy'].values
        
        axes[1].barh(models, accuracy, color=colors)
        axes[1].set_xlabel('Accuracy')
        axes[1].set_title('Model Comparison - Accuracy')
        axes[1].grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(accuracy):
            axes[1].text(v + 0.005, i, f'{v:.4f}', va='center')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Classical ML'),
            Patch(facecolor='salmon', label='Deep Learning')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Model comparison plot saved to 'model_comparison.png'")
    
    def run_full_pipeline(self, data_path=None, X=None, y=None, feature_names=None):
        """
        Run complete pipeline
        
        Parameters:
        -----------
        data_path : str, optional
            Path to data file
        X, y : array-like, optional
            Feature matrix and labels
        feature_names : list, optional
            Feature names
        """
        print("\n" + "="*70)
        print(" "*15 + "OMICS CANCER CLASSIFICATION PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        self.load_data(data_path, X, y, feature_names)
        
        # Step 2: Preprocess
        self.preprocess_data()
        
        # Step 3: Feature selection
        self.select_features()
        
        # Step 4: Train-test split
        self.split_data()
        
        # Step 5: Train classical models
        self.train_classical_models()
        
        # Step 6: Train DL models
        self.train_deep_learning_models()
        
        # Step 7: Compare all models
        comparison_df = self.compare_all_models()
        
        # Step 8: Interpret best model
        self.interpret_best_model()
        
        print("\n" + "="*70)
        print(" "*20 + "PIPELINE COMPLETED!")
        print("="*70)
        print("\n✓ All steps completed successfully")
        print("✓ Results saved to:")
        print("  - model_comparison.png")
        print("  - top_important_features.csv")
        print("  - best_mlp_model.pth (if DL models trained)")
        print("="*70 + "\n")
        
        return comparison_df, self.results


if __name__ == "__main__":
    """
    Example usage with synthetic data
    """
    # Generate synthetic omics data
    np.random.seed(42)
    
    n_samples = 800
    n_features = 5000  # Reduced from 20000 for faster demo
    n_classes = 5
    
    print("Generating synthetic omics data...")
    print(f"  Samples: {n_samples}")
    print(f"  Features (genes): {n_features}")
    print(f"  Cancer types: {n_classes}")
    
    # Simulate RNA-seq like data (negative binomial counts)
    X = np.random.negative_binomial(10, 0.5, size=(n_samples, n_features)).astype(float)
    
    # Make some features discriminative
    for i in range(n_classes):
        class_indices = np.arange(i * n_samples // n_classes, (i+1) * n_samples // n_classes)
        # First 100 features per class are informative
        X[class_indices, i*100:(i+1)*100] *= np.random.uniform(2, 5, size=(len(class_indices), 100))
    
    # Generate labels
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Feature names
    feature_names = [f"GENE_{i}" for i in range(n_features)]
    
    # Run pipeline
    pipeline = OmicsCancerClassificationPipeline()
    comparison_df, results = pipeline.run_full_pipeline(X=X, y=y, feature_names=feature_names)
    
    print("\n" + "="*70)
    print("Demo completed! Check the output files and plots.")
    print("="*70)
