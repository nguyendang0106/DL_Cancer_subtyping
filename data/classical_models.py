"""
Classical Machine Learning Models for Baseline
Bao gồm: SVM, XGBoost, Logistic Regression with L1
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, 
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ClassicalMLModels:
    """
    Classical ML models cho cancer classification
    
    Lý do sử dụng classical models làm baseline:
    1. Interpretability: Dễ hiểu và giải thích hơn deep learning
    2. Performance: Với small-medium datasets, có thể tốt hơn DL
    3. Training time: Nhanh hơn nhiều
    4. Hyperparameter tuning: Ít parameters hơn
    5. Benchmark: So sánh với DL models
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
    def train_svm(self, X_train, y_train, param_grid=None):
        """
        Support Vector Machine (SVM)
        
        Công thức (Linear SVM):
        min_w,b [1/2 ||w||² + C Σ ξ_i]
        subject to: y_i(w^T x_i + b) ≥ 1 - ξ_i
        
        Decision function: f(x) = sign(w^T x + b)
        
        Lý do chọn SVM:
        - Hiệu quả với high-dimensional data (n_features >> n_samples)
        - Maximum margin classifier → good generalization
        - Kernel trick → non-linear boundaries
        - Robust to outliers (với soft margin)
        
        Kernels:
        - Linear: K(x,x') = x^T x' (cho linearly separable data)
        - RBF: K(x,x') = exp(-γ||x-x'||²) (non-linear, popular cho omics)
        
        Hyperparameters:
        - C: Regularization (trade-off giữa margin và errors)
        - gamma: RBF kernel width (only for RBF)
        """
        print("\n" + "="*60)
        print("Training SVM")
        print("="*60)
        
        if param_grid is None:
            param_grid = self.config.get('svm', {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            })
        
        # Create SVM model
        svm = SVC(
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            probability=True  # Enable probability estimates for ROC
        )
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            svm,
            param_grid,
            cv=cv,
            scoring='f1_macro',  # Macro F1 for multi-class
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['svm'] = grid_search.best_estimator_
        self.best_params['svm'] = grid_search.best_params_
        self.cv_scores['svm'] = grid_search.best_score_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        print("="*60)
        
        return self.models['svm']
    
    def train_xgboost(self, X_train, y_train, param_grid=None):
        """
        XGBoost (Extreme Gradient Boosting)
        
        Công thức:
        obj(θ) = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
        
        L: Loss function
        Ω(f) = γT + 1/2 λ||w||²: Regularization (tree complexity + L2)
        
        Boosting: ŷ^(t) = ŷ^(t-1) + f_t(x)
        
        Lý do chọn XGBoost:
        - State-of-the-art ensemble method
        - Handles complex non-linear relationships
        - Built-in regularization → prevents overfitting
        - Feature importance ranking
        - Handles missing values
        - Fast training với parallel processing
        
        Key hyperparameters:
        - n_estimators: Number of trees
        - max_depth: Tree depth (control overfitting)
        - learning_rate (eta): Step size shrinkage
        - subsample: Row sampling ratio
        - colsample_bytree: Feature sampling ratio
        - lambda, alpha: L2, L1 regularization
        """
        print("\n" + "="*60)
        print("Training XGBoost")
        print("="*60)
        
        if param_grid is None:
            param_grid = self.config.get('xgboost', {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            })
        
        # Handle multi-class
        num_classes = len(np.unique(y_train))
        if num_classes > 2:
            objective = 'multi:softprob'
            eval_metric = 'mlogloss'
        else:
            objective = 'binary:logistic'
            eval_metric = 'logloss'
        
        # Create XGBoost model
        xgb_model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False
        )
        
        # Grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['xgboost'] = grid_search.best_estimator_
        self.best_params['xgboost'] = grid_search.best_params_
        self.cv_scores['xgboost'] = grid_search.best_score_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        print("="*60)
        
        return self.models['xgboost']
    
    def train_logistic_l1(self, X_train, y_train, param_grid=None):
        """
        Logistic Regression with L1 Regularization
        
        Công thức:
        min_w [-Σ y_i log(p_i) + (1-y_i)log(1-p_i) + α||w||₁]
        
        p_i = σ(w^T x_i) = 1 / (1 + exp(-w^T x_i))
        
        L1 regularization: ||w||₁ = Σ |w_i|
        
        Lý do chọn Logistic Regression với L1:
        - Sparse solution: Many weights → 0 (automatic feature selection)
        - Interpretability: Linear model, easy to understand
        - Fast training
        - Probabilistic output
        - Works well với high-dimensional data
        
        Multi-class extension:
        - One-vs-Rest (OvR): K binary classifiers
        - Multinomial: Softmax với K classes
        
        Hyperparameter:
        - C = 1/α: Inverse regularization strength
          - Smaller C → stronger regularization → more sparsity
        """
        print("\n" + "="*60)
        print("Training Logistic Regression (L1)")
        print("="*60)
        
        if param_grid is None:
            param_grid = self.config.get('logistic_l1', {
                'C': [0.001, 0.01, 0.1, 1, 10]
            })
        
        # Create Logistic Regression model
        log_reg = LogisticRegression(
            penalty='l1',
            solver='saga',  # 'saga' supports L1 for multi-class
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            multi_class='multinomial'  # Better for multi-class
        )
        
        # Grid search
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            log_reg,
            param_grid,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['logistic_l1'] = grid_search.best_estimator_
        self.best_params['logistic_l1'] = grid_search.best_params_
        self.cv_scores['logistic_l1'] = grid_search.best_score_
        
        # Get feature sparsity
        coef = grid_search.best_estimator_.coef_
        if len(coef.shape) > 1:
            non_zero = np.sum(np.any(coef != 0, axis=0))
        else:
            non_zero = np.sum(coef != 0)
        sparsity = (1 - non_zero / coef.shape[-1]) * 100
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        print(f"Sparsity: {sparsity:.2f}% (features with zero coefficients)")
        print("="*60)
        
        return self.models['logistic_l1']
    
    def evaluate_model(self, model_name, X_test, y_test, class_names=None):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*60)
        print(f"Evaluating {model_name.upper()}")
        print("="*60)
        
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print("="*60)
        
        return results
    
    def plot_confusion_matrix(self, model_name, results, class_names=None, save_path=None):
        """
        Plot confusion matrix
        """
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names if class_names else range(len(cm)),
            yticklabels=class_names if class_names else range(len(cm))
        )
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, X_test, y_test):
        """
        So sánh tất cả models
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = []
        
        for model_name in self.models.keys():
            results = self.evaluate_model(model_name, X_test, y_test)
            
            comparison.append({
                'Model': model_name.upper(),
                'CV F1': self.cv_scores.get(model_name, 0),
                'Test Accuracy': results['accuracy'],
                'Test F1 (Macro)': results['f1_macro'],
                'Test F1 (Weighted)': results['f1_weighted'],
                'Precision': results['precision'],
                'Recall': results['recall']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Test F1 (Macro)', ascending=False)
        
        print("\n", comparison_df.to_string(index=False))
        print("="*60)
        
        return comparison_df


if __name__ == "__main__":
    """
    Example usage
    """
    from config import CLASSICAL_MODELS_CONFIG, DATASET_CONFIG
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = DATASET_CONFIG['n_samples']
    n_features = 1000  # After feature selection
    n_classes = DATASET_CONFIG['n_classes']
    
    # Simulate selected features (z-scored)
    X = np.random.randn(n_samples, n_features)
    
    # Make some features discriminative
    for i in range(n_classes):
        class_indices = np.arange(i * n_samples // n_classes, (i+1) * n_samples // n_classes)
        X[class_indices, i*20:(i+1)*20] += np.random.randn(len(class_indices), 20) * 2
    
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create classifier
    clf = ClassicalMLModels(CLASSICAL_MODELS_CONFIG)
    
    # Train models
    clf.train_svm(X_train, y_train)
    clf.train_xgboost(X_train, y_train)
    clf.train_logistic_l1(X_train, y_train)
    
    # Compare models
    comparison = clf.compare_models(X_test, y_test)
