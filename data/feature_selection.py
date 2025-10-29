"""
Feature Selection Module for Omics Data
Bao gồm: Variance-based, Statistical tests, L1 regularization, PCA
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    mutual_info_classif,
    VarianceThreshold
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selection cho high-dimensional omics data
    
    Lý do cần feature selection:
    - Curse of dimensionality: n_features >> n_samples
    - Nhiều features không liên quan (noise)
    - Giảm overfitting
    - Tăng interpretability
    - Tăng tốc độ training
    
    Methods:
    1. Variance-based: Loại bỏ low-variance features
    2. ANOVA F-test: Statistical significance
    3. Mutual Information: Non-linear dependencies
    4. L1 Regularization (Lasso): Sparse feature selection
    5. Random Forest Importance
    6. PCA: Dimensionality reduction (orthogonal features)
    """
    
    def __init__(self, config):
        self.config = config
        self.selected_features_indices = None
        self.feature_scores = {}
        self.pca = None
        
    def variance_based_selection(self, X, y, n_features=5000):
        """
        Chọn features dựa trên variance
        
        Công thức: Var(X_i) = E[(X_i - μ_i)²]
        
        Lý do:
        - Features với variance cao thường chứa nhiều thông tin hơn
        - Fast và simple
        - Unsupervised (không cần label)
        
        Limitation: Không xem xét correlation với target
        """
        print(f"\nVariance-based selection (keeping top {n_features} features)...")
        
        # Calculate variance for each feature
        variances = np.var(X, axis=0)
        
        # Get indices of top variance features
        top_indices = np.argsort(variances)[::-1][:n_features]
        
        self.feature_scores['variance'] = variances
        
        print(f"Selected {len(top_indices)} features with highest variance")
        print(f"Variance range: {variances[top_indices].min():.4f} - {variances[top_indices].max():.4f}")
        
        return top_indices
    
    def anova_f_test_selection(self, X, y, n_features=2000):
        """
        ANOVA F-test feature selection
        
        Công thức: F = (Between-group variability) / (Within-group variability)
        
        F = [Σ n_i(ȳ_i - ȳ)² / (k-1)] / [Σ Σ (y_ij - ȳ_i)² / (N-k)]
        
        Lý do:
        - Đo lường sự khác biệt expression giữa các cancer subtypes
        - Statistical significance
        - Fast computation
        
        Best for: Linear relationships, Gaussian distributed data
        """
        print(f"\nANOVA F-test selection (keeping top {n_features} features)...")
        
        # Use SelectKBest with f_classif
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        # Get F-scores
        f_scores = selector.scores_
        self.feature_scores['anova_f'] = f_scores
        
        print(f"Selected {len(selected_indices)} features")
        print(f"F-score range: {f_scores[selected_indices].min():.4f} - {f_scores[selected_indices].max():.4f}")
        
        return selected_indices
    
    def mutual_information_selection(self, X, y, n_features=1500):
        """
        Mutual Information feature selection
        
        Công thức: I(X;Y) = Σ Σ p(x,y) log[p(x,y) / (p(x)p(y))]
        
        Lý do:
        - Captures non-linear relationships
        - Không giả định phân phối
        - Đo lường dependency giữa feature và target
        
        Best for: Non-linear relationships, complex dependencies
        """
        print(f"\nMutual Information selection (keeping top {n_features} features)...")
        
        selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        mi_scores = selector.scores_
        self.feature_scores['mutual_info'] = mi_scores
        
        print(f"Selected {len(selected_indices)} features")
        print(f"MI score range: {mi_scores[selected_indices].min():.4f} - {mi_scores[selected_indices].max():.4f}")
        
        return selected_indices
    
    def lasso_selection(self, X, y, n_features=1000, alpha=0.01):
        """
        L1 Regularization (Lasso) feature selection
        
        Công thức: min_w [1/2n ||Xw - y||² + α||w||₁]
        
        ||w||₁ = Σ |w_i| (L1 norm)
        
        Lý do:
        - L1 regularization drives weights to exactly zero
        - Automatic feature selection
        - Considers feature interactions
        - Sparse solution
        
        Best for: Linear models, interpretability
        """
        print(f"\nLasso (L1) selection (target: {n_features} features, alpha={alpha})...")
        
        # Use Logistic Regression with L1 penalty
        lasso = LogisticRegression(
            penalty='l1',
            C=1.0/alpha,  # C = 1/α
            solver='liblinear',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        lasso.fit(X, y)
        
        # Get feature importances (absolute coefficients)
        # For multi-class, take mean across classes
        if len(lasso.coef_.shape) > 1:
            importances = np.mean(np.abs(lasso.coef_), axis=0)
        else:
            importances = np.abs(lasso.coef_)
        
        # Select top n_features
        selected_indices = np.argsort(importances)[::-1][:n_features]
        
        # Filter out zero coefficients
        selected_indices = selected_indices[importances[selected_indices] > 0]
        
        self.feature_scores['lasso'] = importances
        
        print(f"Selected {len(selected_indices)} features with non-zero coefficients")
        print(f"Coefficient range: {importances[selected_indices].min():.4f} - {importances[selected_indices].max():.4f}")
        
        return selected_indices
    
    def random_forest_selection(self, X, y, n_features=1000):
        """
        Random Forest feature importance
        
        Lý do:
        - Captures non-linear relationships
        - Considers feature interactions
        - Built-in feature importance (Gini importance)
        - Robust to outliers
        
        Công thức (Gini importance):
        Importance_i = Σ (decrease in impurity when splitting on feature i)
        """
        print(f"\nRandom Forest selection (keeping top {n_features} features)...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        selected_indices = np.argsort(importances)[::-1][:n_features]
        
        self.feature_scores['random_forest'] = importances
        
        print(f"Selected {len(selected_indices)} features")
        print(f"Importance range: {importances[selected_indices].min():.4f} - {importances[selected_indices].max():.4f}")
        
        return selected_indices
    
    def ensemble_selection(self, X, y, methods=['variance', 'anova', 'lasso'], voting='union'):
        """
        Ensemble feature selection: Kết hợp nhiều methods
        
        Voting strategies:
        - union: Lấy tất cả features được chọn bởi bất kỳ method nào
        - intersection: Chỉ lấy features được chọn bởi TẤT CẢ methods
        - majority: Lấy features được chọn bởi > 50% methods
        
        Lý do:
        - Robust hơn single method
        - Capture different aspects of feature importance
        - Reduce bias của từng method
        """
        print(f"\nEnsemble feature selection using {methods} with {voting} voting...")
        
        all_selected = []
        
        for method in methods:
            if method == 'variance':
                n_features = self.config.get('n_features_variance', 5000)
                indices = self.variance_based_selection(X, y, n_features)
            elif method == 'anova':
                n_features = self.config.get('n_features_anova', 2000)
                indices = self.anova_f_test_selection(X, y, n_features)
            elif method == 'lasso':
                n_features = self.config.get('n_features_lasso', 1000)
                indices = self.lasso_selection(X, y, n_features)
            elif method == 'mutual_info':
                n_features = self.config.get('n_features_mutual_info', 1500)
                indices = self.mutual_information_selection(X, y, n_features)
            elif method == 'random_forest':
                n_features = self.config.get('n_features_rf', 1000)
                indices = self.random_forest_selection(X, y, n_features)
            else:
                continue
            
            all_selected.append(set(indices))
        
        # Combine based on voting strategy
        if voting == 'union':
            final_indices = set.union(*all_selected)
        elif voting == 'intersection':
            final_indices = set.intersection(*all_selected)
        elif voting == 'majority':
            # Count how many methods selected each feature
            from collections import Counter
            feature_counts = Counter()
            for selected in all_selected:
                feature_counts.update(selected)
            # Keep features selected by majority
            threshold = len(methods) / 2
            final_indices = {feat for feat, count in feature_counts.items() if count > threshold}
        else:
            final_indices = set.union(*all_selected)
        
        final_indices = np.array(sorted(list(final_indices)))
        
        print(f"\nEnsemble selected {len(final_indices)} features")
        
        return final_indices
    
    def apply_pca(self, X, n_components=0.95):
        """
        Principal Component Analysis (PCA)
        
        Công thức:
        1. Standardize data: X_std
        2. Compute covariance matrix: C = (1/n)X_std^T X_std
        3. Eigen decomposition: C = VΛV^T
        4. Project: X_pca = X_std @ V[:, :k]
        
        Lý do:
        - Dimensionality reduction
        - Decorrelate features (orthogonal components)
        - Capture maximum variance
        - Reduce multicollinearity
        
        n_components: 
        - If int: number of components
        - If float (0,1): explained variance ratio
        """
        print(f"\nApplying PCA (n_components={n_components})...")
        
        if self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)
        
        print(f"Reduced to {X_pca.shape[1]} principal components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"Top 5 components explain: {self.pca.explained_variance_ratio_[:5].sum():.4f}")
        
        return X_pca
    
    def select_features(self, X, y, methods=None):
        """
        Main feature selection pipeline
        
        Returns:
        --------
        X_selected : Selected features
        selected_indices : Indices of selected features
        """
        print("\n" + "="*60)
        print("Feature Selection Pipeline")
        print("="*60)
        print(f"Input shape: {X.shape}")
        
        if methods is None:
            methods = self.config.get('methods', ['variance', 'anova', 'lasso'])
        
        # Ensemble selection
        selected_indices = self.ensemble_selection(X, y, methods=methods, voting='union')
        
        X_selected = X[:, selected_indices]
        self.selected_features_indices = selected_indices
        
        # Optional: Apply PCA
        if self.config.get('use_pca', False):
            n_components = self.config.get('pca_components', 0.95)
            X_selected = self.apply_pca(X_selected, n_components)
        
        print(f"\nFinal shape: {X_selected.shape}")
        print("="*60 + "\n")
        
        return X_selected, selected_indices
    
    def get_top_features(self, feature_names, n_top=50):
        """
        Lấy danh sách top features quan trọng nhất
        """
        if self.selected_features_indices is None:
            print("Please run select_features first!")
            return None
        
        results = {}
        
        for method, scores in self.feature_scores.items():
            top_indices = np.argsort(scores)[::-1][:n_top]
            top_features = [(feature_names[i], scores[i]) for i in top_indices]
            results[method] = top_features
        
        return results


if __name__ == "__main__":
    """
    Example usage
    """
    from config import FEATURE_SELECTION_CONFIG, DATASET_CONFIG
    
    # Generate synthetic preprocessed data
    np.random.seed(42)
    n_samples = DATASET_CONFIG['n_samples']
    n_features = 5000  # After preprocessing
    n_classes = DATASET_CONFIG['n_classes']
    
    # Simulate preprocessed data (z-scored)
    X = np.random.randn(n_samples, n_features)
    
    # Make some features more informative
    for i in range(n_classes):
        class_indices = np.arange(i * n_samples // n_classes, (i+1) * n_samples // n_classes)
        # First 100 features are class-specific
        X[class_indices, i*20:(i+1)*20] += np.random.randn(len(class_indices), 20) * 3
    
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Create feature selector
    selector = FeatureSelector(FEATURE_SELECTION_CONFIG)
    
    # Select features
    X_selected, selected_indices = selector.select_features(X, y)
    
    print("\nFeature selection completed!")
    print(f"Original features: {X.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Reduction: {(1 - X_selected.shape[1]/X.shape[1])*100:.2f}%")
