"""
Data Preprocessing Module for Omics Cancer Classification
Bao gồm: normalization, batch correction, missing value handling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')


class OmicsPreprocessor:
    """
    Tiền xử lý dữ liệu omics với các phương pháp chuyên biệt
    
    Lý do thiết kế:
    - Dữ liệu omics thường có phân phối lệch (right-skewed) → cần log transformation
    - Batch effects do kỹ thuật đo lường khác nhau → cần batch correction
    - Missing values do giới hạn detection → cần imputation thông minh
    - High-dimensional với nhiều features không thông tin → variance filtering
    """
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.variance_selector = None
        self.imputer = None
        
    def log2_transform(self, X):
        """
        Log2 transformation cho RNA-seq data
        
        Công thức: log2(X + 1)
        Lý do: 
        - Giảm skewness của expression data
        - Ổn định variance
        - +1 để tránh log(0)
        """
        print("Applying log2 transformation...")
        return np.log2(X + 1)
    
    def tpm_normalization(self, X, gene_lengths=None):
        """
        Transcripts Per Million (TPM) normalization
        
        Công thức:
        1. TPM_i = (reads_i / gene_length_i) / sum(reads_j / gene_length_j) * 10^6
        
        Lý do: Chuẩn hóa cho library size và gene length
        """
        if gene_lengths is not None:
            # Normalize by gene length first
            rpk = X / gene_lengths
            # Then normalize by library size
            tpm = rpk / rpk.sum(axis=1, keepdims=True) * 1e6
            return tpm
        else:
            # If no gene lengths, just library size normalization (CPM)
            return X / X.sum(axis=1, keepdims=True) * 1e6
    
    def zscore_normalization(self, X):
        """
        Z-score normalization (standardization)
        
        Công thức: z = (x - μ) / σ
        
        Lý do:
        - Đưa features về cùng scale (mean=0, std=1)
        - Quan trọng cho ML models như SVM, neural networks
        """
        print("Applying Z-score normalization...")
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = self.scaler.transform(X)
        return X_normalized
    
    def quantile_normalization(self, X):
        """
        Quantile normalization
        
        Lý do:
        - Đảm bảo phân phối giống nhau giữa các samples
        - Loại bỏ technical variation
        - Thường dùng cho microarray data
        """
        print("Applying quantile normalization...")
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        return qt.fit_transform(X)
    
    def combat_batch_correction(self, X, batch_labels):
        """
        ComBat batch effect correction
        
        Công thức: Empirical Bayes framework
        X_corrected = (X - γ_batch) / δ_batch
        
        Lý do:
        - Loại bỏ systematic technical variation giữa các batch
        - Giữ lại biological variation
        
        Note: Cần thư viện 'combat' hoặc 'pycombat'
        """
        try:
            from combat.pycombat import pycombat
            print("Applying ComBat batch correction...")
            X_corrected = pycombat(X.T, batch_labels).T
            return X_corrected
        except ImportError:
            print("Warning: pycombat not installed. Skipping batch correction.")
            print("Install: pip install combat")
            return X
    
    def handle_missing_values(self, X, method='mean'):
        """
        Xử lý missing values
        
        Methods:
        - mean: Thay thế bằng mean của feature
        - median: Thay thế bằng median (robust hơn với outliers)
        - knn: K-Nearest Neighbors imputation (giữ được structure)
        
        Lý do:
        - Omics data thường có missing values do detection limits
        - KNN imputation tốt hơn cho biological data vì features có correlation
        """
        print(f"Handling missing values using {method} imputation...")
        
        if method in ['mean', 'median']:
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy=method)
                X_imputed = self.imputer.fit_transform(X)
            else:
                X_imputed = self.imputer.transform(X)
        elif method == 'knn':
            if self.imputer is None:
                self.imputer = KNNImputer(n_neighbors=5)
                X_imputed = self.imputer.fit_transform(X)
            else:
                X_imputed = self.imputer.transform(X)
        else:
            X_imputed = X
            
        return X_imputed
    
    def remove_low_variance_features(self, X, threshold=0.01):
        """
        Loại bỏ features có variance thấp
        
        Lý do:
        - Features với variance thấp (constant hoặc near-constant) không mang thông tin
        - Giảm dimensionality, tăng tốc độ training
        - Giảm overfitting
        
        Threshold: Tỷ lệ variance tối thiểu (0.01 = 1%)
        """
        print(f"Removing low variance features (threshold={threshold})...")
        
        if self.variance_selector is None:
            self.variance_selector = VarianceThreshold(threshold=threshold)
            X_filtered = self.variance_selector.fit_transform(X)
            n_removed = X.shape[1] - X_filtered.shape[1]
            print(f"Removed {n_removed} low-variance features")
        else:
            X_filtered = self.variance_selector.transform(X)
            
        return X_filtered
    
    def preprocess(self, X, y=None, batch_labels=None, is_train=True):
        """
        Complete preprocessing pipeline
        
        Pipeline:
        1. Handle missing values
        2. Log transformation (nếu là RNA-seq)
        3. Normalization
        4. Batch correction (nếu có batch info)
        5. Remove low variance features
        6. Z-score standardization
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Raw expression matrix
        y : array-like, shape (n_samples,), optional
            Target labels
        batch_labels : array-like, optional
            Batch information for each sample
        is_train : bool
            If True, fit the preprocessors; if False, transform only
            
        Returns:
        --------
        X_processed : array-like
            Processed data
        """
        print("\n" + "="*60)
        print("Starting Preprocessing Pipeline")
        print("="*60)
        print(f"Input shape: {X.shape}")
        
        # Step 1: Handle missing values
        X_processed = self.handle_missing_values(X, self.config.get('handle_missing', 'mean'))
        
        # Step 2: Log transformation (for RNA-seq)
        if self.config.get('normalization', '').startswith('log2'):
            X_processed = self.log2_transform(X_processed)
        
        # Step 3: Batch correction
        if batch_labels is not None and self.config.get('batch_correction') == 'combat':
            if is_train:
                X_processed = self.combat_batch_correction(X_processed, batch_labels)
        
        # Step 4: Remove low variance features
        if self.config.get('remove_low_variance', True):
            threshold = self.config.get('variance_threshold', 0.01)
            if is_train:
                X_processed = self.remove_low_variance_features(X_processed, threshold)
            else:
                if self.variance_selector is not None:
                    X_processed = self.variance_selector.transform(X_processed)
        
        # Step 5: Z-score normalization (always last step)
        X_processed = self.zscore_normalization(X_processed)
        
        print(f"Output shape: {X_processed.shape}")
        print("="*60 + "\n")
        
        return X_processed
    
    def get_feature_mask(self):
        """
        Lấy mask của features được giữ lại sau preprocessing
        Useful để map back feature importance về original features
        """
        if self.variance_selector is not None:
            return self.variance_selector.get_support()
        else:
            return None


def load_omics_data(file_path, config):
    """
    Load omics data từ file
    
    Expected format:
    - CSV/TSV file
    - Rows: samples
    - Columns: features (genes/proteins)
    - First column: sample IDs
    - Last column or separate file: labels
    
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    feature_names : list
    sample_ids : list
    """
    print(f"Loading data from {file_path}...")
    
    # Example for CSV file
    df = pd.read_csv(file_path, index_col=0)
    
    # Assume last column is label, others are features
    if 'label' in df.columns or 'class' in df.columns or 'subtype' in df.columns:
        label_col = [col for col in df.columns if col.lower() in ['label', 'class', 'subtype']][0]
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
        feature_names = df.drop(columns=[label_col]).columns.tolist()
    else:
        # If no label column, assume it's in a separate file or will be provided
        X = df.values
        y = None
        feature_names = df.columns.tolist()
    
    sample_ids = df.index.tolist()
    
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    if y is not None:
        unique_classes = np.unique(y)
        print(f"Number of classes: {len(unique_classes)}")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, feature_names, sample_ids


if __name__ == "__main__":
    """
    Example usage
    """
    from config import PREPROCESSING_CONFIG, DATASET_CONFIG
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = DATASET_CONFIG['n_samples']
    n_features = DATASET_CONFIG['n_features']
    n_classes = DATASET_CONFIG['n_classes']
    
    # Simulate RNA-seq like data (counts)
    X_raw = np.random.negative_binomial(10, 0.5, size=(n_samples, n_features)).astype(float)
    
    # Add some missing values (10%)
    missing_mask = np.random.random((n_samples, n_features)) < 0.1
    X_raw[missing_mask] = np.nan
    
    # Generate labels
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Create preprocessor
    preprocessor = OmicsPreprocessor(PREPROCESSING_CONFIG)
    
    # Preprocess data
    X_processed = preprocessor.preprocess(X_raw, y, is_train=True)
    
    print("\nPreprocessing completed successfully!")
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Data statistics:")
    print(f"  Mean: {X_processed.mean():.4f}")
    print(f"  Std: {X_processed.std():.4f}")
    print(f"  Min: {X_processed.min():.4f}")
    print(f"  Max: {X_processed.max():.4f}")
