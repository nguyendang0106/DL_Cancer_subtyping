"""
Configuration file for Omics Cancer Classification Pipeline
"""

# Dataset Configuration
DATASET_CONFIG = {
    'omics_type': 'RNA-seq',  # 'RNA-seq', 'proteomics', 'methylation', 'multi-omics'
    'n_samples': 800,
    'n_features': 20000,  # Number of genes/proteins
    'n_classes': 5,  # Number of cancer subtypes
    'input_format': 'samples_x_features',  # rows: samples, columns: features
    'test_size': 0.2,
    'random_state': 42
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'normalization': 'log2_tpm',  # 'log2_tpm', 'log2_cpm', 'zscore', 'quantile'
    'batch_correction': 'combat',  # 'combat', 'limma', None
    'remove_low_variance': True,
    'variance_threshold': 0.01,
    'handle_missing': 'mean',  # 'mean', 'median', 'knn'
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'methods': ['variance', 'anova', 'lasso'],
    'n_features_variance': 5000,  # Keep top 5000 most variant genes
    'n_features_anova': 2000,     # Keep top 2000 genes by ANOVA F-score
    'n_features_lasso': 1000,     # Keep top 1000 genes by L1 regularization
    'pca_components': 0.95,       # Explain 95% variance
    'use_pca': False,             # Set True to use PCA after feature selection
}

# Classical ML Models Configuration
CLASSICAL_MODELS_CONFIG = {
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'class_weight': 'balanced'
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'logistic_l1': {
        'C': [0.001, 0.01, 0.1, 1],
        'penalty': 'l1',
        'solver': 'liblinear',
        'class_weight': 'balanced'
    }
}

# Deep Learning Configuration
DL_CONFIG = {
    'mlp': {
        'hidden_layers': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'batch_norm': True,
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 15
    },
    'autoencoder': {
        'encoder_layers': [1000, 500, 250, 100],
        'latent_dim': 50,
        'decoder_layers': [100, 250, 500, 1000],
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32
    },
    'gnn': {
        'hidden_channels': 256,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 100
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'cv_folds': 5,
    'stratified': True,
    'metrics': ['accuracy', 'macro_f1', 'weighted_f1', 'precision', 'recall'],
    'confusion_matrix': True,
    'roc_curve': True
}

# Interpretation Configuration
INTERPRETATION_CONFIG = {
    'use_shap': True,
    'use_integrated_gradients': True,
    'top_n_features': 50,  # Number of top important features to report
    'save_feature_importance': True
}

# Paths
PATHS = {
    'data_dir': './data',
    'results_dir': './results',
    'models_dir': './models',
    'figures_dir': './figures',
    'logs_dir': './logs'
}

# Advanced Approaches Configuration
ADVANCED_CONFIG = {
    'multi_omics_fusion': {
        'fusion_type': 'late',  # 'early', 'intermediate', 'late'
        'integration_method': 'concatenate'  # 'concatenate', 'attention', 'graph'
    },
    'pathway_aware_gnn': {
        'use_ppi_network': True,
        'ppi_database': 'STRING',  # 'STRING', 'BioGRID', 'HPRD'
        'confidence_threshold': 0.7
    },
    'transformer': {
        'n_heads': 8,
        'n_layers': 6,
        'd_model': 256,
        'dim_feedforward': 1024
    }
}
