# 🚀 Quick Start Guide - Omics Cancer Classification

## 📦 Installation (5 phút)

### Bước 1: Clone repository (nếu có) hoặc sử dụng code hiện tại

```bash
cd /path/to/your/project
```

### Bước 2: Tạo virtual environment (recommended)

```bash
# Sử dụng venv
python -m venv omics_env
source omics_env/bin/activate  # macOS/Linux
# omics_env\Scripts\activate  # Windows

# Hoặc sử dụng conda
conda create -n omics_env python=3.9
conda activate omics_env
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Nếu gặp lỗi với `torch-geometric`, cài riêng:

```bash
pip install torch torchvision
pip install torch-geometric
```

---

## 🎯 Usage Scenarios

### Scenario 1: Demo với Synthetic Data (2 phút)

```bash
cd data
python main_pipeline.py
```

**Output:**

- `model_comparison.png` - So sánh performance các models
- `top_important_features.csv` - Top genes quan trọng
- `best_mlp_model.pth` - Trained model
- Console output với metrics

---

### Scenario 2: Sử dụng với CSV File (5 phút)

**Chuẩn bị data:**

File: `cancer_data.csv`

```
,GENE_1,GENE_2,GENE_3,...,GENE_N,label
Sample_1,125.3,45.2,78.9,...,103.4,0
Sample_2,98.7,52.1,65.3,...,87.2,1
Sample_3,110.5,48.9,72.1,...,95.8,0
...
```

**Code:**

```python
from main_pipeline import OmicsCancerClassificationPipeline
import pandas as pd

# Load data
df = pd.read_csv('cancer_data.csv', index_col=0)

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values
feature_names = df.drop('label', axis=1).columns.tolist()

# Run pipeline
pipeline = OmicsCancerClassificationPipeline()
comparison_df, results = pipeline.run_full_pipeline(
    X=X,
    y=y,
    feature_names=feature_names
)

# Print results
print(comparison_df)
```

---

### Scenario 3: Step-by-Step Execution (15 phút)

```python
from config import *
from data_preprocessing import OmicsPreprocessor
from feature_selection import FeatureSelector
from classical_models import ClassicalMLModels
from deep_learning_models import DeepLearningTrainer
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Load your data
df = pd.read_csv('your_data.csv', index_col=0)
X_raw = df.drop('label', axis=1).values
y = df['label'].values
feature_names = df.drop('label', axis=1).columns.tolist()

# 2. Preprocessing
preprocessor = OmicsPreprocessor(PREPROCESSING_CONFIG)
X_processed = preprocessor.preprocess(X_raw, y, is_train=True)

# 3. Feature selection
selector = FeatureSelector(FEATURE_SELECTION_CONFIG)
X_selected, selected_indices = selector.select_features(X_processed, y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Train classical models
clf = ClassicalMLModels(CLASSICAL_MODELS_CONFIG)
clf.train_xgboost(X_train, y_train)
results = clf.evaluate_model('xgboost', X_test, y_test)

print(f"XGBoost Test F1: {results['f1_macro']:.4f}")

# 6. Train deep learning model
X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

dl_trainer = DeepLearningTrainer(DL_CONFIG)
model = dl_trainer.train_mlp(X_train_dl, y_train_dl, X_val_dl, y_val_dl)

# Evaluate
y_pred, y_proba = dl_trainer.predict(X_test)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"MLP Test F1: {f1:.4f}")

# 7. Interpret model
from model_interpretation import ModelInterpreter

interpreter = ModelInterpreter(clf.models['xgboost'], feature_names)
shap_values, explainer = interpreter.shap_analysis(
    X_train, X_test, model_type='tree'
)
top_genes = interpreter.get_top_genes(n_top=50)
print("\nTop 10 Important Genes:")
print(top_genes.head(10))
```

---

### Scenario 4: Custom Configuration (10 phút)

```python
from main_pipeline import OmicsCancerClassificationPipeline

# Custom configuration
custom_config = {
    'dataset': {
        'test_size': 0.2,
        'random_state': 42
    },
    'preprocessing': {
        'normalization': 'log2_tpm',
        'batch_correction': None,  # Disable batch correction
        'remove_low_variance': True,
        'variance_threshold': 0.05,  # More aggressive filtering
        'handle_missing': 'knn'
    },
    'feature_selection': {
        'methods': ['anova', 'lasso'],  # Only use these two
        'n_features_anova': 1000,
        'n_features_lasso': 500,
        'use_pca': True,  # Enable PCA
        'pca_components': 0.95
    },
    'classical_models': {
        'xgboost': {
            'n_estimators': [200, 300],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1]
        }
    },
    'deep_learning': {
        'mlp': {
            'hidden_layers': [1024, 512, 256],  # Deeper network
            'dropout_rate': 0.4,
            'batch_size': 64,
            'epochs': 150
        }
    }
}

# Initialize with custom config
pipeline = OmicsCancerClassificationPipeline(config_dict=custom_config)

# Run
comparison_df, results = pipeline.run_full_pipeline(
    data_path='your_data.csv'
)
```

---

## 🔧 Common Tasks

### Task 1: Chỉ train một model cụ thể

```python
from classical_models import ClassicalMLModels
from sklearn.model_selection import train_test_split

# Load and prepare data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train only XGBoost
clf = ClassicalMLModels({'xgboost': {}})
clf.train_xgboost(X_train, y_train)
results = clf.evaluate_model('xgboost', X_test, y_test)
```

---

### Task 2: Feature selection only

```python
from feature_selection import FeatureSelector

selector = FeatureSelector({
    'methods': ['variance', 'anova', 'lasso'],
    'n_features_variance': 5000,
    'n_features_anova': 2000,
    'n_features_lasso': 1000
})

X_selected, selected_indices = selector.select_features(X, y)

# Get selected feature names
selected_features = [feature_names[i] for i in selected_indices]
print(f"Selected {len(selected_features)} features")
```

---

### Task 3: Chỉ interpret existing model

```python
from model_interpretation import ModelInterpreter
import joblib

# Load trained model
model = joblib.load('trained_model.pkl')

# Interpret
interpreter = ModelInterpreter(model, feature_names)

# SHAP analysis
shap_values, explainer = interpreter.shap_analysis(
    X_train, X_test, model_type='tree'
)

# Plot
interpreter.plot_shap_summary(X_test, max_display=20)

# Get top genes
top_genes = interpreter.get_top_genes(n_top=50)
top_genes.to_csv('top_genes.csv', index=False)
```

---

### Task 4: Batch prediction on new data

```python
import joblib
import pandas as pd

# Load trained model and preprocessor
model = joblib.load('trained_xgboost.pkl')
preprocessor = joblib.load('preprocessor.pkl')
selector = joblib.load('feature_selector.pkl')

# Load new data
new_df = pd.read_csv('new_samples.csv', index_col=0)
X_new = new_df.values

# Preprocess
X_processed = preprocessor.preprocess(X_new, is_train=False)

# Select features
X_selected = X_processed[:, selector.selected_features_indices]

# Predict
predictions = model.predict(X_selected)
probabilities = model.predict_proba(X_selected)

# Save results
results_df = pd.DataFrame({
    'Sample_ID': new_df.index,
    'Predicted_Class': predictions,
    'Confidence': probabilities.max(axis=1)
})
results_df.to_csv('predictions.csv', index=False)
```

---

## 📊 Understanding Output

### Model Comparison Table

```
Model                      Type           Test Accuracy  Test F1 (Macro)
XGBOOST                    Classical ML   0.8625         0.8534
MLP                        Deep Learning  0.8750         0.8645
SVM                        Classical ML   0.8375         0.8289
```

**Interpretation:**

- **Test Accuracy**: Overall correctness (TP+TN)/(TP+TN+FP+FN)
- **Test F1 (Macro)**: Average F1 across all classes (equal weight)
- Higher is better
- MLP outperforms in this example

---

### Top Important Features

```
Rank  Gene       Importance
1     TP53       0.1234
2     BRCA1      0.1156
3     MYC        0.0987
```

**Interpretation:**

- Features ranked by importance score (SHAP or other methods)
- Higher importance = more predictive power
- Can guide biological validation

---

### Confusion Matrix

```
           Predicted
           0   1   2   3   4
Actual 0  45   2   1   0   0
       1   3  42   2   1   0
       2   1   2  44   1   0
       3   0   1   2  43   2
       4   0   0   1   2  45
```

**Interpretation:**

- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
- Example: Class 0 → 45 correct, 2 predicted as class 1

---

## ⚠️ Troubleshooting

### Problem 1: Import errors

```
ImportError: No module named 'torch'
```

**Solution:**

```bash
pip install torch torchvision
```

---

### Problem 2: Out of memory

```
RuntimeError: CUDA out of memory
```

**Solution:**

```python
# Reduce batch size
DL_CONFIG['mlp']['batch_size'] = 16

# Or use CPU
dl_trainer = DeepLearningTrainer(DL_CONFIG, device='cpu')
```

---

### Problem 3: Poor performance

**Checklist:**

1. ✅ Check data quality (missing values, outliers)
2. ✅ Try different preprocessing methods
3. ✅ Adjust feature selection (more/less aggressive)
4. ✅ Tune hyperparameters
5. ✅ Check class imbalance
6. ✅ Use cross-validation

**Example: Tune XGBoost**

```python
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.3]
}

clf.train_xgboost(X_train, y_train, param_grid=param_grid)
```

---

### Problem 4: Slow training

**Solutions:**

1. **Reduce features:**

```python
FEATURE_SELECTION_CONFIG['n_features_lasso'] = 500  # Instead of 1000
```

2. **Use smaller validation set:**

```python
X_train, X_val = train_test_split(X_train_full, test_size=0.1)  # Instead of 0.2
```

3. **Reduce epochs:**

```python
DL_CONFIG['mlp']['epochs'] = 50  # Instead of 100
```

4. **Use GPU:**

```python
dl_trainer = DeepLearningTrainer(DL_CONFIG, device='cuda')
```

---

## 💡 Best Practices

### 1. Data Preparation

✅ **DO:**

- Check data distribution
- Handle missing values appropriately
- Remove low-quality samples
- Normalize/standardize

❌ **DON'T:**

- Skip quality control
- Mix train/test data
- Forget to stratify splits

---

### 2. Model Selection

✅ **DO:**

- Start with simple models (Logistic Regression)
- Compare multiple models
- Use cross-validation
- Report confidence intervals

❌ **DON'T:**

- Jump to complex models immediately
- Rely on single train/test split
- Ignore classical ML baselines

---

### 3. Evaluation

✅ **DO:**

- Use multiple metrics (Accuracy, F1, Precision, Recall)
- Check confusion matrix
- Validate on external data if possible
- Consider clinical relevance

❌ **DON'T:**

- Only report accuracy (especially with imbalanced data)
- Cherry-pick best results
- Overfit to validation set

---

### 4. Interpretation

✅ **DO:**

- Always interpret predictions
- Validate biological relevance
- Use multiple interpretation methods
- Visualize results

❌ **DON'T:**

- Treat model as black box
- Ignore domain knowledge
- Skip pathway enrichment

---

## 📚 Next Steps

### After running basic pipeline:

1. **Tune hyperparameters** với GridSearchCV hoặc Optuna
2. **Try advanced models** (GNN, Transformer)
3. **Multi-omics fusion** nếu có multiple data types
4. **Pathway enrichment** để hiểu biological mechanisms
5. **External validation** trên independent datasets
6. **Clinical validation** nếu có patient data

---

## 🆘 Getting Help

**Questions?**

1. Check README.md và THEORY.md
2. Review example code trong `__main__` blocks
3. Open GitHub issue
4. Contact: [your-email@example.com]

---

## ✅ Checklist: Am I Ready?

Before running:

- [ ] Data in correct format (samples × features)
- [ ] Labels are encoded (0, 1, 2, ...)
- [ ] Dependencies installed
- [ ] Configuration reviewed
- [ ] Output directory exists

After running:

- [ ] Results make sense (F1 > 0.5)
- [ ] Confusion matrix reasonable
- [ ] Top features are biologically plausible
- [ ] Saved models and results

---

**Happy Analyzing! 🧬🔬**

Need more help? Check THEORY.md for mathematical details or README.md for comprehensive guide.
