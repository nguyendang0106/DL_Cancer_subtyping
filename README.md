# 🧬 Omics Cancer Classification Pipeline

**Pipeline hoàn chỉnh cho phân loại ung thư trên dữ liệu omics sử dụng Deep Learning và Classical Machine Learning**

---

## 📋 Tổng Quan

Pipeline này được thiết kế để xử lý và phân loại các loại ung thư khác nhau dựa trên dữ liệu omics (genomics, transcriptomics, proteomics, methylation). Hệ thống bao gồm:

- ✅ **Tiền xử lý dữ liệu** chuyên biệt cho omics data
- ✅ **Feature selection** với nhiều phương pháp (ANOVA, Lasso, Mutual Information)
- ✅ **Classical ML models** (SVM, XGBoost, Logistic Regression) làm baseline
- ✅ **Deep Learning models** (MLP, Autoencoder + Classifier)
- ✅ **Graph Neural Networks** sử dụng PPI networks
- ✅ **Model interpretation** (SHAP, Integrated Gradients)
- ✅ **Pathway enrichment analysis**

---

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy pipeline với dữ liệu mẫu

```bash
cd data
python main_pipeline.py
```

### 3. Sử dụng với dữ liệu thực

```python
from main_pipeline import OmicsCancerClassificationPipeline
import pandas as pd

# Load your data
df = pd.read_csv('your_omics_data.csv', index_col=0)
X = df.drop('label', axis=1).values  # Expression matrix
y = df['label'].values  # Cancer subtypes
feature_names = df.drop('label', axis=1).columns.tolist()  # Gene names

# Run pipeline
pipeline = OmicsCancerClassificationPipeline()
comparison_df, results = pipeline.run_full_pipeline(
    X=X, y=y, feature_names=feature_names
)
```

---

## 📊 Dataset Specifications

### Định dạng dữ liệu đầu vào

**Matrix format: `samples × features`**

| Sample_ID | GENE_1 | GENE_2 | ... | GENE_N | Label |
| --------- | ------ | ------ | --- | ------ | ----- |
| Sample_1  | 125.3  | 45.2   | ... | 78.9   | 0     |
| Sample_2  | 98.7   | 52.1   | ... | 103.4  | 1     |
| ...       | ...    | ...    | ... | ...    | ...   |

### Cấu hình mặc định (config.py)

```python
DATASET_CONFIG = {
    'omics_type': 'RNA-seq',      # Loại dữ liệu
    'n_samples': 800,              # Số mẫu
    'n_features': 20000,           # Số gene/protein
    'n_classes': 5,                # Số loại ung thư
    'test_size': 0.2,              # Tỷ lệ test set
    'random_state': 42
}
```

---

## 🔬 Pipeline Architecture

### 1️⃣ **Data Preprocessing** (`data_preprocessing.py`)

#### Các bước xử lý:

```
Raw Data → Missing Values → Log Transform → Batch Correction →
Variance Filtering → Z-score Normalization → Preprocessed Data
```

#### Methods:

- **Log2 Transformation**: `log2(X + 1)` - Giảm skewness
- **TPM/CPM Normalization**: Chuẩn hóa library size
- **Z-score**: Standardization (mean=0, std=1)
- **ComBat**: Batch effect correction
- **Variance Filtering**: Loại bỏ low-variance features

#### Công thức quan trọng:

**Z-score Normalization:**

```
z = (x - μ) / σ
```

**ComBat (Empirical Bayes):**

```
X_corrected = (X - γ_batch) / δ_batch
```

---

### 2️⃣ **Feature Selection** (`feature_selection.py`)

#### Methods được implement:

1. **Variance-based**: `Var(X_i) = E[(X_i - μ_i)²]`
2. **ANOVA F-test**: `F = (Between-group variance) / (Within-group variance)`
3. **Mutual Information**: `I(X;Y) = Σ Σ p(x,y) log[p(x,y)/(p(x)p(y))]`
4. **Lasso (L1)**: `min_w [1/2n ||Xw - y||² + α||w||₁]`
5. **Random Forest Importance**: Gini importance
6. **PCA**: Dimensionality reduction

#### Ensemble Selection:

- **Union**: Lấy tất cả features từ bất kỳ method nào
- **Intersection**: Chỉ lấy features được chọn bởi TẤT CẢ methods
- **Majority**: Features được chọn bởi >50% methods

---

### 3️⃣ **Classical ML Models** (`classical_models.py`)

#### SVM (Support Vector Machine)

**Decision function:**

```
f(x) = sign(w^T x + b)
```

**Optimization:**

```
min_w,b [1/2 ||w||² + C Σ ξ_i]
subject to: y_i(w^T x_i + b) ≥ 1 - ξ_i
```

- **Kernels**: Linear, RBF
- **Best for**: High-dimensional data, clear margins

#### XGBoost

**Objective:**

```
obj(θ) = Σ L(y_i, ŷ_i) + Σ Ω(f_k)
Ω(f) = γT + 1/2 λ||w||²
```

- **Boosting**: Ensemble of trees
- **Best for**: Complex non-linear relationships

#### Logistic Regression (L1)

**Loss with L1 regularization:**

```
min_w [-Σ y_i log(p_i) + (1-y_i)log(1-p_i) + α||w||₁]
```

- **Sparse solution**: Many weights → 0
- **Best for**: Interpretability, feature selection

---

### 4️⃣ **Deep Learning Models** (`deep_learning_models.py`)

#### MLP (Multi-Layer Perceptron)

**Architecture:**

```
Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output
```

**Components:**

1. **Batch Normalization:**

   ```
   x̂ = (x - μ_batch) / √(σ²_batch + ε)
   y = γx̂ + β
   ```

2. **Dropout:**

   - Training: Randomly zero activations (p=0.3)
   - Inference: Scale by (1-p)

3. **Cross-Entropy Loss:**
   ```
   L = -Σ Σ y_ic log(ŷ_ic)
   ```

#### Autoencoder + Classifier

**Two-stage training:**

1. **Pretrain Autoencoder (unsupervised):**

   ```
   Encoder: X → Z (latent)
   Decoder: Z → X̂
   Loss = ||X - X̂||²
   ```

2. **Train Classifier:**
   ```
   Encoder (frozen/fine-tuned) → Classifier → Predictions
   ```

**Advantages:**

- Leverage unlabeled data
- Learn compressed representations
- Better initialization

---

### 5️⃣ **Graph Neural Networks** (`gnn_models.py`)

#### GCN (Graph Convolutional Network)

**Layer formula:**

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

**Components:**

- Ã = A + I (adjacency + self-loops)
- D̃: Degree matrix
- H: Node features (gene expression)

#### GAT (Graph Attention Network)

**Attention mechanism:**

```
α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
h_i' = σ(Σ_j α_ij W h_j)
```

#### PPI Network Construction:

- **Source**: STRING database
- **Nodes**: Genes/Proteins
- **Edges**: Protein-protein interactions
- **Weights**: Confidence scores (0-1)

**Why GNN for omics?**

- Capture gene-gene interactions
- Utilize biological prior knowledge
- Identify functional modules

---

### 6️⃣ **Model Interpretation** (`model_interpretation.py`)

#### SHAP (SHapley Additive exPlanations)

**Shapley value (from game theory):**

```
φ_i = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)!]/|F|! × [f(S∪{i}) - f(S)]
```

**Properties:**

- Local accuracy: `f(x) = φ_0 + Σ φ_i`
- Consistency
- Missingness

**Interpretation:**

- φ_i > 0: Feature increases prediction
- φ_i < 0: Feature decreases prediction

#### Integrated Gradients

**Formula:**

```
IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂f(x' + α(x-x'))/∂x_i dα
```

**Riemann approximation:**

```
IG_i(x) ≈ (x_i - x'_i) × Σ_{k=1}^m [∂f(x^k)/∂x_i] / m
```

#### Permutation Importance

**Algorithm:**

1. Baseline score on validation set
2. For each feature: shuffle → compute score → importance = drop in score
3. Repeat and average

**Advantages:**

- Model-agnostic
- Accounts for interactions
- Real-world interpretation

---

## 📈 Evaluation Metrics

### Classification Metrics

1. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`

2. **Precision**: `TP / (TP + FP)`

3. **Recall**: `TP / (TP + FN)`

4. **F1-score**: `2 × (Precision × Recall) / (Precision + Recall)`

5. **Macro F1**: Average F1 across classes (equal weight)

6. **Weighted F1**: Weighted by class support

### Cross-Validation

**Stratified K-Fold (K=5):**

- Maintains class distribution in each fold
- Reduces overfitting
- Better generalization estimate

---

## 🎯 Usage Examples

### Example 1: Basic Pipeline

```python
from main_pipeline import OmicsCancerClassificationPipeline

# Initialize pipeline
pipeline = OmicsCancerClassificationPipeline()

# Run with your data
comparison_df, results = pipeline.run_full_pipeline(
    data_path='data/cancer_expression.csv'
)
```

### Example 2: Custom Configuration

```python
from config import *

# Modify configuration
PREPROCESSING_CONFIG['normalization'] = 'quantile'
FEATURE_SELECTION_CONFIG['methods'] = ['anova', 'lasso']
DL_CONFIG['mlp']['hidden_layers'] = [1024, 512, 256, 128]

# Create pipeline with custom config
pipeline = OmicsCancerClassificationPipeline()
pipeline.run_full_pipeline(X=X, y=y, feature_names=genes)
```

### Example 3: Individual Components

```python
# Just preprocessing
from data_preprocessing import OmicsPreprocessor

preprocessor = OmicsPreprocessor(PREPROCESSING_CONFIG)
X_processed = preprocessor.preprocess(X_raw, y, is_train=True)

# Just feature selection
from feature_selection import FeatureSelector

selector = FeatureSelector(FEATURE_SELECTION_CONFIG)
X_selected, indices = selector.select_features(X_processed, y)

# Just model training
from classical_models import ClassicalMLModels

clf = ClassicalMLModels(CLASSICAL_MODELS_CONFIG)
clf.train_xgboost(X_train, y_train)
results = clf.evaluate_model('xgboost', X_test, y_test)
```

---

## 🔧 Configuration Guide

### Preprocessing Options

```python
PREPROCESSING_CONFIG = {
    'normalization': 'log2_tpm',    # 'log2_tpm', 'zscore', 'quantile'
    'batch_correction': 'combat',    # 'combat', 'limma', None
    'remove_low_variance': True,
    'variance_threshold': 0.01,
    'handle_missing': 'knn'          # 'mean', 'median', 'knn'
}
```

### Feature Selection Options

```python
FEATURE_SELECTION_CONFIG = {
    'methods': ['variance', 'anova', 'lasso'],
    'n_features_variance': 5000,
    'n_features_anova': 2000,
    'n_features_lasso': 1000,
    'use_pca': False,
    'pca_components': 0.95
}
```

### Deep Learning Hyperparameters

```python
DL_CONFIG = {
    'mlp': {
        'hidden_layers': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'batch_norm': True,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 15
    }
}
```

---

## 📊 Expected Output

### 1. Model Comparison Table

```
Model                      Type           Test Accuracy  Test F1 (Macro)
XGBOOST                    Classical ML   0.8625         0.8534
SVM                        Classical ML   0.8375         0.8289
Logistic Regression (L1)   Classical ML   0.8125         0.8012
MLP                        Deep Learning  0.8750         0.8645
Autoencoder + Classifier   Deep Learning  0.8500         0.8423
```

### 2. Top Important Features

```
Rank  Gene       Importance
1     GENE_45    0.1234
2     GENE_123   0.1156
3     GENE_789   0.0987
...
```

### 3. Visualization Files

- `model_comparison.png` - Performance comparison
- `confusion_matrix_*.png` - Per-model confusion matrices
- `shap_summary.png` - SHAP feature importance
- `training_history.png` - DL training curves

---

## 🧪 Advanced Features

### 1. Multi-Omics Fusion

```python
from advanced_models import MultiOmicsFusion

# Late fusion (concatenate features)
fusion_model = MultiOmicsFusion(
    omics_types=['rna', 'protein', 'methylation'],
    fusion_type='late'
)
```

### 2. Pathway-Aware GNN

```python
from gnn_models import PPINetworkBuilder, GCNClassifier

# Build PPI network
ppi_builder = PPINetworkBuilder(confidence_threshold=0.7)
graph = ppi_builder.load_string_ppi(gene_list, species=9606)

# Train GNN
model = GCNClassifier(input_dim=1, hidden_channels=256, n_classes=5)
```

### 3. Transformer-Based Models

```python
from transformer_models import OmicsTransformer

model = OmicsTransformer(
    n_features=1000,
    n_classes=5,
    n_heads=8,
    n_layers=6,
    d_model=256
)
```

---

## 🔬 Biological Interpretation

### Pathway Enrichment Analysis

```python
from model_interpretation import ModelInterpreter

interpreter = ModelInterpreter(model, feature_names)

# Get top genes
top_genes_df = interpreter.get_top_genes(n_top=50)
top_genes = top_genes_df['Gene'].tolist()

# Enrichment analysis
enrichment = interpreter.pathway_enrichment_analysis(
    top_genes,
    organism='human'
)
```

**Output:**

- GO Biological Processes
- KEGG Pathways
- Reactome Pathways

---

## ⚠️ Important Notes

### 1. Data Quality

- **Missing values**: Handle appropriately (KNN imputation recommended)
- **Batch effects**: Use ComBat if data from multiple batches
- **Outliers**: Check and remove if necessary

### 2. Feature Selection

- High-dimensional data (n_features >> n_samples) → aggressive selection needed
- Balance between reduction and information retention
- Consider biological knowledge (pathway-based selection)

### 3. Model Selection

- **Small datasets (<500 samples)**: Classical ML often better
- **Large datasets (>1000 samples)**: DL can excel
- **Interpretability needed**: Logistic Regression, Trees
- **Performance priority**: XGBoost, Neural Networks

### 4. Evaluation

- **Always use stratified splits**: Maintain class balance
- **Cross-validation**: More robust than single split
- **Class imbalance**: Use balanced metrics (Macro F1, not just Accuracy)

---

## 📚 References

### Methods

1. **Batch Correction**:

   - Johnson et al., 2007: "Adjusting batch effects in microarray expression data"

2. **SHAP**:

   - Lundberg & Lee, 2017: "A Unified Approach to Interpreting Model Predictions"

3. **Integrated Gradients**:

   - Sundararajan et al., 2017: "Axiomatic Attribution for Deep Networks"

4. **GCN**:

   - Kipf & Welling, 2017: "Semi-Supervised Classification with Graph Convolutional Networks"

5. **GAT**:
   - Veličković et al., 2018: "Graph Attention Networks"

### Databases

- **STRING**: Protein-protein interaction database
- **KEGG**: Pathway database
- **Reactome**: Pathway knowledge base
- **GO**: Gene Ontology

---

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```python
# Reduce batch size
DL_CONFIG['mlp']['batch_size'] = 16

# Use less features
FEATURE_SELECTION_CONFIG['n_features_anova'] = 500
```

**2. Overfitting**

```python
# Increase dropout
DL_CONFIG['mlp']['dropout_rate'] = 0.5

# Stronger regularization
CLASSICAL_MODELS_CONFIG['svm']['C'] = 0.1
```

**3. Poor Performance**

- Check data quality
- Try different normalization methods
- Increase feature selection threshold
- Tune hyperparameters

---

## 📞 Support & Contact

For questions or issues:

- Open an issue on GitHub
- Contact: [your-email@example.com]

---

## 📄 License

MIT License - Free to use for research and education

---

## 🎓 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{omics_cancer_pipeline,
  title={Omics Cancer Classification Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/omics-cancer-pipeline}
}
```

---

## 🚀 Future Improvements

- [ ] Multi-omics integration (early/intermediate fusion)
- [ ] Transfer learning from pretrained models
- [ ] Attention-based feature selection
- [ ] Graph transformer architectures
- [ ] Survival analysis integration
- [ ] Interactive visualization dashboard
- [ ] Automated hyperparameter tuning (AutoML)
- [ ] Federated learning support

---

**Happy Analyzing! 🧬🔬**
