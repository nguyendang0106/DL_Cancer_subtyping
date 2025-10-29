# 📚 Lý Thuyết & Công Thức Toán Học - Omics Cancer Classification

## Mục Lục

1. [Preprocessing Methods](#1-preprocessing-methods)
2. [Feature Selection](#2-feature-selection)
3. [Classical ML Models](#3-classical-ml-models)
4. [Deep Learning](#4-deep-learning)
5. [Graph Neural Networks](#5-graph-neural-networks)
6. [Model Interpretation](#6-model-interpretation)
7. [Advanced Methods](#7-advanced-methods)

---

## 1. PREPROCESSING METHODS

### 1.1 Log Transformation

**Công thức:**

```
X_transformed = log₂(X + 1)
```

**Lý do:**

- RNA-seq data có phân phối lệch phải (right-skewed)
- Log transform làm cho data gần với phân phối normal
- Ổn định variance
- +1 để tránh log(0)

**Khi nào sử dụng:**

- RNA-seq count data
- Proteomics abundance data
- Bất kỳ data nào có large dynamic range

---

### 1.2 TPM Normalization (Transcripts Per Million)

**Công thức:**

```
Step 1: RPK_i = reads_i / (gene_length_i / 1000)
Step 2: TPM_i = (RPK_i / Σ_j RPK_j) × 10⁶
```

**Lý do:**

- Chuẩn hóa cho gene length
- Chuẩn hóa cho library size (sequencing depth)
- Comparable across samples

**Alternative: CPM (Counts Per Million)**

```
CPM_i = (reads_i / total_reads) × 10⁶
```

---

### 1.3 Z-score Normalization

**Công thức:**

```
z = (x - μ) / σ

Trong đó:
- μ = mean(X)
- σ = std(X)
```

**Properties:**

- Mean = 0
- Standard deviation = 1
- Preserves outliers

**Lý do:**

- Đưa features về cùng scale
- Quan trọng cho SVM, Neural Networks, PCA
- Giúp gradient descent converge nhanh hơn

---

### 1.4 ComBat Batch Effect Correction

**Model:**

```
Y_ijg = α_g + X β_g + γ_ig + δ_ig ε_ijg

Trong đó:
- Y_ijg: Expression của gene g trong sample j của batch i
- α_g: Overall gene expression
- X β_g: Covariate effects
- γ_ig: Additive batch effect
- δ_ig: Multiplicative batch effect
- ε_ijg: Error term
```

**Empirical Bayes Estimation:**

```
γ̂_ig ~ N(γ̄_g, τ²_g)
δ̂²_ig ~ Inverse-Gamma(λ_g, θ_g)
```

**Corrected values:**

```
Y*_ijg = (Y_ijg - α̂_g - X β̂_g - γ̂_ig) / δ̂_ig + α̂_g + X β̂_g
```

**Lý do:**

- Technical variation giữa các batch (different labs, platforms, time)
- Preserve biological variation
- Better than simple centering/scaling

---

### 1.5 Quantile Normalization

**Algorithm:**

1. Rank expression values trong mỗi sample
2. Replace rank i values với mean của all rank i values
3. Rearrange về original order

**Effect:**

- Đảm bảo identical distribution across samples
- Removes technical variation
- Common trong microarray data

---

## 2. FEATURE SELECTION

### 2.1 Variance-Based Selection

**Công thức:**

```
Var(X_i) = E[(X_i - μ_i)²] = (1/n) Σⱼ (x_ij - μ_i)²
```

**Decision rule:**

- Keep features với Var(X_i) > threshold
- Or: Keep top k features với highest variance

**Pros:** Fast, unsupervised
**Cons:** Không xét correlation với target

---

### 2.2 ANOVA F-test

**Công thức:**

```
F = MSB / MSW

MSB = (Between-group variance) = [Σᵢ nᵢ(ȳᵢ - ȳ)²] / (k-1)
MSW = (Within-group variance) = [ΣᵢΣⱼ (yᵢⱼ - ȳᵢ)²] / (N-k)

Trong đó:
- k: Number of groups (cancer types)
- N: Total samples
- nᵢ: Samples in group i
- ȳᵢ: Mean of group i
- ȳ: Overall mean
```

**p-value:**

```
p = P(F > F_observed | H₀: μ₁ = μ₂ = ... = μₖ)
```

**Decision rule:**

- Reject H₀ if p < α (typically 0.05)
- Keep features với smallest p-values

---

### 2.3 Mutual Information

**Công thức:**

```
I(X;Y) = ΣₓΣᵧ p(x,y) log[p(x,y) / (p(x)p(y))]
      = H(Y) - H(Y|X)
      = H(X) + H(Y) - H(X,Y)

Trong đó:
- H(Y) = -Σᵧ p(y) log p(y)  (Entropy)
- H(Y|X) = -ΣₓΣᵧ p(x,y) log p(y|x)  (Conditional entropy)
```

**Properties:**

- I(X;Y) ≥ 0
- I(X;Y) = 0 if X, Y independent
- I(X;Y) = I(Y;X) (symmetric)

**Lý do:**

- Captures non-linear dependencies
- Không giả định phân phối cụ thể
- Better than correlation cho complex relationships

---

### 2.4 Lasso (L1 Regularization)

**Optimization problem:**

```
min_w [1/(2n) ||Xw - y||² + α||w||₁]

||w||₁ = Σᵢ |wᵢ|  (L1 norm)
```

**Solution characteristics:**

- Sparse solution: Many wᵢ = 0 (automatic feature selection)
- Convex optimization
- Feature selection + regularization

**Vs L2 (Ridge):**

```
L2: min_w [1/(2n) ||Xw - y||² + α||w||²₂]
||w||₂² = Σᵢ wᵢ²
```

- L2 shrinks weights nhưng rarely = 0
- L1 drives weights to exactly 0

---

### 2.5 Principal Component Analysis (PCA)

**Mathematical formulation:**

**Step 1: Standardize data**

```
X_std = (X - μ) / σ
```

**Step 2: Covariance matrix**

```
C = (1/n) X_std^T X_std
```

**Step 3: Eigen decomposition**

```
C = V Λ V^T

Trong đó:
- V: Eigenvectors (principal components)
- Λ: Eigenvalues (variance explained)
```

**Step 4: Projection**

```
X_pca = X_std V[:, :k]

k: Number of components (explained variance ≥ threshold)
```

**Explained variance ratio:**

```
EVR_k = λₖ / Σᵢ λᵢ
```

**Cumulative explained variance:**

```
CEV_k = Σᵢ₌₁ᵏ λᵢ / Σᵢ λᵢ
```

---

## 3. CLASSICAL ML MODELS

### 3.1 Support Vector Machine (SVM)

**Primal problem (Linear SVM):**

```
min_{w,b} [1/2 ||w||² + C Σᵢ ξᵢ]

subject to:
  yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ  ∀i
  ξᵢ ≥ 0  ∀i
```

**Decision function:**

```
f(x) = sign(w^T x + b)
```

**Dual problem:**

```
max_α [Σᵢ αᵢ - 1/2 Σᵢ Σⱼ αᵢαⱼyᵢyⱼxᵢ^T xⱼ]

subject to:
  0 ≤ αᵢ ≤ C  ∀i
  Σᵢ αᵢyᵢ = 0
```

**Kernel trick:**

```
K(xᵢ, xⱼ) = φ(xᵢ)^T φ(xⱼ)

Linear: K(x,x') = x^T x'
RBF: K(x,x') = exp(-γ||x-x'||²)
Polynomial: K(x,x') = (x^T x' + c)^d
```

**Hyperparameters:**

- C: Regularization (trade-off margin vs errors)
  - Large C: Hard margin (risk overfitting)
  - Small C: Soft margin (better generalization)
- γ (for RBF): Kernel width
  - Large γ: Tight fit (overfit risk)
  - Small γ: Smooth decision boundary

---

### 3.2 XGBoost (Extreme Gradient Boosting)

**Objective function:**

```
obj^(t) = Σᵢ L(yᵢ, ŷᵢ^(t-1) + fₜ(xᵢ)) + Ω(fₜ)

Ω(f) = γT + 1/2 λ Σⱼ wⱼ²

Trong đó:
- L: Loss function (e.g., log loss cho classification)
- fₜ: New tree at iteration t
- T: Number of leaves
- wⱼ: Leaf weights
- γ: Complexity penalty
- λ: L2 regularization
```

**Boosting update:**

```
ŷ^(t) = ŷ^(t-1) + η fₜ(x)

η: Learning rate (shrinkage)
```

**Split finding (gradient-based):**

```
Gain = [GL²/(HL + λ) + GR²/(HR + λ) - (GL + GR)²/(HL + HR + λ)] / 2 - γ

Trong đó:
- GL, GR: Sum of gradients (left, right)
- HL, HR: Sum of hessians (left, right)
```

**Key hyperparameters:**

- n_estimators: Number of trees
- max_depth: Tree depth (control overfitting)
- learning_rate (η): Step size (0.01-0.3)
- subsample: Row sampling ratio
- colsample_bytree: Feature sampling ratio
- lambda, alpha: L2, L1 regularization

---

### 3.3 Logistic Regression (L1)

**Model:**

```
p(y=1|x) = σ(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
```

**Loss function (Binary Cross-Entropy):**

```
L = -Σᵢ [yᵢ log(pᵢ) + (1-yᵢ)log(1-pᵢ)]
```

**With L1 regularization:**

```
min_w [-Σᵢ yᵢ log(pᵢ) + (1-yᵢ)log(1-pᵢ) + α||w||₁]
```

**Multi-class (Softmax):**

```
p(y=k|x) = exp(wₖ^T x) / Σⱼ exp(wⱼ^T x)

Loss = -Σᵢ Σₖ yᵢₖ log(pᵢₖ)
```

**Sparse solution:**

- L1 penalty drives many weights to exactly 0
- Automatic feature selection
- Interpretable

---

## 4. DEEP LEARNING

### 4.1 Multi-Layer Perceptron (MLP)

**Forward propagation:**

```
Layer l:
  z^(l) = W^(l) a^(l-1) + b^(l)
  a^(l) = σ(z^(l))

Trong đó:
- a^(0) = x (input)
- W^(l): Weight matrix
- b^(l): Bias vector
- σ: Activation function
```

**Activation functions:**

**ReLU:**

```
σ(z) = max(0, z)

Gradient: dσ/dz = {1 if z > 0, 0 otherwise}
```

**Sigmoid:**

```
σ(z) = 1 / (1 + e^(-z))

Gradient: dσ/dz = σ(z)(1 - σ(z))
```

**Tanh:**

```
σ(z) = (e^z - e^(-z)) / (e^z + e^(-z))

Gradient: dσ/dz = 1 - σ²(z)
```

---

### 4.2 Batch Normalization

**Công thức:**

```
Step 1: Compute batch statistics
  μ_B = 1/m Σᵢ xᵢ
  σ²_B = 1/m Σᵢ (xᵢ - μ_B)²

Step 2: Normalize
  x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

Step 3: Scale and shift (learnable parameters)
  yᵢ = γ x̂ᵢ + β
```

**Lý do:**

- Reduce internal covariate shift
- Allows higher learning rates
- Regularization effect
- Stabilizes training

**Inference:**

```
Use moving average statistics:
  μ_running = momentum × μ_running + (1-momentum) × μ_batch
  σ²_running = momentum × σ²_running + (1-momentum) × σ²_batch
```

---

### 4.3 Dropout

**Training:**

```
For each neuron i:
  rᵢ ~ Bernoulli(p)
  yᵢ = rᵢ × xᵢ

p: Dropout probability (typically 0.3-0.5)
```

**Inference:**

```
y = (1-p) × x  (Scale by keep probability)
```

**Effect:**

- Prevents co-adaptation của neurons
- Ensemble effect (train 2^n networks implicitly)
- Regularization

---

### 4.4 Cross-Entropy Loss

**Binary:**

```
L = -Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

**Multi-class:**

```
L = -Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)

Trong đó:
- yᵢₖ: True label (one-hot encoded)
- ŷᵢₖ: Predicted probability
```

**Properties:**

- Convex (for linear models)
- Gradient well-behaved
- Probabilistic interpretation

---

### 4.5 Adam Optimizer

**Update rule:**

```
Step 1: Compute gradient
  gₜ = ∇L(θₜ)

Step 2: Update biased first moment
  mₜ = β₁ mₜ₋₁ + (1-β₁) gₜ

Step 3: Update biased second moment
  vₜ = β₂ vₜ₋₁ + (1-β₂) gₜ²

Step 4: Bias correction
  m̂ₜ = mₜ / (1 - β₁ᵗ)
  v̂ₜ = vₜ / (1 - β₂ᵗ)

Step 5: Update parameters
  θₜ = θₜ₋₁ - α × m̂ₜ / (√v̂ₜ + ε)
```

**Hyperparameters:**

- α: Learning rate (default: 0.001)
- β₁: First moment decay (default: 0.9)
- β₂: Second moment decay (default: 0.999)
- ε: Numerical stability (default: 10⁻⁸)

---

### 4.6 Autoencoder

**Architecture:**

```
Encoder: x → h = f(Wx + b)
Decoder: h → x̂ = g(W'h + b')

Loss: ||x - x̂||²
```

**Variants:**

**Denoising Autoencoder:**

```
x̃ = x + noise
Minimize: ||x - decoder(encoder(x̃))||²
```

**Variational Autoencoder (VAE):**

```
Encoder: x → (μ, σ²)
Sample: z ~ N(μ, σ²)
Decoder: z → x̂

Loss = Reconstruction + KL Divergence
     = ||x - x̂||² + KL(q(z|x) || p(z))
```

---

## 5. GRAPH NEURAL NETWORKS

### 5.1 Graph Convolutional Network (GCN)

**Layer formula:**

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))

Trong đó:
- A: Adjacency matrix
- Ã = A + I (add self-loops)
- D̃ᵢᵢ = Σⱼ Ãᵢⱼ (degree matrix)
- H^(l): Node features at layer l
- W^(l): Learnable weights
- σ: Activation function
```

**Intuition:**

- Average neighbor features (weighted by degree)
- Apply linear transformation
- Non-linear activation

**Message passing interpretation:**

```
hᵢ^(l+1) = σ(Σⱼ∈N(i) 1/√(dᵢdⱼ) W^(l) hⱼ^(l))
```

---

### 5.2 Graph Attention Network (GAT)

**Attention mechanism:**

```
Step 1: Compute attention coefficients
  eᵢⱼ = LeakyReLU(a^T [Whᵢ || Whⱼ])

Step 2: Normalize (softmax)
  αᵢⱼ = exp(eᵢⱼ) / Σₖ∈N(i) exp(eᵢₖ)

Step 3: Aggregate
  hᵢ' = σ(Σⱼ∈N(i) αᵢⱼ Whⱼ)
```

**Multi-head attention:**

```
hᵢ' = ||ₖ₌₁ᴷ σ(Σⱼ∈N(i) αᵢⱼᵏ Wᵏhⱼ)

||: Concatenation
K: Number of attention heads
```

---

## 6. MODEL INTERPRETATION

### 6.1 SHAP (Shapley Values)

**Shapley value (from cooperative game theory):**

```
φᵢ(f) = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)!] / |F|! × [f(S∪{i}) - f(S)]

Trong đó:
- F: Set of all features
- S: Subset of features
- f(S): Model prediction with feature subset S
- φᵢ: Contribution of feature i
```

**Properties:**

1. **Efficiency (accuracy):**

```
f(x) = φ₀ + Σᵢ φᵢ(x)
```

2. **Symmetry:**

```
If f(S∪{i}) = f(S∪{j}) for all S, then φᵢ = φⱼ
```

3. **Dummy:**

```
If f(S∪{i}) = f(S) for all S, then φᵢ = 0
```

4. **Additivity:**

```
φᵢ(f+g) = φᵢ(f) + φᵢ(g)
```

---

### 6.2 Integrated Gradients

**Formula:**

```
IGᵢ(x) = (xᵢ - x'ᵢ) × ∫₀¹ ∂f(x' + α(x-x')) / ∂xᵢ dα

Riemann approximation:
IGᵢ(x) ≈ (xᵢ - x'ᵢ) × Σₖ₌₁ᵐ [∂f(x' + k/m(x-x')) / ∂xᵢ] / m
```

**Axioms:**

1. **Sensitivity:**

```
If xᵢ ≠ x'ᵢ and f(x) ≠ f(x'), then IGᵢ(x) ≠ 0
```

2. **Implementation Invariance:**

```
Functionally equivalent networks have same attributions
```

3. **Completeness:**

```
f(x) - f(x') = Σᵢ IGᵢ(x)
```

---

### 6.3 Permutation Importance

**Algorithm:**

```
1. Compute baseline score: s_base = score(model, X, y)

2. For each feature i:
   a. Shuffle feature i: X̃ᵢ = shuffle(Xᵢ)
   b. Compute score: sᵢ = score(model, X̃, y)
   c. Importance: Impᵢ = s_base - sᵢ

3. Repeat n_repeats times and average
```

**Statistical significance:**

```
Mean importance: μ = mean(Imp)
Std: σ = std(Imp)

Significant if: μ > k × σ (typically k=2)
```

---

## 7. ADVANCED METHODS

### 7.1 Transformer Self-Attention

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(QK^T / √dₖ) V

Trong đó:
- Q: Query matrix
- K: Key matrix
- V: Value matrix
- dₖ: Dimension of keys (for scaling)
```

**Multi-Head Attention:**

```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) W^O

headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

**Positional Encoding:**

```
PE(pos, 2i) = sin(pos / 10000^(2i/dₘₒdₑₗ))
PE(pos, 2i+1) = cos(pos / 10000^(2i/dₘₒdₑₗ))
```

---

### 7.2 Cox Proportional Hazards

**Model:**

```
h(t|x) = h₀(t) exp(β^T x)

Trong đó:
- h(t|x): Hazard function
- h₀(t): Baseline hazard
- β^T x: Linear predictor (risk score)
```

**Partial Likelihood:**

```
L(β) = Πᵢ [exp(β^T xᵢ) / Σⱼ∈R(tᵢ) exp(β^T xⱼ)]^δᵢ

δᵢ: Event indicator (1=event, 0=censored)
R(tᵢ): Risk set at time tᵢ
```

**C-index (Concordance Index):**

```
C = #{pairs (i,j): tᵢ < tⱼ and risk_i > risk_j} / #{comparable pairs}

Range: [0, 1]
C=0.5: Random
C=1.0: Perfect concordance
```

---

## 📊 Summary Table: When to Use What

| Method          | Best For                             | Pros                                 | Cons                        |
| --------------- | ------------------------------------ | ------------------------------------ | --------------------------- |
| **SVM**         | High-dimensional, clear margins      | Strong theory, kernel trick          | Slow for large data         |
| **XGBoost**     | Tabular data, competitions           | SOTA performance, feature importance | Black box, overfitting risk |
| **Logistic L1** | Interpretability, sparse solutions   | Simple, fast, interpretable          | Linear only                 |
| **MLP**         | Large datasets, complex patterns     | Flexible, powerful                   | Needs много data, black box |
| **Autoencoder** | Unsupervised pretraining             | Leverage unlabeled data              | Two-stage training          |
| **GNN**         | Graph-structured data, PPI           | Use biological knowledge             | Needs graph structure       |
| **Transformer** | Very large datasets, long-range deps | SOTA in many domains                 | Computationally expensive   |

---

## 🎯 Key Takeaways

1. **Preprocessing is crucial** cho omics data (normalization, batch correction)
2. **Feature selection essential** khi n_features >> n_samples
3. **Classical ML** often competitive với small-medium datasets
4. **Deep Learning** excels với large datasets và complex patterns
5. **GNNs** leverage biological prior knowledge (pathways, PPI)
6. **Interpretation** không optional - always explain predictions
7. **Cross-validation** và external validation essential

---

**End of Theory Document** 📚
