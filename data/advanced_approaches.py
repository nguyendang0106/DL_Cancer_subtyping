"""
Advanced Approaches và Future Improvements
Bao gồm: Multi-omics fusion, Pathway-aware GNN, Transformer-based models
"""


# =============================================================================
# 1. MULTI-OMICS FUSION
# =============================================================================

"""
Lý do sử dụng Multi-omics:
- Single omics: Limited view của biological system
- Multi-omics: Comprehensive understanding
  * Genomics: DNA variations (mutations, CNV)
  * Transcriptomics: Gene expression (RNA-seq)
  * Proteomics: Protein abundance
  * Metabolomics: Metabolite levels
  * Epigenomics: DNA methylation, histone modifications

Integration strategies:
1. Early fusion: Concatenate features từ different omics
2. Intermediate fusion: Learn representations riêng rẽ, rồi combine
3. Late fusion: Train separate models, ensemble predictions
"""


# -----------------------------------------------------------------------------
# 1.1 Early Fusion (Feature Concatenation)
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


class EarlyFusionClassifier(nn.Module):
    """
    Early Fusion: Concatenate all omics features
    
    Architecture:
    [RNA features | Protein features | Methylation features] → MLP → Prediction
    
    Pros:
    - Simple implementation
    - Can capture cross-omics interactions
    
    Cons:
    - High dimensionality
    - Assumes all omics equally important
    - Missing data problem (different samples have different omics)
    """
    
    def __init__(self, omics_dims, n_classes, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        # Total input dimension
        total_dim = sum(omics_dims.values())
        
        # Build MLP
        layers = []
        prev_dim = total_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, omics_dict):
        """
        Parameters:
        -----------
        omics_dict : dict
            {'rna': tensor, 'protein': tensor, 'methylation': tensor}
        """
        # Concatenate all omics
        x = torch.cat(list(omics_dict.values()), dim=1)
        return self.model(x)


# -----------------------------------------------------------------------------
# 1.2 Intermediate Fusion (Representation Learning)
# -----------------------------------------------------------------------------

class IntermediateFusionClassifier(nn.Module):
    """
    Intermediate Fusion: Learn representations cho mỗi omics type riêng biệt,
    sau đó combine representations
    
    Architecture:
    RNA → Encoder_RNA → h_RNA \\
    Protein → Encoder_Protein → h_Protein  → [Concat/Attention] → Classifier
    Methylation → Encoder_Methylation → h_Methylation /
    
    Pros:
    - Learn omics-specific patterns
    - Can handle missing omics (use only available ones)
    - More flexible
    
    Cons:
    - More complex
    - More parameters to train
    """
    
    def __init__(self, omics_dims, latent_dim=128, n_classes=5):
        super().__init__()
        
        # Encoders for each omics type
        self.encoders = nn.ModuleDict()
        
        for omics_name, input_dim in omics_dims.items():
            self.encoders[omics_name] = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU()
            )
        
        # Fusion layer (concatenation)
        fusion_dim = latent_dim * len(omics_dims)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, omics_dict):
        """
        Parameters:
        -----------
        omics_dict : dict
            {'rna': tensor, 'protein': tensor, ...}
            Can have missing omics (use zeros for missing)
        """
        # Encode each omics
        representations = []
        
        for omics_name, encoder in self.encoders.items():
            if omics_name in omics_dict:
                h = encoder(omics_dict[omics_name])
            else:
                # Missing omics: use zeros
                batch_size = list(omics_dict.values())[0].size(0)
                h = torch.zeros(batch_size, self.latent_dim).to(next(encoder.parameters()).device)
            
            representations.append(h)
        
        # Concatenate representations
        fused = torch.cat(representations, dim=1)
        
        # Classify
        output = self.classifier(fused)
        
        return output


# -----------------------------------------------------------------------------
# 1.3 Attention-Based Fusion
# -----------------------------------------------------------------------------

class AttentionFusion(nn.Module):
    """
    Attention-based fusion: Learn importance weights cho mỗi omics
    
    Công thức:
    α_i = exp(w_i) / Σ exp(w_j)  (softmax attention)
    h_fused = Σ α_i × h_i
    
    Pros:
    - Adaptive weighting
    - Interpretable (attention weights show omics importance)
    - Can handle variable number of omics
    """
    
    def __init__(self, omics_dims, latent_dim=128, n_classes=5):
        super().__init__()
        
        # Encoders
        self.encoders = nn.ModuleDict()
        for omics_name, input_dim in omics_dims.items():
            self.encoders[omics_name] = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim),
                nn.ReLU()
            )
        
        # Attention weights
        self.attention = nn.Linear(latent_dim, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, omics_dict):
        # Encode each omics
        representations = []
        
        for omics_name, encoder in self.encoders.items():
            if omics_name in omics_dict:
                h = encoder(omics_dict[omics_name])
                representations.append(h)
        
        # Stack representations: [batch, n_omics, latent_dim]
        h_stack = torch.stack(representations, dim=1)
        
        # Compute attention weights
        attention_scores = self.attention(h_stack)  # [batch, n_omics, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        h_fused = (h_stack * attention_weights).sum(dim=1)  # [batch, latent_dim]
        
        # Classify
        output = self.classifier(h_fused)
        
        return output, attention_weights


# =============================================================================
# 2. PATHWAY-AWARE GRAPH NEURAL NETWORKS
# =============================================================================

"""
Lý do sử dụng biological pathways:
- Genes không hoạt động độc lập
- Pathways: Functional modules (e.g., cell cycle, apoptosis)
- Incorporating pathway structure → better generalization

Sources:
- KEGG: Kyoto Encyclopedia of Genes and Genomes
- Reactome: Pathway database
- Gene Ontology (GO): Biological processes

Graph construction:
- Nodes: Genes
- Edges: Pathway membership (genes in same pathway)
- Node features: Expression values
- Edge weights: Pathway confidence or co-expression
"""

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class PathwayAwareGNN(nn.Module):
    """
    GNN sử dụng pathway structure
    
    Architecture:
    Gene Expression → Graph Convolution (với pathway edges) → 
    Pathway-level representations → Classification
    
    Advantages:
    - Utilize biological prior knowledge
    - Interpretable: Can identify important pathways
    - Better generalization (regularized by pathway structure)
    """
    
    def __init__(self, n_genes, hidden_channels=256, n_pathways=100, n_classes=5):
        super().__init__()
        
        # Gene-level GNN
        self.gene_conv1 = GCNConv(1, hidden_channels)  # Each gene has 1 feature (expression)
        self.gene_conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Pathway-level aggregation
        # Map genes to pathways
        self.gene_to_pathway = nn.Linear(hidden_channels, n_pathways)
        
        # Pathway-level GNN (optional)
        # Build pathway-pathway interaction graph
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_pathways, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, edge_index, gene_to_pathway_map, batch=None):
        """
        Parameters:
        -----------
        x : Gene expression features [n_nodes, 1]
        edge_index : Gene-gene interactions [2, n_edges]
        gene_to_pathway_map : Mapping from genes to pathways [n_genes, n_pathways]
        batch : Batch assignment for multiple graphs
        """
        # Gene-level message passing
        x = self.gene_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.gene_conv2(x, edge_index)
        x = F.relu(x)
        
        # Aggregate to pathway level
        # Option 1: Linear projection
        pathway_features = self.gene_to_pathway(x)  # [n_genes, n_pathways]
        
        # Option 2: Use predefined gene-pathway mapping
        # pathway_features = torch.matmul(gene_to_pathway_map.t(), x)
        
        # Global pooling (per sample)
        if batch is not None:
            pathway_features = global_mean_pool(pathway_features, batch)
        else:
            pathway_features = pathway_features.mean(dim=0, keepdim=True)
        
        # Classify
        output = self.classifier(pathway_features)
        
        return output


# =============================================================================
# 3. TRANSFORMER-BASED MODELS
# =============================================================================

"""
Transformer for omics data:

Motivation:
- Self-attention: Capture long-range dependencies between genes
- Positional encoding: Can incorporate gene positions (for genomics)
- Powerful feature learning
- State-of-the-art in NLP, vision → apply to omics

Challenges:
- Computational complexity: O(n²) for n genes
- Need large datasets
- Interpretability
"""


class OmicsTransformer(nn.Module):
    """
    Transformer for cancer classification
    
    Architecture:
    Gene Expression → Embedding → [Transformer Encoder] × L → 
    Global Pooling → Classifier
    
    Key components:
    1. Multi-Head Self-Attention
    2. Positional Encoding
    3. Feed-Forward Networks
    4. Layer Normalization
    """
    
    def __init__(self, n_features, n_classes, d_model=256, n_heads=8, 
                 n_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)  # Each gene: 1 value → d_model
        
        # Positional encoding (optional for genes)
        self.positional_encoding = PositionalEncoding(d_model, max_len=n_features)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        """
        Parameters:
        -----------
        x : Gene expression [batch_size, n_genes]
        """
        # Reshape: [batch, n_genes] → [batch, n_genes, 1]
        x = x.unsqueeze(-1)
        
        # Project to d_model
        x = self.input_projection(x)  # [batch, n_genes, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, n_genes, d_model]
        
        # Global pooling (mean over genes)
        x = x.mean(dim=1)  # [batch, d_model]
        
        # Classify
        output = self.classifier(x)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    
    Công thức:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


# =============================================================================
# 4. SURVIVAL ANALYSIS INTEGRATION
# =============================================================================

"""
Survival analysis: Predict time-to-event (e.g., patient survival time)

Models:
- Cox Proportional Hazards
- DeepSurv (neural network for survival)
- Random Survival Forest

Loss function:
- Cox partial likelihood
- C-index (concordance index)
"""


class DeepSurvivalModel(nn.Module):
    """
    Deep learning for survival analysis
    
    Cox Proportional Hazards:
    h(t|x) = h_0(t) × exp(β^T x)
    
    Neural network learns: β^T x
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output: risk score (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Returns:
        --------
        risk_score : Relative risk (higher = worse prognosis)
        """
        return self.model(x)


def cox_loss(risk_scores, times, events):
    """
    Cox Partial Likelihood Loss
    
    Parameters:
    -----------
    risk_scores : Predicted risk scores
    times : Survival times
    events : Event indicators (1=event occurred, 0=censored)
    
    Formula:
    L = -Σ_i [event_i × (risk_i - log Σ_{j: time_j ≥ time_i} exp(risk_j))]
    """
    # Sort by time
    idx = torch.argsort(times, descending=True)
    risk_scores = risk_scores[idx]
    events = events[idx]
    
    # Compute log cumulative sum of exp(risk)
    log_cumsum_exp = torch.logcumsumexp(risk_scores, dim=0)
    
    # Cox loss
    loss = -(risk_scores - log_cumsum_exp) * events
    loss = loss.sum() / events.sum()
    
    return loss


# =============================================================================
# 5. FEDERATED LEARNING FOR PRIVACY-PRESERVING MULTI-CENTER STUDIES
# =============================================================================

"""
Federated Learning: Train model on distributed data without sharing raw data

Scenario:
- Multiple hospitals have cancer patient data
- Privacy concerns: Cannot share patient data
- Solution: Train model locally, share only model updates

Algorithm (FedAvg):
1. Server initializes global model w_0
2. For each round t:
   a. Send w_t to all clients
   b. Each client trains on local data: w_t^k = w_t - η∇L_k(w_t)
   c. Server aggregates: w_{t+1} = Σ (n_k/n) w_t^k
3. Repeat until convergence
"""


class FederatedLearningServer:
    """
    Server for federated learning
    """
    
    def __init__(self, model):
        self.global_model = model
    
    def aggregate(self, client_models, client_weights):
        """
        Aggregate client models (FedAvg)
        
        Parameters:
        -----------
        client_models : List of client model parameters
        client_weights : List of client data sizes
        """
        # Weighted average of parameters
        total_weight = sum(client_weights)
        
        # Get global model state dict
        global_dict = self.global_model.state_dict()
        
        # Average each parameter
        for key in global_dict.keys():
            global_dict[key] = sum([
                client_models[i][key] * client_weights[i] / total_weight
                for i in range(len(client_models))
            ])
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        return self.global_model


# =============================================================================
# 6. RECOMMENDATIONS & BEST PRACTICES
# =============================================================================

"""
RECOMMENDATIONS:

1. Data Quality:
   - Remove batch effects (ComBat, Limma)
   - Quality control: Remove low-quality samples
   - Normalization: TPM/CPM for RNA-seq

2. Feature Engineering:
   - Domain knowledge: Use pathway information
   - Feature selection: Aggressive for high-dimensional data
   - Feature scaling: Z-score normalization

3. Model Selection:
   - Small data (<500 samples): Classical ML (XGBoost, SVM)
   - Medium data (500-5000): MLP, Ensemble
   - Large data (>5000): Deep learning, Transformers
   - Multi-omics: Intermediate fusion with attention

4. Evaluation:
   - Stratified K-fold cross-validation
   - Hold-out external validation set
   - Report confidence intervals
   - Class imbalance: Use balanced metrics

5. Interpretation:
   - Always interpret predictions (SHAP, IG)
   - Validate biological relevance (pathway enrichment)
   - Compare with known biomarkers
   - Visualize important features

6. Reproducibility:
   - Set random seeds
   - Save preprocessing parameters
   - Version control (git)
   - Document hyperparameters

7. Clinical Translation:
   - External validation on independent cohorts
   - Prospective validation
   - Consider clinical utility
   - Regulatory approval (FDA)
"""


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Multi-omics fusion
    """
    
    # Synthetic multi-omics data
    batch_size = 32
    
    omics_data = {
        'rna': torch.randn(batch_size, 1000),      # 1000 genes
        'protein': torch.randn(batch_size, 200),   # 200 proteins
        'methylation': torch.randn(batch_size, 500) # 500 CpG sites
    }
    
    omics_dims = {k: v.shape[1] for k, v in omics_data.items()}
    
    # Early fusion
    model_early = EarlyFusionClassifier(omics_dims, n_classes=5)
    output_early = model_early(omics_data)
    print(f"Early fusion output shape: {output_early.shape}")
    
    # Intermediate fusion
    model_intermediate = IntermediateFusionClassifier(omics_dims, latent_dim=128, n_classes=5)
    output_intermediate = model_intermediate(omics_data)
    print(f"Intermediate fusion output shape: {output_intermediate.shape}")
    
    # Attention fusion
    model_attention = AttentionFusion(omics_dims, latent_dim=128, n_classes=5)
    output_attention, attention_weights = model_attention(omics_data)
    print(f"Attention fusion output shape: {output_attention.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights (normalized): {attention_weights.squeeze().mean(dim=0)}")
    
    # Transformer
    model_transformer = OmicsTransformer(
        n_features=1000, 
        n_classes=5,
        d_model=256,
        n_heads=8,
        n_layers=4
    )
    rna_data = torch.randn(batch_size, 1000)
    output_transformer = model_transformer(rna_data)
    print(f"Transformer output shape: {output_transformer.shape}")
    
    print("\nAll advanced models initialized successfully!")
