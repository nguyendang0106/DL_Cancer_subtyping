"""
Model Interpretation and Explainability
Bao gồm: SHAP, Integrated Gradients, Feature Importance
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """
    Interpret và explain model predictions
    
    Lý do cần model interpretation:
    1. Clinical trust: Doctors cần hiểu tại sao model predict một diagnosis
    2. Biological insights: Identify key genes/biomarkers
    3. Regulatory compliance: Medical AI cần explainability
    4. Model debugging: Detect biases và errors
    5. Scientific discovery: Generate hypotheses về disease mechanisms
    
    Methods:
    1. SHAP (SHapley Additive exPlanations)
    2. Integrated Gradients (for neural networks)
    3. Permutation Importance
    4. Attention Weights (for GNN/Transformer)
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.feature_importance = None
    
    def shap_analysis(self, X_train, X_test, model_type='tree'):
        """
        SHAP (SHapley Additive exPlanations)
        
        Lundberg & Lee, 2017: "A Unified Approach to Interpreting Model Predictions"
        
        Công thức (Shapley value từ game theory):
        φ_i = Σ_{S⊆F\{i}} [|S|!(|F|-|S|-1)!]/|F|! × [f(S∪{i}) - f(S)]
        
        Trong đó:
        - φ_i: SHAP value của feature i
        - F: Set of all features
        - S: Subset of features
        - f(S): Model prediction với feature subset S
        
        Interpretation:
        - φ_i > 0: Feature i increases prediction
        - φ_i < 0: Feature i decreases prediction
        - |φ_i|: Magnitude of impact
        
        Properties:
        1. Local accuracy: f(x) = φ_0 + Σ φ_i
        2. Missingness: φ_i = 0 if feature i is missing
        3. Consistency: If feature becomes more important, SHAP increases
        
        Types:
        - TreeExplainer: For tree-based models (fast)
        - DeepExplainer: For neural networks (uses DeepLIFT)
        - KernelExplainer: Model-agnostic (slow)
        """
        print("\n" + "="*60)
        print("SHAP Analysis")
        print("="*60)
        
        try:
            import shap
            
            # Select explainer based on model type
            if model_type == 'tree':
                # For XGBoost, Random Forest, etc.
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test)
            elif model_type == 'deep':
                # For neural networks
                background = shap.sample(X_train, 100)  # Background dataset
                explainer = shap.DeepExplainer(self.model, background)
                shap_values = explainer.shap_values(X_test)
            else:
                # Model-agnostic (slower)
                explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    shap.sample(X_train, 100)
                )
                shap_values = explainer.shap_values(X_test)
            
            self.shap_values = shap_values
            
            # Summary statistics
            if isinstance(shap_values, list):  # Multi-class
                # Average absolute SHAP values across classes
                mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Top features
            top_indices = np.argsort(mean_abs_shap)[::-1][:20]
            
            print("Top 20 most important features by SHAP:")
            for i, idx in enumerate(top_indices, 1):
                feat_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
                print(f"{i}. {feat_name}: {mean_abs_shap[idx]:.4f}")
            
            print("="*60)
            
            return shap_values, explainer
            
        except ImportError:
            print("SHAP not installed. Install: pip install shap")
            return None, None
    
    def plot_shap_summary(self, X_test, shap_values=None, max_display=20, save_path=None):
        """
        Plot SHAP summary
        
        Summary plot shows:
        - Features ranked by importance (top to bottom)
        - SHAP values distribution (left-right)
        - Feature values (color: red=high, blue=low)
        """
        try:
            import shap
            
            if shap_values is None:
                shap_values = self.shap_values
            
            if shap_values is None:
                print("Run shap_analysis() first!")
                return
            
            plt.figure(figsize=(10, 8))
            
            # For multi-class, plot class 0 by default
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
            
            shap.summary_plot(
                shap_values_plot, 
                X_test,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error plotting SHAP: {e}")
    
    def integrated_gradients(self, model, X, baseline=None, steps=50):
        """
        Integrated Gradients for neural networks
        
        Sundararajan et al., 2017: "Axiomatic Attribution for Deep Networks"
        
        Công thức:
        IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂f(x' + α(x-x'))/∂x_i dα
        
        Approximation (Riemann sum):
        IG_i(x) ≈ (x_i - x'_i) × Σ_{k=1}^m [∂f(x^k)/∂x_i] / m
        
        Trong đó:
        - x: Input sample
        - x': Baseline (typically zeros or mean)
        - α: Interpolation coefficient
        - m: Number of steps
        
        Intuition:
        - Accumulate gradients along path from baseline to input
        - Satisfies axioms: Sensitivity và Implementation Invariance
        - More stable than vanilla gradients
        
        Advantages:
        - Theoretically grounded
        - No need for perturbations
        - Works for any differentiable model
        """
        print("\n" + "="*60)
        print("Integrated Gradients Analysis")
        print("="*60)
        
        if not isinstance(model, torch.nn.Module):
            print("Integrated Gradients only works for PyTorch models")
            return None
        
        model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Baseline: zeros or mean
        if baseline is None:
            baseline = torch.zeros_like(X_tensor)
        else:
            baseline = torch.FloatTensor(baseline)
        
        # Requires gradient
        X_tensor.requires_grad = True
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps)
        
        integrated_grads = torch.zeros_like(X_tensor)
        
        for alpha in alphas:
            # Interpolate
            x_interp = baseline + alpha * (X_tensor - baseline)
            x_interp.requires_grad = True
            
            # Forward pass
            output = model(x_interp)
            
            # For multi-class, use predicted class
            pred_class = output.argmax(dim=1)
            
            # Backward pass
            model.zero_grad()
            output[range(len(output)), pred_class].sum().backward()
            
            # Accumulate gradients
            integrated_grads += x_interp.grad
        
        # Average and scale
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (X_tensor - baseline)
        
        # Convert to numpy
        integrated_grads = integrated_grads.detach().numpy()
        
        # Feature importance (mean absolute IG across samples)
        feature_importance = np.abs(integrated_grads).mean(axis=0)
        
        # Top features
        top_indices = np.argsort(feature_importance)[::-1][:20]
        
        print("Top 20 most important features by Integrated Gradients:")
        for i, idx in enumerate(top_indices, 1):
            feat_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            print(f"{i}. {feat_name}: {feature_importance[idx]:.4f}")
        
        print("="*60)
        
        self.feature_importance = feature_importance
        
        return integrated_grads, feature_importance
    
    def permutation_importance_analysis(self, model, X, y, n_repeats=10):
        """
        Permutation Feature Importance
        
        Breiman, 2001: "Random Forests"
        
        Algorithm:
        1. Compute baseline score on validation set
        2. For each feature i:
           a. Randomly shuffle feature i
           b. Compute new score
           c. Importance_i = baseline_score - shuffled_score
        3. Repeat n_repeats times and average
        
        Interpretation:
        - High importance: Model performance drops significantly when feature is shuffled
        - Low importance: Feature is not used by model
        
        Advantages:
        - Model-agnostic
        - Accounts for feature interactions
        - Real-world interpretation (performance-based)
        
        Disadvantages:
        - Computationally expensive
        - Can be biased with correlated features
        """
        print("\n" + "="*60)
        print("Permutation Importance Analysis")
        print("="*60)
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
            scoring='f1_macro'
        )
        
        # Sort by importance
        sorted_idx = result.importances_mean.argsort()[::-1][:20]
        
        print("Top 20 most important features by Permutation Importance:")
        for i, idx in enumerate(sorted_idx, 1):
            feat_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            mean_imp = result.importances_mean[idx]
            std_imp = result.importances_std[idx]
            print(f"{i}. {feat_name}: {mean_imp:.4f} ± {std_imp:.4f}")
        
        print("="*60)
        
        return result
    
    def plot_feature_importance(self, importance_dict, top_n=20, save_path=None):
        """
        Plot feature importance comparison
        
        Parameters:
        -----------
        importance_dict : dict
            {'method_name': importance_array}
        """
        plt.figure(figsize=(12, 8))
        
        n_methods = len(importance_dict)
        
        for i, (method, importance) in enumerate(importance_dict.items(), 1):
            # Get top features
            top_indices = np.argsort(importance)[::-1][:top_n]
            top_values = importance[top_indices]
            
            if self.feature_names:
                top_names = [self.feature_names[idx] for idx in top_indices]
            else:
                top_names = [f"Feature_{idx}" for idx in top_indices]
            
            # Plot
            plt.subplot(1, n_methods, i)
            plt.barh(range(top_n), top_values[::-1])
            plt.yticks(range(top_n), top_names[::-1], fontsize=8)
            plt.xlabel('Importance Score')
            plt.title(f'{method}')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_top_genes(self, n_top=50):
        """
        Get top important genes/features
        
        Returns:
        --------
        DataFrame with gene names and importance scores
        """
        if self.feature_importance is None:
            print("Run interpretation methods first!")
            return None
        
        top_indices = np.argsort(self.feature_importance)[::-1][:n_top]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            gene_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            importance = self.feature_importance[idx]
            
            results.append({
                'Rank': rank,
                'Gene': gene_name,
                'Importance': importance
            })
        
        df = pd.DataFrame(results)
        return df
    
    def pathway_enrichment_analysis(self, top_genes, organism='human'):
        """
        Gene Ontology (GO) và Pathway Enrichment Analysis
        
        Sử dụng top important genes để:
        1. GO enrichment: Biological processes, molecular functions
        2. KEGG pathway: Disease-related pathways
        3. Reactome: Pathway database
        
        Tools:
        - Enrichr API
        - DAVID
        - g:Profiler
        
        Note: Requires API access or local databases
        """
        print("\n" + "="*60)
        print("Pathway Enrichment Analysis")
        print("="*60)
        
        print(f"Analyzing {len(top_genes)} genes...")
        
        try:
            import requests
            
            # Enrichr API
            ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'
            
            # Submit gene list
            genes_str = '\n'.join(top_genes)
            
            response = requests.post(
                ENRICHR_URL + '/addList',
                files={'list': (None, genes_str)}
            )
            
            if response.status_code == 200:
                user_list_id = response.json()['userListId']
                
                # Query enrichment
                gene_set_libraries = [
                    'GO_Biological_Process_2021',
                    'KEGG_2021_Human',
                    'Reactome_2022'
                ]
                
                enrichment_results = {}
                
                for lib in gene_set_libraries:
                    response = requests.get(
                        f'{ENRICHR_URL}/enrich',
                        params={'userListId': user_list_id, 'backgroundType': lib}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        enrichment_results[lib] = data[lib][:10]  # Top 10
                        
                        print(f"\nTop enriched terms from {lib}:")
                        for i, term in enumerate(data[lib][:5], 1):
                            print(f"{i}. {term[1]} (p-value: {term[2]:.2e})")
                
                print("="*60)
                return enrichment_results
            else:
                print(f"Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error in enrichment analysis: {e}")
            print("Consider using local tools: clusterProfiler (R), DAVID, g:Profiler")
            return None


if __name__ == "__main__":
    """
    Example usage
    """
    # Example with synthetic data
    np.random.seed(42)
    
    n_samples = 100
    n_features = 1000
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 5, n_samples)
    
    # Create feature names
    feature_names = [f"GENE_{i}" for i in range(n_features)]
    
    # Simulate a simple model (for demonstration)
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create interpreter
    interpreter = ModelInterpreter(model, feature_names)
    
    # SHAP analysis
    shap_values, explainer = interpreter.shap_analysis(X[:80], X[80:], model_type='tree')
    
    # Permutation importance
    perm_importance = interpreter.permutation_importance_analysis(model, X[80:], y[80:])
    
    # Get top genes
    top_genes_df = interpreter.get_top_genes(n_top=50)
    print("\n", top_genes_df.head(10))
