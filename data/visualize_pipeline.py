"""
Visualization utilities for omics cancer classification pipeline
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches


def plot_pipeline_flowchart(save_path='pipeline_flowchart.png'):
    """
    Vẽ flowchart của toàn bộ pipeline
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors
    color_data = '#E8F4F8'
    color_preprocess = '#D4E6F1'
    color_feature = '#AED6F1'
    color_model = '#85C1E2'
    color_eval = '#5DADE2'
    color_interp = '#3498DB'
    
    # Title
    ax.text(5, 13.5, 'Omics Cancer Classification Pipeline', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Step 1: Raw Data
    box1 = FancyBboxPatch((1, 12), 3, 0.8, boxstyle="round,pad=0.1", 
                          facecolor=color_data, edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.5, 12.4, 'Raw Omics Data\n(samples × genes)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrow
    ax.arrow(2.5, 12, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Step 2: Preprocessing
    box2 = FancyBboxPatch((0.5, 10), 4, 1.2, boxstyle="round,pad=0.1",
                          facecolor=color_preprocess, edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(2.5, 10.9, 'Preprocessing', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 10.5, '• Log2 Transform\n• TPM Normalization\n• Batch Correction\n• Z-score',
            ha='center', va='center', fontsize=9)
    
    # Arrow
    ax.arrow(2.5, 10, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Step 3: Feature Selection
    box3 = FancyBboxPatch((0.5, 7.5), 4, 1.8, boxstyle="round,pad=0.1",
                          facecolor=color_feature, edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(2.5, 9, 'Feature Selection', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 8.4, '• Variance Filter\n• ANOVA F-test\n• Mutual Information\n• Lasso (L1)\n• PCA (optional)',
            ha='center', va='center', fontsize=9)
    
    # Arrow splits into two
    ax.arrow(2.5, 7.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Step 4a: Classical ML
    box4a = FancyBboxPatch((0.2, 5), 2, 1.5, boxstyle="round,pad=0.1",
                           facecolor=color_model, edgecolor='black', linewidth=2)
    ax.add_patch(box4a)
    ax.text(1.2, 6.2, 'Classical ML', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.2, 5.7, '• SVM\n• XGBoost\n• Logistic L1',
            ha='center', va='center', fontsize=9)
    
    # Step 4b: Deep Learning
    box4b = FancyBboxPatch((2.8, 5), 2, 1.5, boxstyle="round,pad=0.1",
                           facecolor=color_model, edgecolor='black', linewidth=2)
    ax.add_patch(box4b)
    ax.text(3.8, 6.2, 'Deep Learning', ha='center', fontsize=11, fontweight='bold')
    ax.text(3.8, 5.7, '• MLP\n• Autoencoder\n• GNN',
            ha='center', va='center', fontsize=9)
    
    # Arrows merge
    ax.arrow(1.2, 5, 0.6, -0.5, head_width=0.15, head_length=0.08, fc='black', ec='black')
    ax.arrow(3.8, 5, -0.6, -0.5, head_width=0.15, head_length=0.08, fc='black', ec='black')
    
    # Step 5: Evaluation
    box5 = FancyBboxPatch((0.5, 3), 4, 1.2, boxstyle="round,pad=0.1",
                          facecolor=color_eval, edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(2.5, 3.9, 'Model Evaluation', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 3.5, '• Stratified K-Fold CV\n• Accuracy, F1, Precision, Recall\n• Confusion Matrix',
            ha='center', va='center', fontsize=9)
    
    # Arrow
    ax.arrow(2.5, 3, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Step 6: Interpretation
    box6 = FancyBboxPatch((0.5, 0.8), 4, 1.5, boxstyle="round,pad=0.1",
                          facecolor=color_interp, edgecolor='black', linewidth=2)
    ax.add_patch(box6)
    ax.text(2.5, 2, 'Model Interpretation', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 1.4, '• SHAP Values\n• Integrated Gradients\n• Feature Importance\n• Pathway Enrichment',
            ha='center', va='center', fontsize=9)
    
    # Right side: Advanced approaches
    ax.text(7.5, 13.5, 'Advanced Approaches', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Multi-omics
    box_adv1 = FancyBboxPatch((5.5, 11.5), 4, 1, boxstyle="round,pad=0.1",
                              facecolor='#F9E79F', edgecolor='black', linewidth=2)
    ax.add_patch(box_adv1)
    ax.text(7.5, 12.2, 'Multi-Omics Fusion', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 11.85, 'Early/Intermediate/Late Fusion',
            ha='center', va='center', fontsize=9)
    
    # Pathway-aware GNN
    box_adv2 = FancyBboxPatch((5.5, 10), 4, 1, boxstyle="round,pad=0.1",
                              facecolor='#ABEBC6', edgecolor='black', linewidth=2)
    ax.add_patch(box_adv2)
    ax.text(7.5, 10.7, 'Pathway-Aware GNN', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 10.35, 'PPI Networks + Graph Conv',
            ha='center', va='center', fontsize=9)
    
    # Transformer
    box_adv3 = FancyBboxPatch((5.5, 8.5), 4, 1, boxstyle="round,pad=0.1",
                              facecolor='#F8B4D9', edgecolor='black', linewidth=2)
    ax.add_patch(box_adv3)
    ax.text(7.5, 9.2, 'Transformer Models', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 8.85, 'Self-Attention Mechanism',
            ha='center', va='center', fontsize=9)
    
    # Survival Analysis
    box_adv4 = FancyBboxPatch((5.5, 7), 4, 1, boxstyle="round,pad=0.1",
                              facecolor='#D7BDE2', edgecolor='black', linewidth=2)
    ax.add_patch(box_adv4)
    ax.text(7.5, 7.7, 'Survival Analysis', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 7.35, 'DeepSurv, Cox PH',
            ha='center', va='center', fontsize=9)
    
    # Output
    ax.text(7.5, 6, 'Outputs', fontsize=14, fontweight='bold', ha='center')
    
    output_items = [
        '✓ Trained Models',
        '✓ Performance Metrics',
        '✓ Feature Rankings',
        '✓ Pathway Insights',
        '✓ Predictions'
    ]
    
    y_pos = 5.3
    for item in output_items:
        ax.text(7.5, y_pos, item, ha='center', fontsize=10)
        y_pos -= 0.4
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Pipeline flowchart saved to {save_path}")
    plt.show()


def plot_data_flow_diagram(save_path='data_flow.png'):
    """
    Vẽ data flow diagram
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Data Transformation Flow', 
            fontsize=18, fontweight='bold', ha='center')
    
    stages = [
        ('Raw Data\n(800 × 20,000)', 8.5),
        ('After Preprocessing\n(800 × 15,000)', 7),
        ('After Feature Selection\n(800 × 1,000)', 5.5),
        ('Train/Test Split\n(640/160 × 1,000)', 4),
        ('Model Predictions\n(160 × 5 classes)', 2.5),
        ('Evaluation Metrics\n(F1, Accuracy, etc.)', 1)
    ]
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
    
    for i, (stage, y_pos) in enumerate(stages):
        # Box
        box = FancyBboxPatch((2, y_pos-0.4), 6, 0.8, 
                            boxstyle="round,pad=0.1",
                            facecolor=colors[i], 
                            edgecolor='black', 
                            linewidth=2)
        ax.add_patch(box)
        
        # Text
        ax.text(5, y_pos, stage, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            ax.arrow(5, y_pos-0.4, 0, -0.5, 
                    head_width=0.3, head_length=0.15, 
                    fc='black', ec='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Data flow diagram saved to {save_path}")
    plt.show()


def plot_method_comparison_table(save_path='method_comparison.png'):
    """
    Vẽ bảng so sánh các methods
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Data
    methods = ['SVM', 'XGBoost', 'Logistic L1', 'MLP', 'Autoencoder', 'GNN']
    
    data = {
        'Method': methods,
        'Best For': [
            'High-dim, clear margins',
            'Tabular, complex patterns',
            'Sparse, interpretable',
            'Large data, flexible',
            'Unsupervised pretrain',
            'Graph-structured data'
        ],
        'Pros': [
            'Strong theory, kernels',
            'SOTA, feature import.',
            'Simple, interpretable',
            'Powerful, flexible',
            'Leverage unlabeled',
            'Use prior knowledge'
        ],
        'Cons': [
            'Slow for large data',
            'Black box, overfit risk',
            'Linear only',
            'Needs много data',
            'Two-stage training',
            'Needs graph'
        ],
        'Complexity': [
            'Medium',
            'Medium',
            'Low',
            'High',
            'High',
            'High'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create table
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns,
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#3498DB')
        cell.set_text_props(weight='bold', color='white')
    
    # Color rows
    colors = ['#E8F4F8', '#D4E6F1']
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor(colors[(i-1) % 2])
    
    plt.title('Method Comparison Table', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Method comparison table saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    """
    Generate all visualization diagrams
    """
    print("Generating pipeline visualizations...")
    
    # 1. Pipeline flowchart
    plot_pipeline_flowchart('pipeline_flowchart.png')
    
    # 2. Data flow diagram
    plot_data_flow_diagram('data_flow.png')
    
    # 3. Method comparison table
    plot_method_comparison_table('method_comparison.png')
    
    print("\n✓ All visualizations generated successfully!")
    print("Files created:")
    print("  - pipeline_flowchart.png")
    print("  - data_flow.png")
    print("  - method_comparison.png")
