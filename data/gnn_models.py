"""
Graph Neural Network (GNN) Models using Protein-Protein Interaction (PPI) Networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class GCNClassifier(nn.Module):
    """
    Graph Convolutional Network (GCN) for omics classification
    
    Kipf & Welling, 2017: "Semi-Supervised Classification with Graph Convolutional Networks"
    
    Công thức GCN layer:
    H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
    
    Trong đó:
    - Ã = A + I (adjacency matrix với self-loops)
    - D̃: Degree matrix của Ã
    - H^(l): Node features ở layer l
    - W^(l): Learnable weights
    - σ: Activation function (ReLU)
    
    Lý do sử dụng GNN cho omics:
    1. Biological prior knowledge:
       - Genes không hoạt động độc lập
       - Protein-protein interactions (PPI)
       - Gene regulatory networks
       - Pathway information
    
    2. Graph structure:
       - Nodes: Genes/Proteins
       - Edges: Interactions (from STRING, BioGRID, etc.)
       - Node features: Expression values
    
    3. Message passing:
       - Aggregate information từ neighboring genes
       - Capture functional modules
       - Better than treating features independently
    
    4. Interpretability:
       - Identify important subgraphs
       - Pathway-level understanding
    """
    
    def __init__(self, input_dim, hidden_channels, n_classes, num_layers=3, dropout=0.3):
        super(GCNClassifier, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last conv layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, n_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Parameters:
        -----------
        x : Node features [num_nodes, input_dim]
        edge_index : Edge indices [2, num_edges]
        batch : Batch assignment [num_nodes] (for batched graphs)
        """
        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (if batched)
        if batch is not None:
            # Mean pooling per graph
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            # Single graph - mean pooling
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x


class GATClassifier(nn.Module):
    """
    Graph Attention Network (GAT)
    
    Veličković et al., 2018: "Graph Attention Networks"
    
    Công thức GAT layer:
    α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
    h_i' = σ(Σ_j α_ij W h_j)
    
    Lý do sử dụng GAT:
    1. Attention mechanism:
       - Học weights khác nhau cho các neighbors
       - Không phụ thuộc vào graph structure (không cần D̃^(-1/2))
       - More flexible than GCN
    
    2. Multi-head attention:
       - Stabilize learning
       - Attend to different aspects
       - Similar to Transformer
    
    3. Interpretability:
       - Attention weights show which genes are important
       - Identify key interactions
    """
    
    def __init__(self, input_dim, hidden_channels, n_classes, num_layers=3, 
                 heads=8, dropout=0.3):
        super(GATClassifier, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # First layer (multi-head)
        self.convs.append(GATConv(input_dim, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                     heads=heads, dropout=dropout))
        
        # Last layer (single head for classification)
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                 heads=1, concat=False, dropout=dropout))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, n_classes)
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PPINetworkBuilder:
    """
    Build PPI network từ databases (STRING, BioGRID, etc.)
    
    STRING database:
    - Comprehensive PPI database
    - Confidence scores (0-1000)
    - Multiple evidence types:
      * Experimental
      * Database
      * Text mining
      * Co-expression
      * Homology
    
    Steps:
    1. Download PPI data from STRING API
    2. Filter by confidence threshold (e.g., > 700 = high confidence)
    3. Map protein IDs to gene symbols
    4. Create graph structure
    """
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.graph = None
    
    def load_string_ppi(self, gene_list, species=9606):
        """
        Load PPI from STRING database
        
        Parameters:
        -----------
        gene_list : list
            List of gene symbols
        species : int
            NCBI taxonomy ID (9606 = Homo sapiens)
        confidence_threshold : float
            Minimum confidence score (0-1)
        
        Returns:
        --------
        networkx.Graph
        """
        print(f"Loading PPI network for {len(gene_list)} genes from STRING...")
        
        try:
            import requests
            
            # STRING API
            string_api_url = "https://string-db.org/api"
            output_format = "json"
            method = "network"
            
            # Build query
            genes_str = "%0d".join(gene_list)
            
            # Request
            request_url = f"{string_api_url}/{output_format}/{method}"
            params = {
                "identifiers": genes_str,
                "species": species,
                "required_score": int(self.confidence_threshold * 1000)
            }
            
            response = requests.post(request_url, data=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Build graph
                G = nx.Graph()
                
                # Add nodes
                for gene in gene_list:
                    G.add_node(gene)
                
                # Add edges
                for item in data:
                    gene1 = item['preferredName_A']
                    gene2 = item['preferredName_B']
                    score = item['score']
                    
                    if gene1 in gene_list and gene2 in gene_list:
                        G.add_edge(gene1, gene2, weight=score)
                
                self.graph = G
                
                print(f"PPI network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
                
                return G
            else:
                print(f"Error loading from STRING: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error: {e}")
            print("Falling back to random graph...")
            return self._create_random_graph(gene_list)
    
    def _create_random_graph(self, gene_list, edge_prob=0.05):
        """
        Create random graph (fallback nếu không load được từ STRING)
        
        Erdős–Rényi random graph
        """
        print(f"Creating random graph with {len(gene_list)} nodes...")
        
        G = nx.erdos_renyi_graph(len(gene_list), edge_prob)
        
        # Map node indices to gene names
        mapping = {i: gene for i, gene in enumerate(gene_list)}
        G = nx.relabel_nodes(G, mapping)
        
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
        
        self.graph = G
        
        print(f"Random graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def get_adjacency_info(self, gene_order):
        """
        Get adjacency matrix info for PyTorch Geometric
        
        Returns:
        --------
        edge_index : torch.Tensor [2, num_edges]
            COO format edge indices
        edge_weight : torch.Tensor [num_edges]
            Edge weights
        """
        if self.graph is None:
            raise ValueError("Graph not loaded. Call load_string_ppi() first.")
        
        # Create mapping from gene to index
        gene_to_idx = {gene: i for i, gene in enumerate(gene_order)}
        
        # Extract edges
        edge_list = []
        edge_weights = []
        
        for u, v, data in self.graph.edges(data=True):
            if u in gene_to_idx and v in gene_to_idx:
                idx_u = gene_to_idx[u]
                idx_v = gene_to_idx[v]
                weight = data.get('weight', 1.0)
                
                # Add both directions (undirected graph)
                edge_list.append([idx_u, idx_v])
                edge_list.append([idx_v, idx_u])
                edge_weights.append(weight)
                edge_weights.append(weight)
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weight


def create_graph_data(X, y, edge_index, edge_weight=None):
    """
    Create PyTorch Geometric Data object
    
    Parameters:
    -----------
    X : array-like [n_samples, n_features]
        Node features (gene expression)
    y : array-like [n_samples]
        Labels
    edge_index : torch.Tensor [2, num_edges]
        Edge indices
    edge_weight : torch.Tensor [num_edges], optional
        Edge weights
    
    Returns:
    --------
    data_list : list of Data objects
        One graph per sample
    """
    data_list = []
    
    for i in range(len(X)):
        # Each sample = one graph với same structure, different node features
        x = torch.FloatTensor(X[i].reshape(-1, 1))  # [n_genes, 1]
        y_label = torch.LongTensor([y[i]])
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y_label
        )
        
        data_list.append(data)
    
    return data_list


if __name__ == "__main__":
    """
    Example usage
    """
    from config import DL_CONFIG
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_genes = 500  # Number of genes
    n_classes = 5
    
    X = np.random.randn(n_samples, n_genes).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create gene list
    gene_list = [f"GENE_{i}" for i in range(n_genes)]
    
    # Build PPI network
    ppi_builder = PPINetworkBuilder(confidence_threshold=0.7)
    graph = ppi_builder._create_random_graph(gene_list, edge_prob=0.05)
    
    # Get adjacency info
    edge_index, edge_weight = ppi_builder.get_adjacency_info(gene_list)
    
    print(f"\nEdge index shape: {edge_index.shape}")
    print(f"Edge weight shape: {edge_weight.shape}")
    
    # Create graph dataset
    data_list = create_graph_data(X, y, edge_index, edge_weight)
    
    print(f"\nCreated {len(data_list)} graphs")
    print(f"First graph: {data_list[0]}")
    
    # Create model
    gnn_config = DL_CONFIG['gnn']
    model = GCNClassifier(
        input_dim=1,  # Each node has 1 feature (expression value)
        hidden_channels=gnn_config['hidden_channels'],
        n_classes=n_classes,
        num_layers=gnn_config['num_layers'],
        dropout=gnn_config['dropout']
    )
    
    print(f"\nModel:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
