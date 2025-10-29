"""
Deep Learning Models for Cancer Classification
Bao gồm: MLP with Dropout + BatchNorm, Autoencoder + Classifier
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class OmicsDataset(Dataset):
    """Custom Dataset for Omics data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron với Dropout và Batch Normalization
    
    Architecture:
    Input → [Linear → BatchNorm → ReLU → Dropout] × N → Output
    
    Lý do thiết kế:
    1. Batch Normalization:
       - Normalize activations: x̂ = (x - μ_batch) / √(σ²_batch + ε)
       - Ổn định training, tăng learning rate
       - Regularization effect
       - Giảm internal covariate shift
    
    2. Dropout:
       - During training: Randomly zero activations với prob p
       - During inference: Scale by (1-p)
       - Prevent co-adaptation của neurons
       - Ensemble effect (implicitly trains multiple networks)
    
    3. ReLU activation:
       - f(x) = max(0, x)
       - Non-linear, mitigates vanishing gradient
       - Sparse activation
       - Fast computation
    
    4. Progressive dimensionality reduction:
       - Gradually compress information
       - Learn hierarchical representations
       - Example: 1000 → 512 → 256 → 128 → 64 → n_classes
    """
    
    def __init__(self, input_dim, hidden_layers, n_classes, dropout_rate=0.3, use_batchnorm=True):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    """
    Autoencoder for unsupervised feature learning
    
    Architecture:
    Encoder: Input → Hidden layers → Latent space (bottleneck)
    Decoder: Latent space → Hidden layers → Reconstruction
    
    Công thức:
    Loss = ||X - X̂||² (Reconstruction error)
    
    Lý do sử dụng Autoencoder:
    1. Unsupervised pre-training:
       - Learn compressed representation without labels
       - Extract meaningful features from high-dimensional data
    
    2. Dimensionality reduction:
       - Non-linear alternative to PCA
       - Can capture complex patterns
    
    3. Denoising:
       - Add noise to input, train to reconstruct clean version
       - Learn robust features
    
    4. Transfer learning:
       - Pretrain encoder on large unlabeled data
       - Fine-tune with labeled data for classification
    
    Latent space: Compressed representation (e.g., 50-100 dims)
    """
    
    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder = []
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            encoder.append(nn.Linear(prev_dim, hidden_dim))
            encoder.append(nn.BatchNorm1d(hidden_dim))
            encoder.append(nn.ReLU())
            prev_dim = hidden_dim
        
        encoder.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        
        # Decoder
        decoder = []
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            decoder.append(nn.Linear(prev_dim, hidden_dim))
            decoder.append(nn.BatchNorm1d(hidden_dim))
            decoder.append(nn.ReLU())
            prev_dim = hidden_dim
        
        decoder.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)


class AutoencoderClassifier(nn.Module):
    """
    Autoencoder + Classifier
    
    Two-stage training:
    1. Pretrain autoencoder (unsupervised)
    2. Add classifier on top of encoder and fine-tune
    
    Architecture:
    Input → Encoder → Latent → Classifier → Class probabilities
                    ↓
                 Decoder → Reconstruction (only for pretraining)
    
    Advantages:
    - Leverage unlabeled data
    - Better initialization than random
    - Regularization: Encoder must preserve information
    """
    
    def __init__(self, autoencoder, latent_dim, n_classes):
        super(AutoencoderClassifier, self).__init__()
        
        self.encoder = autoencoder.encoder
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        output = self.classifier(encoded)
        return output


class DeepLearningTrainer:
    """
    Trainer class cho deep learning models
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    
    def train_mlp(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train MLP classifier
        
        Training procedure:
        1. Forward pass: Compute predictions
        2. Compute loss: Cross-entropy
        3. Backward pass: Compute gradients
        4. Update weights: Adam optimizer
        5. Early stopping: Stop if val loss doesn't improve
        
        Loss function (Cross-Entropy):
        L = -Σ y_i log(ŷ_i)
        
        For multi-class: L = -Σ Σ y_ic log(ŷ_ic)
        
        Adam optimizer:
        - Adaptive learning rate cho mỗi parameter
        - Combines momentum + RMSprop
        - m_t = β₁m_{t-1} + (1-β₁)g_t (momentum)
        - v_t = β₂v_{t-1} + (1-β₂)g_t² (adaptive learning rate)
        - θ_t = θ_{t-1} - α·m_t/√(v_t + ε)
        """
        print("\n" + "="*60)
        print("Training MLP Classifier")
        print("="*60)
        
        mlp_config = self.config['mlp']
        
        # Model
        input_dim = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_layers=mlp_config['hidden_layers'],
            n_classes=n_classes,
            dropout_rate=mlp_config['dropout_rate'],
            use_batchnorm=mlp_config['batch_norm']
        ).to(self.device)
        
        print(f"Model architecture:\n{self.model}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=mlp_config['learning_rate'],
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Data loaders
        train_dataset = OmicsDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=mlp_config['batch_size'], 
            shuffle=True
        )
        
        if X_val is not None and y_val is not None:
            val_dataset = OmicsDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=mlp_config['batch_size'], 
                shuffle=False
            )
        else:
            val_loader = None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = mlp_config['early_stopping_patience']
        
        for epoch in range(mlp_config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_f1, val_acc = self._validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_f1)
                self.history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_mlp_model.pth')
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{mlp_config['epochs']}: "
                          f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                          f"Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{mlp_config['epochs']}: Train Loss={train_loss:.4f}")
        
        # Load best model
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_mlp_model.pth'))
        
        print("="*60)
        return self.model
    
    def train_autoencoder(self, X_train, X_val=None):
        """
        Pretrain autoencoder (unsupervised)
        
        Loss: Mean Squared Error (Reconstruction loss)
        L = 1/n Σ ||x_i - x̂_i||²
        """
        print("\n" + "="*60)
        print("Pretraining Autoencoder")
        print("="*60)
        
        ae_config = self.config['autoencoder']
        
        # Model
        input_dim = X_train.shape[1]
        
        autoencoder = Autoencoder(
            input_dim=input_dim,
            encoder_layers=ae_config['encoder_layers'],
            latent_dim=ae_config['latent_dim'],
            decoder_layers=ae_config['decoder_layers']
        ).to(self.device)
        
        print(f"Autoencoder architecture:\n{autoencoder}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=ae_config['learning_rate'])
        
        # Data loaders
        train_tensor = torch.FloatTensor(X_train)
        train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target
        train_loader = DataLoader(train_dataset, batch_size=ae_config['batch_size'], shuffle=True)
        
        if X_val is not None:
            val_tensor = torch.FloatTensor(X_val)
            val_dataset = TensorDataset(val_tensor, val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=ae_config['batch_size'], shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        for epoch in range(ae_config['epochs']):
            autoencoder.train()
            train_loss = 0.0
            
            for batch_X, _ in train_loader:
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = autoencoder(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None and (epoch + 1) % 10 == 0:
                autoencoder.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, _ in val_loader:
                        batch_X = batch_X.to(self.device)
                        reconstructed = autoencoder(batch_X)
                        loss = criterion(reconstructed, batch_X)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                
                print(f"Epoch {epoch+1}/{ae_config['epochs']}: "
                      f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{ae_config['epochs']}: Train Loss={train_loss:.4f}")
        
        print("="*60)
        return autoencoder
    
    def train_autoencoder_classifier(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Autoencoder + Classifier
        
        Two-stage training:
        1. Pretrain autoencoder
        2. Add classifier and fine-tune
        """
        # Stage 1: Pretrain autoencoder
        autoencoder = self.train_autoencoder(X_train, X_val)
        
        # Stage 2: Train classifier
        print("\n" + "="*60)
        print("Training Classifier on Autoencoder Features")
        print("="*60)
        
        ae_config = self.config['autoencoder']
        n_classes = len(np.unique(y_train))
        
        self.model = AutoencoderClassifier(
            autoencoder=autoencoder,
            latent_dim=ae_config['latent_dim'],
            n_classes=n_classes
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Data loaders
        train_dataset = OmicsDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = OmicsDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        else:
            val_loader = None
        
        # Training loop (similar to MLP)
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(100):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss, val_f1, val_acc = self._validate(val_loader, criterion)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_ae_classifier.pth')
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        if val_loader is not None:
            self.model.load_state_dict(torch.load('best_ae_classifier.pth'))
        
        print("="*60)
        return self.model
    
    def _validate(self, val_loader, criterion):
        """Validation step"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_f1, val_acc
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
        
        return preds.cpu().numpy(), probs.cpu().numpy()
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # F1 Score
        if self.history['val_f1']:
            axes[1].plot(self.history['val_f1'], label='Val F1', color='green')
            axes[1].plot(self.history['val_acc'], label='Val Accuracy', color='blue')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Training History - Metrics')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    """Example usage"""
    from config import DL_CONFIG, DATASET_CONFIG
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = DATASET_CONFIG['n_samples']
    n_features = 1000
    n_classes = DATASET_CONFIG['n_classes']
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Train MLP
    trainer = DeepLearningTrainer(DL_CONFIG)
    model = trainer.train_mlp(X_train, y_train, X_val, y_val)
    
    # Evaluate
    y_pred, y_proba = trainer.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest F1: {f1:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Plot history
    trainer.plot_training_history()
