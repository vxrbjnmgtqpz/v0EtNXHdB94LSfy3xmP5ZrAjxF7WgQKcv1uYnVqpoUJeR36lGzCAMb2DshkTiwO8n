#!/usr/bin/env python3
"""
PNBTR PyTorch Models - Phase 2
Real neural network implementations for signal reconstruction training.
Implements the anti-float philosophy with proper 24-bit precision handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import yaml

class PNBTRBaseTorchModel(nn.Module):
    """Base class for all PNBTR PyTorch models"""
    
    def __init__(self, input_size, config=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size  # Same length output for reconstruction
        self.config = config or {}
        self.name = "PNBTR_Base"
        
        # Training state
        self.training_mode = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def predict(self, input_signal):
        """Standard prediction interface compatible with training loop"""
        # Convert numpy to tensor if needed
        if isinstance(input_signal, np.ndarray):
            x = torch.from_numpy(input_signal.astype(np.float32))
        else:
            x = input_signal.float()
        
        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        elif x.dim() > 2:
            x = x.flatten(1)  # Flatten to (batch, features)
        
        # Move to device
        x = x.to(self.device)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        
        # Convert back to numpy
        if output.dim() > 1:
            output = output.squeeze(0)  # Remove batch dimension
        
        return output.cpu().numpy().astype(np.float64)
    
    def step(self, loss):
        """Training step placeholder - real training handled by optimizer"""
        if hasattr(self, 'optimizer'):
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def save(self, path):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'input_size': self.input_size
        }, path)
        print(f"💾 Model saved: {path}")
    
    def load(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"📁 Model loaded: {path}")

class PNBTRMLPModel(PNBTRBaseTorchModel):
    """
    Multi-Layer Perceptron for direct waveform reconstruction.
    Optimized for sample-level prediction with anti-float precision.
    """
    
    def __init__(self, input_size, config=None):
        super().__init__(input_size, config)
        self.name = "PNBTR_MLP_PyTorch"
        
        # Get MLP configuration
        mlp_config = config.get("model", {}).get("mlp", {}) if config else {}
        self.hidden_layers = mlp_config.get("hidden_layers", [512, 256, 128, 64])
        self.activation = mlp_config.get("activation", "relu")
        self.dropout_rate = mlp_config.get("dropout", 0.1)
        self.batch_norm = mlp_config.get("batch_norm", True)
        
        # Build network layers
        layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(self.hidden_layers):
            # Linear layer
            layers.append(nn.Linear(current_size, hidden_size))
            
            # Batch normalization (optional)
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if self.activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01, inplace=True))
            elif self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "gelu":
                layers.append(nn.GELU())
            
            # Dropout
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            current_size = hidden_size
        
        # Output layer (no activation - linear reconstruction)
        layers.append(nn.Linear(current_size, self.output_size))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        print(f"   📐 MLP: {len(self.hidden_layers)} hidden layers, {self._count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Forward pass through MLP"""
        return self.network(x)

class PNBTRConv1DModel(PNBTRBaseTorchModel):
    """
    1D Convolutional model for temporal pattern recognition.
    Optimized for preserving signal structure and transients.
    """
    
    def __init__(self, input_size, config=None):
        super().__init__(input_size, config)
        self.name = "PNBTR_Conv1D_PyTorch"
        
        # Get Conv1D configuration
        conv_config = config.get("model", {}).get("conv1d", {}) if config else {}
        self.channels = conv_config.get("channels", [64, 128, 256, 128, 64])
        self.kernel_sizes = conv_config.get("kernel_sizes", [7, 5, 3, 3, 7])
        self.activation = conv_config.get("activation", "relu")
        self.dropout_rate = conv_config.get("dropout", 0.15)
        
        # Ensure we have matching channels and kernel sizes
        if len(self.channels) != len(self.kernel_sizes):
            self.kernel_sizes = [5] * len(self.channels)
        
        # Build encoder (downsampling)
        encoder_layers = []
        current_channels = 1  # Input is mono signal
        
        for i, (out_channels, kernel_size) in enumerate(zip(self.channels, self.kernel_sizes)):
            # Convolution
            padding = kernel_size // 2  # Same padding
            encoder_layers.append(nn.Conv1d(current_channels, out_channels, kernel_size, 
                                          padding=padding, bias=False))
            encoder_layers.append(nn.BatchNorm1d(out_channels))
            
            # Activation
            if self.activation == "relu":
                encoder_layers.append(nn.ReLU(inplace=True))
            elif self.activation == "leaky_relu":
                encoder_layers.append(nn.LeakyReLU(0.01, inplace=True))
            
            # Dropout
            if self.dropout_rate > 0:
                encoder_layers.append(nn.Dropout1d(self.dropout_rate))
            
            current_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (upsampling) - reverse the channel order
        decoder_layers = []
        reversed_channels = list(reversed(self.channels[:-1])) + [1]  # End with 1 channel
        
        for i, (out_channels, kernel_size) in enumerate(zip(reversed_channels, reversed(self.kernel_sizes))):
            padding = kernel_size // 2
            decoder_layers.append(nn.Conv1d(current_channels, out_channels, kernel_size,
                                          padding=padding, bias=False))
            
            # No activation on final layer
            if i < len(reversed_channels) - 1:
                decoder_layers.append(nn.BatchNorm1d(out_channels))
                if self.activation == "relu":
                    decoder_layers.append(nn.ReLU(inplace=True))
                elif self.activation == "leaky_relu":
                    decoder_layers.append(nn.LeakyReLU(0.01, inplace=True))
                
                if self.dropout_rate > 0:
                    decoder_layers.append(nn.Dropout1d(self.dropout_rate))
            
            current_channels = out_channels
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        print(f"   📐 Conv1D: {len(self.channels)} layers, {self._count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize convolutional weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
    
    def _count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Forward pass through encoder-decoder"""
        # Reshape for Conv1d: (batch, channels, sequence)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(encoded)
        
        # Remove channel dimension and return to original shape
        if decoded.dim() == 3 and decoded.size(1) == 1:
            decoded = decoded.squeeze(1)
        
        return decoded

class PNBTRHybridModel(PNBTRBaseTorchModel):
    """
    Hybrid Conv1D + MLP model.
    Combines temporal feature extraction with nonlinear prediction.
    """
    
    def __init__(self, input_size, config=None):
        super().__init__(input_size, config)
        self.name = "PNBTR_Hybrid_PyTorch"
        
        # Get hybrid configuration
        hybrid_config = config.get("model", {}).get("hybrid", {}) if config else {}
        self.conv_layers = hybrid_config.get("conv_layers", 3)
        self.conv_channels = hybrid_config.get("conv_channels", [32, 64, 32])
        self.conv_kernels = hybrid_config.get("conv_kernels", [5, 3, 5])
        self.mlp_layers = hybrid_config.get("mlp_layers", [256, 128])
        
        # Ensure proper lengths
        if len(self.conv_channels) != self.conv_layers:
            self.conv_channels = [32, 64, 32][:self.conv_layers]
        if len(self.conv_kernels) != self.conv_layers:
            self.conv_kernels = [5] * self.conv_layers
        
        # Build convolutional feature extractor
        conv_layers = []
        current_channels = 1
        
        for i in range(self.conv_layers):
            out_channels = self.conv_channels[i]
            kernel_size = self.conv_kernels[i]
            padding = kernel_size // 2
            
            conv_layers.append(nn.Conv1d(current_channels, out_channels, kernel_size, padding=padding))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout1d(0.1))
            
            current_channels = out_channels
        
        self.conv_extractor = nn.Sequential(*conv_layers)
        
        # Build MLP predictor
        # After conv, we have (batch, channels, sequence) -> flatten to (batch, channels * sequence)
        mlp_input_size = current_channels * input_size
        
        mlp_layers = []
        current_size = mlp_input_size
        
        for hidden_size in self.mlp_layers:
            mlp_layers.append(nn.Linear(current_size, hidden_size))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(0.1))
            current_size = hidden_size
        
        # Output layer
        mlp_layers.append(nn.Linear(current_size, input_size))
        
        self.mlp_predictor = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        print(f"   📐 Hybrid: {self.conv_layers} conv + {len(self.mlp_layers)} MLP, {self._count_parameters():,} parameters")
    
    def _initialize_weights(self):
        """Initialize weights for both conv and linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Forward pass: Conv feature extraction -> MLP prediction"""
        # Reshape for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Extract features with convolution
        conv_features = self.conv_extractor(x)
        
        # Flatten for MLP
        batch_size = conv_features.size(0)
        flattened = conv_features.view(batch_size, -1)
        
        # Predict with MLP
        output = self.mlp_predictor(flattened)
        
        return output

class PNBTRTransformerModel(PNBTRBaseTorchModel):
    """
    Transformer model for PNBTR (experimental).
    Uses self-attention for long-range temporal dependencies.
    """
    
    def __init__(self, input_size, config=None):
        super().__init__(input_size, config)
        self.name = "PNBTR_Transformer_PyTorch"
        
        # Get transformer configuration
        transformer_config = config.get("model", {}).get("transformer", {}) if config else {}
        self.d_model = transformer_config.get("d_model", 256)
        self.nhead = transformer_config.get("nhead", 8)
        self.num_layers = transformer_config.get("num_layers", 4)
        self.dim_feedforward = transformer_config.get("dim_feedforward", 1024)
        
        # Input/output projection
        self.input_projection = nn.Linear(1, self.d_model)  # Project each sample to d_model
        self.output_projection = nn.Linear(self.d_model, 1)  # Project back to sample
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(input_size, self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        print(f"   📐 Transformer: d_model={self.d_model}, heads={self.nhead}, layers={self.num_layers}, {self._count_parameters():,} parameters")
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def _initialize_weights(self):
        """Initialize transformer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        """Forward pass through transformer"""
        # Reshape input: (batch, sequence) -> (batch, sequence, 1)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, sequence, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Project back to output
        x = self.output_projection(x)  # (batch, sequence, 1)
        
        # Remove feature dimension
        x = x.squeeze(-1)  # (batch, sequence)
        
        return x

def create_pytorch_model(model_type, input_size, config=None):
    """
    Factory function to create PyTorch PNBTR models.
    Updated version of model_factory.create_pnbtr_model for Phase 2.
    """
    if config is None:
        # Load default config
        try:
            config_path = Path(__file__).parent.parent / "config" / "training_params.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except:
            config = {}
    
    print(f"🏗️  Creating PyTorch PNBTR model: {model_type}")
    
    if model_type == "mlp":
        return PNBTRMLPModel(input_size, config)
    elif model_type == "conv1d":
        return PNBTRConv1DModel(input_size, config)
    elif model_type == "hybrid":
        return PNBTRHybridModel(input_size, config)
    elif model_type == "transformer":
        return PNBTRTransformerModel(input_size, config)
    else:
        raise ValueError(f"Unknown PyTorch model type: {model_type}")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR PyTorch Models Test")
    
    # Test signal
    input_size = 1024
    test_signal = torch.randn(input_size) * 0.1
    
    # Test different model types
    model_types = ["mlp", "conv1d", "hybrid", "transformer"]
    
    for model_type in model_types:
        print(f"\n🏗️  Testing {model_type} model:")
        
        try:
            model = create_pytorch_model(model_type, input_size)
            
            # Test forward pass
            with torch.no_grad():
                output = model(test_signal.unsqueeze(0))  # Add batch dimension
            
            print(f"   ✅ Forward pass successful")
            print(f"   📊 Input shape: {test_signal.shape}")
            print(f"   📊 Output shape: {output.shape}")
            print(f"   🔢 Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test prediction interface
            pred_output = model.predict(test_signal.numpy())
            print(f"   📈 Prediction interface working: {pred_output.shape}")
            
        except Exception as e:
            print(f"   ❌ Error testing {model_type}: {e}")
    
    print("\n✅ PyTorch models test complete") 