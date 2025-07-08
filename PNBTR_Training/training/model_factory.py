#!/usr/bin/env python3
"""
PNBTR Model Factory - Phase 2 Enhanced
Creates and initializes different neural network architectures for
predictive signal reconstruction training.

Phase 2: Now supports real PyTorch models alongside dummy models.
"""

import numpy as np
import yaml
from pathlib import Path

# Try to import PyTorch models
try:
    from .pytorch_models import create_pytorch_model
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

def load_model_config(config_path=None):
    """Load model configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "training_params.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  Error loading model config: {e}, using defaults")
        return get_default_config()

def get_default_config():
    """Default configuration if YAML loading fails"""
    return {
        "model": {
            "type": "mlp",
            "mlp": {
                "hidden_layers": [512, 256, 128, 64],
                "activation": "relu",
                "dropout": 0.1,
                "batch_norm": True
            }
        },
        "optimization": {
            "learning_rate": {"initial": 0.001}
        }
    }

def create_pnbtr_model(model_type=None, config=None, input_size=1024, use_pytorch=True):
    """
    Factory function to create PNBTR models.
    
    Args:
        model_type: "mlp", "conv1d", "hybrid", "transformer", "dummy" 
        config: Configuration dict (loaded from YAML if None)
        input_size: Input signal length
        use_pytorch: Use PyTorch models if available (Phase 2)
        
    Returns:
        Model instance compatible with training loop
    """
    if config is None:
        config = load_model_config()
    
    if model_type is None:
        model_type = config.get("model", {}).get("type", "mlp")
    
    print(f"🏗️  Creating PNBTR model: {model_type}")
    
    # Phase 2: Try PyTorch models first
    if use_pytorch and PYTORCH_AVAILABLE and model_type != "dummy":
        try:
            pytorch_model = create_pytorch_model(model_type, input_size, config)
            print(f"   ✅ Using PyTorch implementation")
            return pytorch_model
        except Exception as e:
            print(f"   ⚠️  PyTorch model failed: {e}")
            print(f"   🔄 Falling back to dummy model")
    
    # Fallback to dummy models
    if model_type == "mlp":
        return MLPModel(config, input_size)
    elif model_type == "conv1d":
        return Conv1DModel(config, input_size)
    elif model_type == "hybrid":
        return HybridModel(config, input_size)
    elif model_type == "transformer":
        return TransformerModel(config, input_size)
    elif model_type == "dummy":
        return DummyModel(input_size)
    else:
        print(f"⚠️  Unknown model type: {model_type}, using dummy model")
        return DummyModel(input_size)

class BasePNBTRModel:
    """Base class for all PNBTR models"""
    
    def __init__(self, config, input_size):
        self.config = config
        self.input_size = input_size
        self.output_size = input_size  # Same length output
        self.training_mode = True
        
    def predict(self, input_signal):
        """Standard prediction interface"""
        if hasattr(self, 'forward'):
            return self.forward(input_signal)
        else:
            return self._internal_predict(input_signal)
    
    def step(self, loss):
        """Training step (placeholder - real implementation model-specific)"""
        print(f"🔄 Training step with loss: {loss:.6f}")
        
    def save(self, path):
        """Save model weights"""
        print(f"💾 Model save requested: {path}")
        
    def load(self, path):
        """Load model weights"""
        print(f"📁 Model load requested: {path}")

class DummyModel(BasePNBTRModel):
    """
    Dummy model for testing - applies simple signal processing.
    Good for testing the training pipeline without ML dependencies.
    """
    
    def __init__(self, input_size):
        super().__init__({}, input_size)
        self.name = "PNBTR_Dummy"
        self.smoothing_factor = 0.1
        self.gain = 1.0
        
    def _internal_predict(self, input_signal):
        """Simple smoothing + slight gain adjustment"""
        # Convert to numpy if needed
        if hasattr(input_signal, 'numpy'):
            signal = input_signal.numpy()
        else:
            signal = np.asarray(input_signal, dtype=np.float64)
        
        # Simple smoothing filter (moving average)
        smoothed = np.copy(signal)
        for i in range(1, len(signal)):
            smoothed[i] = (
                (1 - self.smoothing_factor) * signal[i] + 
                self.smoothing_factor * smoothed[i-1]
            )
        
        # Slight gain adjustment
        output = smoothed * self.gain
        
        # Ensure same length as input
        return output[:len(input_signal)]
    
    def step(self, loss):
        """Dummy training step - slightly adjust parameters based on loss"""
        if loss > 0.5:
            # High loss - reduce smoothing
            self.smoothing_factor = max(0.01, self.smoothing_factor * 0.99)
        elif loss < 0.1:
            # Low loss - can increase smoothing slightly
            self.smoothing_factor = min(0.3, self.smoothing_factor * 1.001)
        
        # Adjust gain towards unity
        if self.gain > 1.0:
            self.gain = max(1.0, self.gain * 0.999)
        elif self.gain < 1.0:
            self.gain = min(1.0, self.gain * 1.001)
        
        super().step(loss)

class MLPModel(BasePNBTRModel):
    """
    Multi-Layer Perceptron model for PNBTR.
    Good for direct waveform prediction.
    """
    
    def __init__(self, config, input_size):
        super().__init__(config, input_size)
        self.name = "PNBTR_MLP_Dummy"
        
        mlp_config = config.get("model", {}).get("mlp", {})
        self.hidden_layers = mlp_config.get("hidden_layers", [512, 256, 128, 64])
        self.activation = mlp_config.get("activation", "relu")
        self.dropout = mlp_config.get("dropout", 0.1)
        
        # Initialize weights (placeholder - real implementation would use PyTorch)
        self.weights = self._initialize_weights()
        
        print(f"   📐 MLP layers: {[input_size] + self.hidden_layers + [input_size]} (dummy)")
        
    def _initialize_weights(self):
        """Initialize random weights for the network"""
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        weights = []
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            fan_in, fan_out = layers[i], layers[i+1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            weight = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias = np.zeros(fan_out)
            weights.append((weight, bias))
            
        return weights
    
    def _internal_predict(self, input_signal):
        """Forward pass through MLP"""
        # Convert to numpy if needed
        if hasattr(input_signal, 'numpy'):
            x = input_signal.numpy().reshape(-1)
        else:
            x = np.asarray(input_signal, dtype=np.float64).reshape(-1)
        
        # Ensure correct input size
        if len(x) != self.input_size:
            if len(x) > self.input_size:
                x = x[:self.input_size]
            else:
                padding = np.zeros(self.input_size - len(x))
                x = np.concatenate([x, padding])
        
        # Forward pass
        current = x
        for i, (weight, bias) in enumerate(self.weights[:-1]):
            # Linear transformation
            current = np.dot(current, weight) + bias
            
            # Activation function
            if self.activation == "relu":
                current = np.maximum(0, current)
            elif self.activation == "tanh":
                current = np.tanh(current)
            elif self.activation == "leaky_relu":
                current = np.where(current > 0, current, current * 0.01)
            
            # Dropout simulation (random zeroing during training)
            if self.training_mode and self.dropout > 0:
                mask = np.random.random(current.shape) > self.dropout
                current = current * mask / (1 - self.dropout)
        
        # Output layer (no activation)
        weight, bias = self.weights[-1]
        output = np.dot(current, weight) + bias
        
        return output
    
    def step(self, loss):
        """Simulate gradient descent step"""
        # Simulate weight updates (real implementation would use proper backprop)
        learning_rate = 0.001
        
        for i, (weight, bias) in enumerate(self.weights):
            # Add small random perturbations (simulated gradients)
            grad_weight = np.random.normal(0, loss * 0.01, weight.shape)
            grad_bias = np.random.normal(0, loss * 0.01, bias.shape)
            
            # Update weights
            self.weights[i] = (
                weight - learning_rate * grad_weight,
                bias - learning_rate * grad_bias
            )
        
        super().step(loss)

class Conv1DModel(BasePNBTRModel):
    """
    1D Convolutional model for PNBTR.
    Good for temporal pattern recognition in audio.
    """
    
    def __init__(self, config, input_size):
        super().__init__(config, input_size)
        self.name = "PNBTR_Conv1D_Dummy"
        
        conv_config = config.get("model", {}).get("conv1d", {})
        self.channels = conv_config.get("channels", [64, 128, 256, 128, 64])
        self.kernel_sizes = conv_config.get("kernel_sizes", [7, 5, 3, 3, 7])
        
        print(f"   📐 Conv1D channels: {[1] + self.channels + [1]} (dummy)")
        print(f"   📐 Kernel sizes: {self.kernel_sizes}")
        
    def _internal_predict(self, input_signal):
        """Simplified convolution simulation"""
        # Convert to numpy
        if hasattr(input_signal, 'numpy'):
            signal = input_signal.numpy()
        else:
            signal = np.asarray(input_signal, dtype=np.float64)
        
        # Simple smoothing convolution as placeholder
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Simple smoothing kernel
        
        # Apply convolution
        output = np.convolve(signal, kernel, mode='same')
        
        return output
    
    def step(self, loss):
        """Simulate conv layer training"""
        print(f"🔄 Conv1D training step (dummy)")
        super().step(loss)

class HybridModel(BasePNBTRModel):
    """
    Hybrid MLP + Conv1D model.
    Combines temporal feature extraction with nonlinear prediction.
    """
    
    def __init__(self, config, input_size):
        super().__init__(config, input_size)
        self.name = "PNBTR_Hybrid_Dummy"
        
        # Initialize both components
        self.conv_model = Conv1DModel(config, input_size)
        self.mlp_model = MLPModel(config, input_size // 2)  # Reduced size after conv
        
        print(f"   📐 Hybrid: Conv1D + MLP (dummy)")
        
    def _internal_predict(self, input_signal):
        """Two-stage prediction: Conv1D -> MLP"""
        # Stage 1: Convolution
        conv_output = self.conv_model._internal_predict(input_signal)
        
        # Stage 2: Downsample and process with MLP
        downsampled = conv_output[::2]  # Simple 2x downsampling
        mlp_output = self.mlp_model._internal_predict(downsampled)
        
        # Upsample back to original size
        upsampled = np.repeat(mlp_output, 2)[:len(input_signal)]
        
        return upsampled
    
    def step(self, loss):
        """Train both components"""
        self.conv_model.step(loss)
        self.mlp_model.step(loss)
        super().step(loss)

class TransformerModel(BasePNBTRModel):
    """
    Transformer model for PNBTR (experimental).
    Uses attention mechanisms for long-range dependencies.
    """
    
    def __init__(self, config, input_size):
        super().__init__(config, input_size)
        self.name = "PNBTR_Transformer_Dummy"
        
        transformer_config = config.get("model", {}).get("transformer", {})
        self.d_model = transformer_config.get("d_model", 256)
        self.nhead = transformer_config.get("nhead", 8)
        
        print(f"   📐 Transformer: d_model={self.d_model}, heads={self.nhead} (dummy)")
        
    def _internal_predict(self, input_signal):
        """Simplified transformer simulation"""
        # Placeholder: simple smoothing with different window sizes (simulating attention)
        if hasattr(input_signal, 'numpy'):
            signal = input_signal.numpy()
        else:
            signal = np.asarray(input_signal, dtype=np.float64)
        
        # Multi-scale smoothing (simulating multi-head attention)
        outputs = []
        for window_size in [3, 5, 7, 11]:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(signal, kernel, mode='same')
            outputs.append(smoothed)
        
        # Combine outputs (simulating attention aggregation)
        output = np.mean(outputs, axis=0)
        
        return output
    
    def step(self, loss):
        """Simulate transformer training"""
        print(f"🔄 Transformer training step (dummy)")
        super().step(loss)

# Utility functions

def get_model_info(model):
    """Get information about a model instance"""
    info = {
        "name": getattr(model, 'name', 'Unknown'),
        "input_size": getattr(model, 'input_size', 0),
        "output_size": getattr(model, 'output_size', 0),
        "parameters": estimate_parameter_count(model),
        "is_pytorch": hasattr(model, 'state_dict')  # Check if it's a PyTorch model
    }
    return info

def estimate_parameter_count(model):
    """Estimate parameter count for model"""
    # PyTorch models
    if hasattr(model, 'parameters'):
        try:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            pass
    
    # Dummy models with weights list
    if hasattr(model, 'weights') and isinstance(model.weights, list):
        count = 0
        for weight, bias in model.weights:
            count += np.prod(weight.shape) + np.prod(bias.shape)
        return count
    
    return 0  # Unknown

def benchmark_model(model, input_size=1024, num_runs=100):
    """Benchmark model prediction speed"""
    import time
    
    # Generate test signal
    test_signal = np.random.normal(0, 0.1, input_size).astype(np.float64)
    
    # Warm up
    for _ in range(5):
        _ = model.predict(test_signal)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(test_signal)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    
    return {
        "avg_prediction_time_ms": avg_time_ms,
        "predictions_per_second": 1000 / avg_time_ms,
        "meets_realtime_target": avg_time_ms < 1.0
    }

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Model Factory Test - Phase 2")
    print(f"PyTorch Available: {PYTORCH_AVAILABLE}")
    
    # Test different model types
    models_to_test = ["dummy", "mlp", "conv1d", "hybrid"]
    
    for model_type in models_to_test:
        print(f"\n🏗️  Testing {model_type} model:")
        
        # Test both PyTorch and dummy versions
        for use_pytorch in [True, False]:
            if not use_pytorch or not PYTORCH_AVAILABLE:
                if use_pytorch:
                    continue  # Skip if PyTorch not available
                print(f"   🔧 Dummy implementation:")
            else:
                print(f"   🚀 PyTorch implementation:")
            
            try:
                model = create_pnbtr_model(model_type, input_size=1024, use_pytorch=use_pytorch)
                info = get_model_info(model)
                
                print(f"      Name: {info['name']}")
                print(f"      Parameters: {info['parameters']:,}")
                print(f"      PyTorch: {info['is_pytorch']}")
                
                # Test prediction
                test_input = np.random.normal(0, 0.1, 1024).astype(np.float64)
                output = model.predict(test_input)
                
                print(f"      Input shape: {test_input.shape}")
                print(f"      Output shape: {output.shape}")
                print(f"      Output range: [{np.min(output):.4f}, {np.max(output):.4f}]")
                
                # Test training step
                model.step(0.5)
                
                # Quick benchmark
                benchmark = benchmark_model(model, num_runs=10)
                print(f"      Speed: {benchmark['avg_prediction_time_ms']:.2f}ms")
                print(f"      Real-time: {'✅' if benchmark['meets_realtime_target'] else '❌'}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    print("\n✅ Model factory Phase 2 test complete") 