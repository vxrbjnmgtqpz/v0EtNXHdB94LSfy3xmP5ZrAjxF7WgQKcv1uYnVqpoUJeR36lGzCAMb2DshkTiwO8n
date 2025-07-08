#!/usr/bin/env python3
"""
PNBTR PyTorch Trainer - Phase 2
Real gradient-based training with advanced loss functions and optimization.
Implements the "no rest until mastery" philosophy with proper PyTorch training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
import yaml

class PNBTRAdvancedLoss(nn.Module):
    """
    Advanced loss function combining multiple signal quality metrics.
    Implements the real metrics that matter: SDR, spectral fidelity, envelope preservation.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # Loss weights from config
        loss_weights = self.config.get("optimization", {}).get("loss_weights", {})
        self.mse_weight = loss_weights.get("mse_loss", 0.40)
        self.spectral_weight = loss_weights.get("spectral_loss", 0.35)
        self.envelope_weight = loss_weights.get("envelope_loss", 0.25)
        
        print(f"   📊 Loss weights: MSE={self.mse_weight:.2f}, Spectral={self.spectral_weight:.2f}, Envelope={self.envelope_weight:.2f}")
    
    def forward(self, predicted, target):
        """
        Compute composite loss combining time and frequency domain metrics.
        
        Args:
            predicted: Model output signal
            target: Ground truth signal
            
        Returns:
            torch.Tensor: Composite loss value
        """
        # Ensure tensors are on same device and have same shape
        if predicted.shape != target.shape:
            min_len = min(predicted.size(-1), target.size(-1))
            predicted = predicted[..., :min_len]
            target = target[..., :min_len]
        
        # 1. Time-domain MSE loss
        mse_loss = F.mse_loss(predicted, target)
        
        # 2. Spectral loss (FFT-based)
        spectral_loss = self._spectral_loss(predicted, target)
        
        # 3. Envelope loss (dynamic preservation)
        envelope_loss = self._envelope_loss(predicted, target)
        
        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.spectral_weight * spectral_loss +
            self.envelope_weight * envelope_loss
        )
        
        return total_loss
    
    def _spectral_loss(self, predicted, target):
        """Spectral fidelity loss using FFT comparison"""
        # Apply FFT to both signals
        pred_fft = torch.fft.rfft(predicted, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Magnitude spectrum loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        magnitude_loss = F.mse_loss(pred_mag, target_mag)
        
        # Phase coherence loss (important for reconstruction quality)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Phase difference (wrapped to [-π, π])
        phase_diff = torch.remainder(pred_phase - target_phase + np.pi, 2 * np.pi) - np.pi
        phase_loss = torch.mean(phase_diff ** 2)
        
        return magnitude_loss + 0.1 * phase_loss  # Weight phase less than magnitude
    
    def _envelope_loss(self, predicted, target):
        """Envelope preservation loss for dynamic range"""
        # Calculate envelope using Hilbert transform approximation
        pred_envelope = self._calculate_envelope(predicted)
        target_envelope = self._calculate_envelope(target)
        
        return F.mse_loss(pred_envelope, target_envelope)
    
    def _calculate_envelope(self, signal):
        """Calculate signal envelope using moving window RMS"""
        # Use causal moving window for real-time compatibility
        window_size = 64  # Small window for responsive envelope tracking
        
        # Squared signal
        signal_squared = signal ** 2
        
        # Create causal moving average kernel
        kernel = torch.ones(1, 1, window_size, device=signal.device) / window_size
        
        # Ensure signal has proper dimensions for conv1d
        if signal.dim() == 1:
            signal_squared = signal_squared.unsqueeze(0).unsqueeze(0)
        elif signal.dim() == 2:
            signal_squared = signal_squared.unsqueeze(1)
        
        # Apply causal convolution
        padding = window_size - 1
        envelope_squared = F.conv1d(F.pad(signal_squared, (padding, 0)), kernel)
        
        # Remove padding and take square root
        envelope = torch.sqrt(envelope_squared[..., :signal.size(-1)] + 1e-8)
        
        # Return to original shape
        if signal.dim() == 1:
            envelope = envelope.squeeze()
        elif signal.dim() == 2:
            envelope = envelope.squeeze(1)
        
        return envelope

class PNBTRPyTorchTrainer:
    """
    PyTorch-based trainer for PNBTR models.
    Implements proper gradient-based optimization with advanced scheduling.
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or {}
        self.device = model.device if hasattr(model, 'device') else torch.device('cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self._setup_optimizer()
        
        # Initialize loss function
        self.criterion = PNBTRAdvancedLoss(config)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.plateau_count = 0
        
        # Get training configuration
        training_config = config.get("training_loop", {})
        self.max_epochs = training_config.get("max_epochs", 1000)
        self.patience = training_config.get("patience", 50)
        
        print(f"🚀 PyTorch trainer initialized:")
        print(f"   📱 Device: {self.device}")
        print(f"   ⚙️  Optimizer: {type(self.optimizer).__name__}")
        print(f"   📊 Loss: Advanced composite loss")
        print(f"   🎯 Max epochs: {self.max_epochs}, Patience: {self.patience}")
    
    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        opt_config = self.config.get("optimization", {})
        optimizer_name = opt_config.get("optimizer", "adam").lower()
        
        # Learning rate settings
        lr_config = opt_config.get("learning_rate", {})
        initial_lr = lr_config.get("initial", 0.001)
        weight_decay = opt_config.get("weight_decay", 0.0001)
        
        # Create optimizer
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=initial_lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            print(f"⚠️  Unknown optimizer {optimizer_name}, using Adam")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay
            )
        
        # Learning rate scheduler
        scheduler_type = lr_config.get("scheduler", "step")
        if scheduler_type == "step":
            step_size = lr_config.get("step_size", 100)
            gamma = lr_config.get("gamma", 0.95)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs
            )
        elif scheduler_type == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=20
            )
        else:
            self.scheduler = None
    
    def train_single_sample(self, input_signal, target_signal):
        """
        Train on a single input-target pair until mastery achieved.
        Implements "no rest until mastery" philosophy.
        
        Args:
            input_signal: Degraded/incomplete audio array
            target_signal: Ground truth reconstruction target
            
        Returns:
            dict: Training results including final accuracy
        """
        # Convert to tensors
        if isinstance(input_signal, np.ndarray):
            input_tensor = torch.from_numpy(input_signal.astype(np.float32))
        else:
            input_tensor = input_signal.float()
        
        if isinstance(target_signal, np.ndarray):
            target_tensor = torch.from_numpy(target_signal.astype(np.float32))
        else:
            target_tensor = target_signal.float()
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        
        # Ensure proper batch dimensions
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)
        
        # Training loop
        self.model.train()
        epoch = 0
        best_loss = float('inf')
        plateau_count = 0
        
        print(f"🏃 Training single sample ({len(input_signal)} samples)...")
        
        while epoch < self.max_epochs:
            # Forward pass
            self.optimizer.zero_grad()
            predicted = self.model(input_tensor)
            loss = self.criterion(predicted, target_tensor)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config.get("optimization", {}), "gradient_clipping"):
                clip_value = self.config["optimization"]["gradient_clipping"]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            self.optimizer.step()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.item())
                else:
                    self.scheduler.step()
            
            # Check for improvement
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Progress reporting
            if epoch % 50 == 0 or epoch < 10:
                print(f"   Epoch {epoch:4d}: Loss = {current_loss:.6f}, "
                      f"LR = {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if plateau_count >= self.patience:
                print(f"   🛑 Early stopping after {epoch} epochs (no improvement)")
                break
            
            # Mastery check - convert to our scoring metrics
            if epoch % 10 == 0:
                accuracy = self._evaluate_accuracy(predicted, target_tensor)
                if accuracy >= 0.90:  # 90% mastery threshold
                    print(f"   🏆 MASTERY ACHIEVED after {epoch} epochs! Accuracy: {accuracy:.3f}")
                    break
            
            epoch += 1
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predicted = self.model(input_tensor)
            final_loss = self.criterion(final_predicted, target_tensor)
            final_accuracy = self._evaluate_accuracy(final_predicted, target_tensor)
        
        return {
            "epochs_trained": epoch,
            "final_loss": final_loss.item(),
            "accuracy": final_accuracy,
            "mastery_achieved": final_accuracy >= 0.90,
            "early_stopped": plateau_count >= self.patience
        }
    
    def _evaluate_accuracy(self, predicted, target):
        """
        Evaluate reconstruction accuracy using PNBTR metrics.
        Converts signals to numpy and uses existing scoring system.
        """
        try:
            # Convert to numpy
            pred_np = predicted.squeeze().cpu().numpy().astype(np.float64)
            target_np = target.squeeze().cpu().numpy().astype(np.float64)
            
            # Use existing metrics evaluation
            from .loss_functions import evaluate_metrics
            from ..metrics.scoring import score_accuracy
            
            metrics = evaluate_metrics(pred_np, target_np, sample_rate=48000)
            accuracy = score_accuracy(metrics)
            
            return accuracy
            
        except Exception as e:
            # Fallback to simple correlation if metrics fail
            pred_np = predicted.squeeze().cpu().numpy()
            target_np = target.squeeze().cpu().numpy()
            
            # Pearson correlation as simple accuracy measure
            correlation = np.corrcoef(pred_np, target_np)[0, 1]
            return max(0.0, correlation)  # Clamp to [0, 1]
    
    def step(self, loss):
        """Legacy interface for compatibility with existing training loop"""
        # This is handled internally by train_single_sample now
        pass
    
    def save_checkpoint(self, path, epoch, accuracy):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"📁 Checkpoint loaded: {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('accuracy', 0.0)

def create_pytorch_trainer(model, config=None):
    """
    Factory function to create PyTorch trainer.
    Integrates with existing PNBTR training system.
    """
    # Check if model is PyTorch-based
    if not hasattr(model, 'state_dict'):
        raise ValueError("Model must be a PyTorch nn.Module for PyTorch training")
    
    return PNBTRPyTorchTrainer(model, config)

# Enhanced training loop that uses PyTorch trainer when available
def enhanced_train_single_sample(input_signal, target_signal, model, config=None):
    """
    Enhanced training function that automatically selects PyTorch trainer for PyTorch models.
    Falls back to original training loop for dummy models.
    """
    # Check if this is a PyTorch model
    if hasattr(model, 'state_dict'):
        # Use PyTorch trainer
        trainer = create_pytorch_trainer(model, config)
        return trainer.train_single_sample(input_signal, target_signal)
    else:
        # Use original training loop for dummy models
        from .train_loop import PNBTRTrainingLoop
        trainer = PNBTRTrainingLoop()
        return trainer.train_single_sample(input_signal, target_signal, model)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR PyTorch Trainer Test")
    
    try:
        # Test with PyTorch model
        from .pytorch_models import create_pytorch_model
        from .waveform_utils import generate_test_signal
        
        # Create test data
        input_signal = generate_test_signal(1000, 48000, "complex")
        target_signal = input_signal + np.random.normal(0, 0.01, len(input_signal))  # Slight noise
        
        # Create PyTorch model
        model = create_pytorch_model("mlp", len(input_signal))
        
        # Create trainer
        trainer = create_pytorch_trainer(model)
        
        # Test training
        print("\n🏃 Testing PyTorch training...")
        result = trainer.train_single_sample(input_signal, target_signal)
        
        print(f"\n📊 Training Results:")
        print(f"   Epochs: {result['epochs_trained']}")
        print(f"   Final Loss: {result['final_loss']:.6f}")
        print(f"   Accuracy: {result['accuracy']:.3f}")
        print(f"   Mastery: {'✅' if result['mastery_achieved'] else '❌'}")
        
        # Test enhanced training function
        print("\n🔄 Testing enhanced training function...")
        enhanced_result = enhanced_train_single_sample(input_signal, target_signal, model)
        print(f"   Enhanced training accuracy: {enhanced_result['accuracy']:.3f}")
        
        print("\n✅ PyTorch trainer test complete")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 