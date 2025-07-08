#!/usr/bin/env python3
"""
PNBTR Office Mode Training Loop - Phase 2 Enhanced
"No Rest Until Mastery" - keeps training until ≥90% accuracy achieved

Phase 2: Now integrates real PyTorch training alongside original dummy training.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

# Core training components
from .waveform_utils import load_audio, align_signals, generate_test_signal, save_audio
from .model_factory import create_pnbtr_model, get_model_info
from ..metrics.scoring import score_accuracy, meets_mastery_threshold, print_score_summary
from ..config.thresholds import load_thresholds

# Try to import PyTorch trainer for Phase 2
try:
    from .pytorch_trainer import enhanced_train_single_sample
    PYTORCH_TRAINING_AVAILABLE = True
except ImportError:
    PYTORCH_TRAINING_AVAILABLE = False

# Training configuration
ACCURACY_THRESHOLD = 0.90  # 90% composite accuracy required
MAX_ATTEMPTS = 1000        # Maximum training iterations per sample
SESSION_LOG_DIR = Path("../logs/sessions")
GUIDANCE_DIR = Path("../guidance")

class PNBTRTrainingLoop:
    """
    Enhanced PNBTR training loop supporting both PyTorch and dummy models.
    Implements the brutal perfectionist that refuses to settle for mediocrity.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize training loop with configuration.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config = self._load_config(config_path)
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Create log directory
        self.log_dir = Path(__file__).parent.parent / "logs" / "sessions" / self.session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training statistics
        self.stats = {
            "samples_trained": 0,
            "samples_mastered": 0,
            "total_epochs": 0,
            "total_training_time": 0,
            "pytorch_models": 0,
            "dummy_models": 0
        }
        
        print(f"🏁 PNBTR Training Session: {self.session_id}")
        print(f"📁 Logs: {self.log_dir}")
        print(f"🚀 PyTorch training: {'✅ Available' if PYTORCH_TRAINING_AVAILABLE else '❌ Not available'}")
    
    def _load_config(self, config_path):
        """Load training configuration"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "training_params.yaml"
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️  Could not load config: {e}, using defaults")
            return {}
    
    def train_single_sample(self, input_signal, target_signal, model=None):
        """
        Train on a single input-target pair until mastery achieved.
        Enhanced to use PyTorch training when available.
        
        Args:
            input_signal: Degraded/incomplete audio array  
            target_signal: Ground truth reconstruction target
            model: Model to train (created if None)
            
        Returns:
            dict: Training results including accuracy and metrics
        """
        start_time = time.time()
        
        # Create model if not provided
        if model is None:
            model = create_pnbtr_model(
                config=self.config,
                input_size=len(input_signal),
                use_pytorch=True  # Prefer PyTorch in Phase 2
            )
        
        # Get model info for logging
        model_info = get_model_info(model)
        is_pytorch = model_info.get('is_pytorch', False)
        
        print(f"\n🎯 Training Sample ({len(input_signal)} samples)")
        print(f"   🏗️  Model: {model_info['name']}")
        print(f"   🔢 Parameters: {model_info['parameters']:,}")
        print(f"   🚀 Training Type: {'PyTorch' if is_pytorch else 'Dummy'}")
        
        # Align signals for accurate comparison
        aligned_input, aligned_target = align_signals(input_signal, target_signal)
        
        # Use enhanced training if PyTorch available
        if PYTORCH_TRAINING_AVAILABLE and is_pytorch:
            result = enhanced_train_single_sample(
                aligned_input, aligned_target, model, self.config
            )
            self.stats["pytorch_models"] += 1
        else:
            # Fallback to original training loop
            result = self._original_train_single_sample(
                aligned_input, aligned_target, model
            )
            self.stats["dummy_models"] += 1
        
        # Calculate training time
        training_time = time.time() - start_time
        result["training_time_seconds"] = training_time
        
        # Update statistics
        self.stats["samples_trained"] += 1
        self.stats["total_training_time"] += training_time
        if result.get("mastery_achieved", False):
            self.stats["samples_mastered"] += 1
        if "epochs_trained" in result:
            self.stats["total_epochs"] += result["epochs_trained"]
        
        # Log results
        self._log_training_result(result, model_info)
        
        # Print summary
        accuracy = result.get("accuracy", 0.0)
        mastery_icon = "🏆" if result.get("mastery_achieved", False) else "📈"
        print(f"   {mastery_icon} Final accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   ⏱️  Training time: {training_time:.1f}s")
        
        return result
    
    def _original_train_single_sample(self, input_signal, target_signal, model):
        """
        Original training loop for dummy models.
        Maintains compatibility with Phase 1 implementation.
        """
        from .loss_functions import evaluate_metrics
        
        print(f"🔄 Using original training loop...")
        
        attempt = 0
        best_accuracy = 0.0
        plateau_count = 0
        
        while attempt < MAX_ATTEMPTS:
            # Get model prediction
            prediction = model.predict(input_signal)
            
            # Evaluate reconstruction quality
            metrics = evaluate_metrics(prediction, target_signal, sample_rate=48000)
            accuracy = score_accuracy(metrics)
            
            # Check for improvement
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                plateau_count = 0
            else:
                plateau_count += 1
            
            # Calculate loss for model training step
            loss = 1.0 - accuracy  # Convert accuracy to loss
            
            # Training step
            model.step(loss)
            
            # Progress reporting
            if attempt % 50 == 0 or attempt < 10:
                print(f"   Attempt {attempt:4d}: Accuracy = {accuracy:.3f}")
            
            # Check mastery threshold
            if meets_mastery_threshold(accuracy):
                print(f"   🏆 MASTERY ACHIEVED after {attempt} attempts!")
                break
            
            # Early stopping for dummy models
            if plateau_count >= 100:  # More patience for dummy models
                print(f"   🛑 Plateau detected after {attempt} attempts")
                break
            
            attempt += 1
        
        return {
            "epochs_trained": attempt,
            "accuracy": best_accuracy,
            "mastery_achieved": meets_mastery_threshold(best_accuracy),
            "final_loss": 1.0 - best_accuracy,
            "early_stopped": plateau_count >= 100
        }
    
    def train_dataset(self, input_dir, target_dir, model_type=None):
        """
        Train on a complete dataset directory.
        
        Args:
            input_dir: Directory containing input audio files
            target_dir: Directory containing target audio files  
            model_type: Type of model to use ("mlp", "conv1d", etc.)
            
        Returns:
            dict: Overall training statistics
        """
        input_path = Path(input_dir)
        target_path = Path(target_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target directory not found: {target_path}")
        
        # Find matching audio files
        input_files = sorted(input_path.glob("*.wav"))
        target_files = sorted(target_path.glob("*.wav"))
        
        if len(input_files) == 0:
            raise ValueError(f"No WAV files found in {input_path}")
        
        print(f"📂 Dataset Training")
        print(f"   📁 Input: {input_path} ({len(input_files)} files)")
        print(f"   📁 Target: {target_path} ({len(target_files)} files)")
        
        # Create single model for entire dataset
        if len(input_files) > 0:
            # Load first file to get input size
            first_input, _, _ = load_audio(input_files[0])
            model = create_pnbtr_model(
                model_type=model_type,
                config=self.config,
                input_size=len(first_input),
                use_pytorch=True
            )
        
        results = []
        
        # Train on each file pair
        for i, input_file in enumerate(input_files):
            # Find matching target file
            target_file = target_path / input_file.name
            if not target_file.exists():
                print(f"⚠️  No matching target for {input_file.name}, skipping")
                continue
            
            print(f"\n📄 File {i+1}/{len(input_files)}: {input_file.name}")
            
            try:
                # Load audio files
                input_signal, input_sr, _ = load_audio(input_file)
                target_signal, target_sr, _ = load_audio(target_file)
                
                # Validate sample rates match
                if input_sr != target_sr:
                    print(f"⚠️  Sample rate mismatch: {input_sr} vs {target_sr}")
                
                # Train on this pair
                result = self.train_single_sample(input_signal, target_signal, model)
                result["filename"] = input_file.name
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {input_file.name}: {e}")
                continue
        
        # Generate summary
        summary = self._generate_training_summary(results)
        self._save_session_summary(summary)
        
        return summary
    
    def _log_training_result(self, result, model_info):
        """Log training result to session file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_info,
            "result": result
        }
        
        log_file = self.log_dir / "training_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _generate_training_summary(self, results):
        """Generate summary statistics from training results"""
        if not results:
            return {"error": "No successful training results"}
        
        accuracies = [r["accuracy"] for r in results]
        mastery_count = sum(1 for r in results if r.get("mastery_achieved", False))
        
        summary = {
            "session_id": self.session_id,
            "session_duration": str(datetime.now() - self.session_start),
            "files_processed": len(results),
            "mastery_achieved": mastery_count,
            "mastery_rate": mastery_count / len(results),
            "accuracy_stats": {
                "mean": float(np.mean(accuracies)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
                "std": float(np.std(accuracies))
            },
            "training_stats": self.stats,
            "individual_results": results
        }
        
        return summary
    
    def _save_session_summary(self, summary):
        """Save session summary to JSON file"""
        summary_file = self.log_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Session Summary:")
        print(f"   📄 Files processed: {summary['files_processed']}")
        print(f"   🏆 Mastery achieved: {summary['mastery_achieved']}/{summary['files_processed']}")
        print(f"   📈 Mastery rate: {summary['mastery_rate']:.1%}")
        print(f"   🎯 Mean accuracy: {summary['accuracy_stats']['mean']:.3f}")
        print(f"   📁 Summary saved: {summary_file}")

def quick_test():
    """Quick test function for development"""
    print("🧪 PNBTR Training Loop Quick Test")
    
    # Generate test signals
    input_signal = generate_test_signal(1000, 48000, "complex")
    # Add slight degradation to create training target
    target_signal = input_signal + np.random.normal(0, 0.02, len(input_signal))
    
    # Create training loop
    trainer = PNBTRTrainingLoop()
    
    # Test single sample training
    result = trainer.train_single_sample(input_signal, target_signal)
    
    print(f"\n✅ Quick test completed:")
    print(f"   Accuracy: {result['accuracy']:.3f}")
    print(f"   Mastery: {'Yes' if result.get('mastery_achieved', False) else 'No'}")
    
    return result

# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    print("🚀 PNBTR Training Loop - Phase 2")
    print("=" * 50)
    
    # Quick test
    quick_test() 