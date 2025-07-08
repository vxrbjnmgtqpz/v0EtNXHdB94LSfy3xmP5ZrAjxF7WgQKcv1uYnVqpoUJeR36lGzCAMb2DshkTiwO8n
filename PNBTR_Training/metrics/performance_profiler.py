#!/usr/bin/env python3
"""
PNBTR Performance Profiler - Phase 3
Comprehensive performance analysis for training optimization and deployment readiness.
Memory usage, speed benchmarks, model profiling, and real-time constraints validation.
"""

import time
import gc
import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass
from contextlib import contextmanager

# Optional profiling dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    execution_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    model_size_mb: Optional[float] = None
    parameters_count: Optional[int] = None
    inference_time_ms: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None

class PerformanceProfiler:
    """
    Comprehensive performance profiling system for PNBTR training and inference.
    """
    
    def __init__(self):
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.baseline_memory = self._get_memory_usage()
        
        # Performance history
        self.training_profiles = []
        self.inference_profiles = []
        self.memory_timeline = []
        
        # Real-time constraints
        self.realtime_threshold_ms = 1.0  # 1ms latency requirement
        self.memory_limit_mb = 1024      # 1GB memory limit
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            return self.process.memory_info().rss / 1024 / 1024
        else:
            # Fallback estimation
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
    
    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB"""
        if PSUTIL_AVAILABLE:
            return self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else self._get_memory_usage()
        else:
            return self._get_memory_usage()
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage"""
        if PSUTIL_AVAILABLE:
            return self.process.cpu_percent()
        else:
            return 0.0
    
    @contextmanager
    def profile_execution(self, operation_name: str = "operation"):
        """
        Context manager for profiling code execution.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Yields:
            PerformanceMetrics object that gets populated during execution
        """
        # Initial measurements
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        # Force garbage collection for cleaner measurement
        gc.collect()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            execution_time_ms=0.0,
            memory_usage_mb=start_memory,
            peak_memory_mb=start_memory,
            cpu_percent=0.0
        )
        
        try:
            yield metrics
        finally:
            # Final measurements
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            peak_memory = self._get_peak_memory()
            cpu_usage = self._get_cpu_percent()
            
            # Update metrics
            metrics.execution_time_ms = (end_time - start_time) * 1000
            metrics.memory_usage_mb = end_memory - start_memory
            metrics.peak_memory_mb = peak_memory
            metrics.cpu_percent = cpu_usage
            
            # Log to timeline
            self.memory_timeline.append({
                'timestamp': time.time(),
                'operation': operation_name,
                'memory_mb': end_memory,
                'memory_delta_mb': end_memory - start_memory
            })
    
    def profile_model_size(self, model) -> Dict[str, Any]:
        """
        Analyze model size and parameter count.
        
        Args:
            model: Model to analyze (PyTorch or custom)
            
        Returns:
            Dictionary with model size metrics
        """
        results = {
            'parameters_count': 0,
            'model_size_mb': 0.0,
            'memory_footprint_mb': 0.0,
            'is_pytorch': False
        }
        
        # PyTorch model analysis
        if TORCH_AVAILABLE and hasattr(model, 'parameters'):
            results['is_pytorch'] = True
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results['parameters_count'] = total_params
            results['trainable_parameters'] = trainable_params
            
            # Estimate model size (4 bytes per float32 parameter)
            model_size_bytes = total_params * 4
            results['model_size_mb'] = model_size_bytes / 1024 / 1024
            
            # Memory footprint (includes gradients, optimizer states, etc.)
            # Rough estimate: 3x model size (weights + gradients + optimizer)
            results['memory_footprint_mb'] = results['model_size_mb'] * 3
            
            # Layer-by-layer analysis
            layer_info = []
            for name, param in model.named_parameters():
                layer_info.append({
                    'name': name,
                    'shape': list(param.shape),
                    'parameters': param.numel(),
                    'size_mb': param.numel() * 4 / 1024 / 1024,
                    'requires_grad': param.requires_grad
                })
            
            results['layer_details'] = layer_info
            
        else:
            # Custom model analysis
            if hasattr(model, 'weights') and isinstance(model.weights, list):
                # Dummy model with weights list
                total_params = 0
                for weight, bias in model.weights:
                    total_params += weight.size + bias.size
                
                results['parameters_count'] = total_params
                results['model_size_mb'] = total_params * 8 / 1024 / 1024  # 8 bytes for float64
        
        return results
    
    def profile_training_step(self, model, input_data, target_data, 
                            loss_function=None, optimizer=None) -> PerformanceMetrics:
        """
        Profile a single training step.
        
        Args:
            model: Model being trained
            input_data: Training input
            target_data: Training target
            loss_function: Loss function (optional)
            optimizer: Optimizer (optional)
            
        Returns:
            PerformanceMetrics for the training step
        """
        with self.profile_execution("training_step") as metrics:
            try:
                # Forward pass
                prediction = model.predict(input_data)
                
                # Loss calculation (if available)
                if loss_function and TORCH_AVAILABLE:
                    if hasattr(model, 'state_dict'):  # PyTorch model
                        # Convert to tensors
                        pred_tensor = torch.from_numpy(prediction.astype('float32'))
                        target_tensor = torch.from_numpy(target_data.astype('float32'))
                        loss = loss_function(pred_tensor, target_tensor)
                        
                        # Backward pass
                        if optimizer:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                
                # Training step for dummy models
                elif hasattr(model, 'step'):
                    # Calculate simple loss
                    mse_loss = ((prediction - target_data) ** 2).mean()
                    model.step(mse_loss)
                
                # Model size analysis
                model_metrics = self.profile_model_size(model)
                metrics.model_size_mb = model_metrics['model_size_mb']
                metrics.parameters_count = model_metrics['parameters_count']
                
            except Exception as e:
                print(f"⚠️  Training step profiling failed: {e}")
        
        # Store in history
        self.training_profiles.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        return metrics
    
    def profile_inference(self, model, input_data, 
                         num_runs: int = 100) -> Dict[str, float]:
        """
        Profile model inference performance and real-time compatibility.
        
        Args:
            model: Model to profile
            input_data: Input data for inference
            num_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary with inference performance metrics
        """
        inference_times = []
        memory_usage = []
        
        # Warm-up runs
        for _ in range(5):
            _ = model.predict(input_data)
        
        # Timed runs
        for run in range(num_runs):
            with self.profile_execution(f"inference_run_{run}") as metrics:
                prediction = model.predict(input_data)
                inference_times.append(metrics.execution_time_ms)
                memory_usage.append(metrics.memory_usage_mb)
        
        # Calculate statistics
        results = {
            'mean_inference_time_ms': float(sum(inference_times) / len(inference_times)),
            'min_inference_time_ms': float(min(inference_times)),
            'max_inference_time_ms': float(max(inference_times)),
            'std_inference_time_ms': float((sum((t - sum(inference_times)/len(inference_times))**2 for t in inference_times) / len(inference_times))**0.5),
            'mean_memory_usage_mb': float(sum(memory_usage) / len(memory_usage)),
            'samples_per_second': len(input_data) / (sum(inference_times) / len(inference_times) / 1000),
            'realtime_compatible': float(sum(inference_times) / len(inference_times)) < self.realtime_threshold_ms,
            'memory_efficient': float(sum(memory_usage) / len(memory_usage)) < self.memory_limit_mb
        }
        
        # Store in history
        self.inference_profiles.append({
            'timestamp': time.time(),
            'input_size': len(input_data),
            'num_runs': num_runs,
            'results': results
        })
        
        return results
    
    def benchmark_model_architectures(self, architectures: List[str], 
                                    input_size: int = 1024,
                                    config: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Benchmark different model architectures for performance comparison.
        
        Args:
            architectures: List of model architecture names
            input_size: Size of input for testing
            config: Configuration for model creation
            
        Returns:
            Dictionary with benchmark results for each architecture
        """
        from ..training.model_factory import create_pnbtr_model
        import numpy as np
        
        results = {}
        test_input = np.random.normal(0, 0.1, input_size).astype(np.float64)
        test_target = test_input + 0.01 * np.random.normal(0, 1, input_size)
        
        for arch in architectures:
            print(f"📊 Benchmarking {arch} architecture...")
            
            try:
                # Create model
                model = create_pnbtr_model(arch, config=config, input_size=input_size)
                
                # Model size analysis
                size_metrics = self.profile_model_size(model)
                
                # Training step performance
                training_metrics = self.profile_training_step(model, test_input, test_target)
                
                # Inference performance
                inference_metrics = self.profile_inference(model, test_input, num_runs=50)
                
                results[arch] = {
                    'model_size': size_metrics,
                    'training_performance': {
                        'time_per_step_ms': training_metrics.execution_time_ms,
                        'memory_usage_mb': training_metrics.memory_usage_mb,
                        'peak_memory_mb': training_metrics.peak_memory_mb
                    },
                    'inference_performance': inference_metrics,
                    'realtime_score': self._calculate_realtime_score(training_metrics, inference_metrics, size_metrics)
                }
                
            except Exception as e:
                print(f"❌ Benchmark failed for {arch}: {e}")
                results[arch] = {'error': str(e)}
        
        return results
    
    def _calculate_realtime_score(self, training_metrics: PerformanceMetrics,
                                inference_metrics: Dict[str, float],
                                size_metrics: Dict[str, Any]) -> float:
        """
        Calculate a composite real-time readiness score (0-1).
        
        Higher scores indicate better real-time performance.
        """
        # Inference time score (1.0 if under threshold, decreases linearly)
        inference_score = max(0, 1.0 - inference_metrics['mean_inference_time_ms'] / self.realtime_threshold_ms)
        
        # Memory score (1.0 if under limit, decreases linearly)
        memory_score = max(0, 1.0 - size_metrics['memory_footprint_mb'] / self.memory_limit_mb)
        
        # Model size score (smaller is better, normalized to reasonable range)
        size_score = max(0, 1.0 - size_metrics['model_size_mb'] / 100)  # 100MB as reference
        
        # Consistency score (lower variance is better)
        consistency_score = max(0, 1.0 - inference_metrics['std_inference_time_ms'] / inference_metrics['mean_inference_time_ms'])
        
        # Weighted composite score
        composite_score = (
            0.4 * inference_score +     # Inference speed is most important
            0.3 * memory_score +        # Memory usage is critical
            0.2 * consistency_score +   # Consistency matters for real-time
            0.1 * size_score           # Model size affects deployment
        )
        
        return float(min(1.0, max(0.0, composite_score)))
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report"""
        report = {
            'session_info': {
                'timestamp': time.time(),
                'baseline_memory_mb': self.baseline_memory,
                'current_memory_mb': self._get_memory_usage(),
                'memory_increase_mb': self._get_memory_usage() - self.baseline_memory,
                'psutil_available': PSUTIL_AVAILABLE,
                'torch_available': TORCH_AVAILABLE
            },
            'training_profiles': self.training_profiles,
            'inference_profiles': self.inference_profiles,
            'memory_timeline': self.memory_timeline,
            'constraints': {
                'realtime_threshold_ms': self.realtime_threshold_ms,
                'memory_limit_mb': self.memory_limit_mb
            }
        }
        
        # Calculate summary statistics
        if self.training_profiles:
            training_times = [p['metrics'].execution_time_ms for p in self.training_profiles]
            report['training_summary'] = {
                'mean_time_per_step_ms': sum(training_times) / len(training_times),
                'min_time_ms': min(training_times),
                'max_time_ms': max(training_times),
                'total_steps': len(training_times)
            }
        
        if self.inference_profiles:
            inference_data = [p['results'] for p in self.inference_profiles]
            if inference_data:
                report['inference_summary'] = {
                    'mean_inference_time_ms': sum(d['mean_inference_time_ms'] for d in inference_data) / len(inference_data),
                    'best_throughput_sps': max(d['samples_per_second'] for d in inference_data),
                    'realtime_compatible_ratio': sum(1 for d in inference_data if d['realtime_compatible']) / len(inference_data)
                }
        
        return report
    
    def save_performance_report(self, filepath: Path):
        """Save performance report to JSON file"""
        report = self.generate_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance report saved: {filepath}")
    
    def print_performance_summary(self):
        """Print a formatted performance summary"""
        report = self.generate_performance_report()
        
        print("\n" + "="*60)
        print("🔍 PNBTR PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        # Session info
        session = report['session_info']
        print(f"📊 Memory Usage:")
        print(f"   Baseline: {session['baseline_memory_mb']:.1f} MB")
        print(f"   Current: {session['current_memory_mb']:.1f} MB")
        print(f"   Increase: {session['memory_increase_mb']:.1f} MB")
        
        # Training performance
        if 'training_summary' in report:
            training = report['training_summary']
            print(f"\n🏃 Training Performance:")
            print(f"   Mean time per step: {training['mean_time_per_step_ms']:.2f} ms")
            print(f"   Range: {training['min_time_ms']:.2f} - {training['max_time_ms']:.2f} ms")
            print(f"   Total steps profiled: {training['total_steps']}")
        
        # Inference performance
        if 'inference_summary' in report:
            inference = report['inference_summary']
            print(f"\n⚡ Inference Performance:")
            print(f"   Mean inference time: {inference['mean_inference_time_ms']:.2f} ms")
            print(f"   Best throughput: {inference['best_throughput_sps']:.0f} samples/sec")
            print(f"   Real-time compatible: {inference['realtime_compatible_ratio']:.1%}")
        
        # Real-time constraints
        constraints = report['constraints']
        print(f"\n🎯 Real-Time Constraints:")
        print(f"   Latency threshold: {constraints['realtime_threshold_ms']:.1f} ms")
        print(f"   Memory limit: {constraints['memory_limit_mb']} MB")
        
        print("="*60)

def create_performance_profiler() -> PerformanceProfiler:
    """Factory function to create performance profiler"""
    return PerformanceProfiler()

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Performance Profiler Test")
    
    # Create profiler
    profiler = create_performance_profiler()
    
    # Test with simple operations
    import numpy as np
    
    # Profile some operations
    print("\n🔍 Profiling basic operations...")
    
    with profiler.profile_execution("numpy_operations") as metrics:
        # Simulate some computation
        data = np.random.random((10000, 100))
        result = np.dot(data, data.T)
        eigenvals = np.linalg.eigvals(result[:100, :100])
    
    print(f"Numpy operation: {metrics.execution_time_ms:.2f} ms, {metrics.memory_usage_mb:.2f} MB")
    
    # Test model profiling if possible
    try:
        from training.model_factory import create_pnbtr_model
        
        print("\n🏗️  Profiling PNBTR models...")
        
        # Create test input
        test_input = np.random.normal(0, 0.1, 1000).astype(np.float64)
        test_target = test_input + 0.01 * np.random.normal(0, 1, 1000)
        
        # Profile different architectures
        architectures = ["dummy", "mlp"]  # Start with dummy and mlp
        
        benchmark_results = profiler.benchmark_model_architectures(
            architectures, input_size=1000
        )
        
        for arch, results in benchmark_results.items():
            if 'error' not in results:
                print(f"\n📐 {arch.upper()} Architecture:")
                print(f"   Model size: {results['model_size']['model_size_mb']:.2f} MB")
                print(f"   Parameters: {results['model_size']['parameters_count']:,}")
                print(f"   Training time: {results['training_performance']['time_per_step_ms']:.2f} ms/step")
                print(f"   Inference time: {results['inference_performance']['mean_inference_time_ms']:.2f} ms")
                print(f"   Real-time score: {results['realtime_score']:.3f}")
                print(f"   Real-time compatible: {'✅' if results['inference_performance']['realtime_compatible'] else '❌'}")
            else:
                print(f"\n❌ {arch}: {results['error']}")
    
    except Exception as e:
        print(f"⚠️  Model profiling test failed: {e}")
    
    # Generate and print summary
    profiler.print_performance_summary()
    
    # Save report
    report_path = Path("performance_report.json")
    profiler.save_performance_report(report_path)
    
    print(f"\nPsutil available: {PSUTIL_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print("✅ Performance profiler test complete") 