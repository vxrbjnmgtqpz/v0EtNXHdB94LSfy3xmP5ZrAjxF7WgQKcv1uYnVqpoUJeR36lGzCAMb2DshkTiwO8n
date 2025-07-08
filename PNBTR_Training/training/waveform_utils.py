#!/usr/bin/env python3
"""
PNBTR Waveform Utilities
Audio loading, signal processing, and reconstruction utilities that respect
the anti-float, anti-dither philosophy: native 192kHz, 24-bit precision.
"""

import numpy as np
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def load_audio(path, force_sample_rate=None):
    """
    Load audio file at native sample rate and bit depth.
    Philosophy: Preserve original resolution, no automatic resampling.
    
    Args:
        path: Path to audio file
        force_sample_rate: Optional override (use with caution)
        
    Returns:
        tuple: (audio_data, sample_rate, metadata)
        audio_data: numpy array, float64 for maximum precision
        sample_rate: Native sample rate of the file
        metadata: Dict with bit depth, channels, etc.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Try librosa first (best for analysis)
    try:
        audio_data, sample_rate = _load_with_librosa(path, force_sample_rate)
        metadata = {"loader": "librosa", "precision": "float64"}
        
    except Exception as librosa_error:
        print(f"⚠️  librosa failed: {librosa_error}")
        
        # Fallback to scipy for raw WAV loading
        try:
            audio_data, sample_rate = _load_with_scipy(path)
            metadata = {"loader": "scipy", "precision": "native"}
            
        except Exception as scipy_error:
            print(f"❌ All loaders failed. librosa: {librosa_error}, scipy: {scipy_error}")
            raise RuntimeError(f"Could not load {path}")
    
    # Ensure consistent data type (float64 for maximum precision)
    if audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float64)
    
    # Handle stereo/mono conversion if needed
    if audio_data.ndim > 1:
        # JELLIE uses dual-channel, preserve stereo structure
        if audio_data.shape[1] == 2:
            metadata["channels"] = "stereo"
        else:
            # Sum to mono if more than 2 channels
            audio_data = np.mean(audio_data, axis=1)
            metadata["channels"] = "mono_converted"
    else:
        metadata["channels"] = "mono"
    
    # Validate sample rate expectations
    if sample_rate not in [44100, 48000, 96000, 192000]:
        print(f"⚠️  Unusual sample rate: {sample_rate}Hz")
    
    metadata.update({
        "sample_rate": sample_rate,
        "length_samples": len(audio_data),
        "duration_ms": len(audio_data) / sample_rate * 1000,
        "peak_amplitude": np.max(np.abs(audio_data)),
        "rms_level": np.sqrt(np.mean(audio_data ** 2))
    })
    
    print(f"📁 Loaded: {path.name} ({sample_rate}Hz, {metadata['channels']}, "
          f"{metadata['duration_ms']:.1f}ms)")
    
    return audio_data, sample_rate, metadata

def _load_with_librosa(path, force_sample_rate=None):
    """Load audio using librosa (preferred method)"""
    try:
        import librosa
        
        # Load at native sample rate (sr=None) unless forced
        sr = force_sample_rate
        
        # Load with maximum precision
        audio_data, sample_rate = librosa.load(
            str(path), 
            sr=sr,           # Native rate unless overridden
            mono=False,      # Preserve stereo
            dtype=np.float64 # Maximum precision
        )
        
        return audio_data, sample_rate
        
    except ImportError:
        raise RuntimeError("librosa not available - install with: pip install librosa")

def _load_with_scipy(path):
    """Fallback audio loading using scipy (WAV files only)"""
    try:
        from scipy.io import wavfile
        
        sample_rate, audio_data = wavfile.read(str(path))
        
        # Convert to float64 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float64) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float64) / 2147483648.0
        elif audio_data.dtype == np.int8:
            audio_data = audio_data.astype(np.float64) / 128.0
        else:
            audio_data = audio_data.astype(np.float64)
        
        return audio_data, sample_rate
        
    except ImportError:
        raise RuntimeError("scipy not available")

def align_signals(signal1, signal2, max_offset=None):
    """
    Align two signals for accurate comparison.
    Essential for training - ground truth must be sample-perfect aligned.
    
    Args:
        signal1, signal2: Audio arrays to align
        max_offset: Maximum samples to search for alignment (default: 10% of length)
        
    Returns:
        tuple: (aligned_signal1, aligned_signal2)
    """
    if max_offset is None:
        max_offset = min(len(signal1), len(signal2)) // 10
    
    # Cross-correlation to find optimal alignment
    from scipy import signal as scipy_signal
    
    # Truncate to same length for correlation
    min_len = min(len(signal1), len(signal2))
    s1_trunc = signal1[:min_len]
    s2_trunc = signal2[:min_len]
    
    # Calculate cross-correlation
    correlation = scipy_signal.correlate(s1_trunc, s2_trunc, mode='full')
    
    # Find peak correlation
    peak_idx = np.argmax(np.abs(correlation))
    offset = peak_idx - (len(s1_trunc) - 1)
    
    # Limit offset to reasonable range
    offset = np.clip(offset, -max_offset, max_offset)
    
    print(f"🔄 Signal alignment offset: {offset} samples")
    
    # Apply alignment
    if offset > 0:
        # signal2 leads, trim its beginning
        aligned_s1 = signal1
        aligned_s2 = signal2[offset:]
    elif offset < 0:
        # signal1 leads, trim its beginning  
        aligned_s1 = signal1[-offset:]
        aligned_s2 = signal2
    else:
        # Already aligned
        aligned_s1 = signal1
        aligned_s2 = signal2
    
    # Ensure same final length
    final_len = min(len(aligned_s1), len(aligned_s2))
    aligned_s1 = aligned_s1[:final_len]
    aligned_s2 = aligned_s2[:final_len]
    
    return aligned_s1, aligned_s2

def reconstruct_signal(input_signal, model):
    """
    Use PNBTR model to reconstruct/enhance input signal.
    This is the core prediction function that the training loop calls.
    
    Args:
        input_signal: Degraded/incomplete audio array
        model: PNBTR model instance
        
    Returns:
        numpy.ndarray: Reconstructed signal
    """
    # Ensure input is proper format
    if input_signal.dtype != np.float64:
        input_signal = input_signal.astype(np.float64)
    
    try:
        # Model-specific reconstruction
        if hasattr(model, 'predict'):
            # Standard ML model interface
            prediction = model.predict(input_signal)
        elif hasattr(model, 'forward'):
            # PyTorch-style model
            import torch
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_signal).float()
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
                output_tensor = model.forward(input_tensor)
                prediction = output_tensor.squeeze().numpy()
        elif hasattr(model, '__call__'):
            # Callable model
            prediction = model(input_signal)
        else:
            # Fallback: model might be a simple function
            prediction = model(input_signal)
        
        # Ensure output is proper format
        if prediction.dtype != np.float64:
            prediction = prediction.astype(np.float64)
        
        # Ensure same length as input (truncate or pad as needed)
        if len(prediction) != len(input_signal):
            if len(prediction) > len(input_signal):
                prediction = prediction[:len(input_signal)]
            else:
                # Pad with zeros if needed
                padding = np.zeros(len(input_signal) - len(prediction))
                prediction = np.concatenate([prediction, padding])
        
        return prediction
        
    except Exception as e:
        print(f"❌ Model reconstruction failed: {e}")
        # Fallback: return input unchanged (0% improvement)
        return input_signal.copy()

def save_audio(audio_data, path, sample_rate=192000, bit_depth=24):
    """
    Save audio with specified quality settings.
    Maintains PNBTR philosophy: 24-bit, no dither, native sample rates.
    
    Args:
        audio_data: Audio array to save
        path: Output file path
        sample_rate: Sample rate (default 192kHz)
        bit_depth: Bit depth (default 24-bit)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure proper data type and range
    audio_data = np.asarray(audio_data, dtype=np.float64)
    
    # Clip to valid range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    try:
        # Try soundfile first (best quality control)
        import soundfile as sf
        
        sf.write(
            str(path),
            audio_data,
            sample_rate,
            subtype=f'PCM_{bit_depth}'  # PCM_24 for 24-bit
        )
        print(f"💾 Saved: {path} ({sample_rate}Hz, {bit_depth}-bit PCM)")
        
    except ImportError:
        # Fallback to scipy (WAV only, limited bit depths)
        try:
            from scipy.io import wavfile
            
            if bit_depth == 16:
                audio_int = (audio_data * 32767).astype(np.int16)
            elif bit_depth == 24:
                # 24-bit via 32-bit container
                audio_int = (audio_data * 8388607).astype(np.int32)
            else:
                audio_int = (audio_data * 2147483647).astype(np.int32)
            
            wavfile.write(str(path), sample_rate, audio_int)
            print(f"💾 Saved: {path} ({sample_rate}Hz, {bit_depth}-bit via scipy)")
            
        except ImportError:
            raise RuntimeError("No audio writing library available (soundfile or scipy)")

def generate_test_signal(duration_ms=1000, sample_rate=192000, signal_type="sweep"):
    """
    Generate test signals for PNBTR evaluation.
    
    Args:
        duration_ms: Signal duration in milliseconds
        sample_rate: Sample rate
        signal_type: "sweep", "sine", "noise", "impulse", "complex"
        
    Returns:
        numpy.ndarray: Test signal
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples)
    
    if signal_type == "sweep":
        # Logarithmic frequency sweep (20Hz to 20kHz)
        f_start, f_end = 20, 20000
        return np.sin(2 * np.pi * f_start * t * 
                     (f_end / f_start) ** (t / t[-1]))
    
    elif signal_type == "sine":
        # 1kHz test tone
        return np.sin(2 * np.pi * 1000 * t)
    
    elif signal_type == "noise":
        # White noise with proper scaling
        return np.random.normal(0, 0.1, num_samples)
    
    elif signal_type == "impulse":
        # Impulse response test
        signal = np.zeros(num_samples)
        signal[num_samples // 10] = 1.0  # Impulse at 10% through
        return signal
    
    elif signal_type == "complex":
        # Complex musical-like signal
        fundamental = 440  # A4
        harmonics = [1, 0.5, 0.3, 0.2, 0.1]  # Harmonic series
        
        signal = np.zeros(num_samples)
        for i, amplitude in enumerate(harmonics):
            frequency = fundamental * (i + 1)
            if frequency < sample_rate / 2:  # Avoid aliasing
                signal += amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add envelope
        envelope = np.exp(-3 * t)  # Exponential decay
        return signal * envelope
    
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

def analyze_signal_quality(audio_data, sample_rate):
    """
    Quick quality analysis of audio signal.
    Useful for validating loaded files.
    
    Returns:
        dict: Signal quality metrics
    """
    analysis = {
        "peak_amplitude": float(np.max(np.abs(audio_data))),
        "rms_level": float(np.sqrt(np.mean(audio_data ** 2))),
        "crest_factor": 0.0,
        "dc_offset": float(np.mean(audio_data)),
        "clipping": False,
        "silence": False,
        "dynamic_range_db": 0.0
    }
    
    # Crest factor (peak to RMS ratio)
    if analysis["rms_level"] > 1e-10:
        analysis["crest_factor"] = analysis["peak_amplitude"] / analysis["rms_level"]
    
    # Clipping detection
    analysis["clipping"] = analysis["peak_amplitude"] >= 0.99
    
    # Silence detection
    analysis["silence"] = analysis["peak_amplitude"] < 1e-6
    
    # Dynamic range estimation
    if not analysis["silence"]:
        # RMS in dB
        rms_db = 20 * np.log10(analysis["rms_level"] + 1e-15)
        peak_db = 20 * np.log10(analysis["peak_amplitude"] + 1e-15)
        analysis["dynamic_range_db"] = peak_db - rms_db
    
    return analysis

def validate_signal_pair(input_signal, target_signal):
    """
    Validate that input and target signals are suitable for training.
    
    Returns:
        dict: Validation results with warnings/errors
    """
    validation = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Length check
    if len(input_signal) != len(target_signal):
        validation["errors"].append(
            f"Length mismatch: input={len(input_signal)}, target={len(target_signal)}"
        )
        validation["valid"] = False
    
    # Quality checks
    input_quality = analyze_signal_quality(input_signal, 192000)
    target_quality = analyze_signal_quality(target_signal, 192000)
    
    # Check for silence
    if input_quality["silence"] or target_quality["silence"]:
        validation["warnings"].append("One or both signals are silent")
    
    # Check for clipping
    if input_quality["clipping"]:
        validation["warnings"].append("Input signal shows clipping")
    if target_quality["clipping"]:
        validation["warnings"].append("Target signal shows clipping")
    
    # Check for extreme DC offset
    if abs(input_quality["dc_offset"]) > 0.1:
        validation["warnings"].append(f"Input has DC offset: {input_quality['dc_offset']:.3f}")
    if abs(target_quality["dc_offset"]) > 0.1:
        validation["warnings"].append(f"Target has DC offset: {target_quality['dc_offset']:.3f}")
    
    # Check dynamic range
    if input_quality["dynamic_range_db"] < 10:
        validation["warnings"].append("Input has low dynamic range")
    if target_quality["dynamic_range_db"] < 10:
        validation["warnings"].append("Target has low dynamic range")
    
    return validation

# Utility for debugging and development

def print_signal_info(audio_data, sample_rate, label="Signal"):
    """Print comprehensive signal information"""
    quality = analyze_signal_quality(audio_data, sample_rate)
    
    print(f"\n📊 {label} Analysis:")
    print(f"   Length: {len(audio_data)} samples ({len(audio_data)/sample_rate*1000:.1f}ms)")
    print(f"   Sample Rate: {sample_rate}Hz") 
    print(f"   Peak: {quality['peak_amplitude']:.4f}")
    print(f"   RMS: {quality['rms_level']:.4f}")
    print(f"   Crest Factor: {quality['crest_factor']:.2f}")
    print(f"   DC Offset: {quality['dc_offset']:.6f}")
    print(f"   Dynamic Range: {quality['dynamic_range_db']:.1f}dB")
    
    if quality['clipping']:
        print("   ⚠️  CLIPPING DETECTED")
    if quality['silence']:
        print("   ⚠️  SILENCE DETECTED")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 PNBTR Waveform Utils Test")
    
    # Generate test signal
    test_signal = generate_test_signal(1000, 192000, "complex")
    print_signal_info(test_signal, 192000, "Generated Test Signal")
    
    # Save and reload test
    save_audio(test_signal, "test_output.wav", 192000, 24)
    loaded_signal, sr, metadata = load_audio("test_output.wav")
    print_signal_info(loaded_signal, sr, "Reloaded Signal")
    
    print("✅ Waveform utilities test complete") 