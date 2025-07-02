#!/usr/bin/env python3
"""
Setup and Quick Start Script for Chord Progression Generator
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run a shell command with progress feedback"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['torch', 'transformers', 'numpy', 'sklearn', 'matplotlib', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies from requirements.txt...")
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def download_models():
    """Download required models"""
    print("🤖 Downloading BERT model...")
    try:
        from transformers import BertTokenizer, BertModel
        print("📥 Downloading BERT tokenizer...")
        BertTokenizer.from_pretrained('bert-base-uncased')
        print("📥 Downloading BERT model...")
        BertModel.from_pretrained('bert-base-uncased')
        print("✅ BERT model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download BERT model: {e}")
        return False

def verify_database():
    """Verify the emotion progression database is valid"""
    print("📊 Verifying emotion progression database...")
    
    try:
        with open('emotion_progression_database.json', 'r') as f:
            data = json.load(f)
        
        # Check structure
        assert 'emotions' in data
        assert len(data['emotions']) == 12
        
        for emotion_name, emotion_data in data['emotions'].items():
            assert 'progression_pool' in emotion_data
            assert len(emotion_data['progression_pool']) == 12
        
        print("✅ Database structure is valid")
        return True
    
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

def test_basic_functionality():
    """Test basic model functionality"""
    print("🧪 Testing basic functionality...")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        print("  Creating model instance...")
        model = ChordProgressionModel()
        
        print("  Testing emotion parsing...")
        emotion_weights = model.emotion_parser.parse_emotion_weights("happy and excited")
        assert isinstance(emotion_weights, dict)
        assert len(emotion_weights) == 12
        
        print("  Testing mode blending...")
        primary_mode, mode_blend = model.mode_blender.get_primary_mode(emotion_weights)
        assert isinstance(primary_mode, str)
        assert isinstance(mode_blend, dict)
        
        print("  Testing progression generation...")
        results = model.generate_from_prompt("happy and excited", "Pop", 1)
        assert len(results) == 1
        assert 'chords' in results[0]
        
        print("✅ Basic functionality test passed")
        return True
    
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def create_example_output():
    """Create example outputs to verify everything works"""
    print("🎵 Creating example outputs...")
    
    try:
        from chord_progression_model import ChordProgressionModel
        
        model = ChordProgressionModel()
        
        # Test prompts
        test_prompts = [
            "happy and excited",
            "sad but beautiful",
            "dark and mysterious",
            "romantic and warm"
        ]
        
        print("📝 Example generations:")
        for prompt in test_prompts:
            results = model.generate_from_prompt(prompt, "Pop", 1)
            result = results[0]
            progression = ' → '.join(result['chords'])
            top_emotion = max(result['emotion_weights'], key=result['emotion_weights'].get)
            
            print(f"  '{prompt}' → {top_emotion} → {progression}")
        
        # Export a sample session
        session = model.export_jam_session(results, "setup_test_session")
        with open('example_session.jam', 'w') as f:
            json.dump(session, f, indent=2)
        
        print("✅ Example outputs created successfully")
        print("📄 Sample session saved as 'example_session.jam'")
        return True
    
    except Exception as e:
        print(f"❌ Example output creation failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎼 Chord Progression Generator - Setup & Quick Start")
    print("=" * 60)
    
    # Check current directory
    if not Path('chord_progression_model.py').exists():
        print("❌ Setup must be run from the project directory")
        print("   Make sure you're in the directory containing chord_progression_model.py")
        sys.exit(1)
    
    success_count = 0
    total_steps = 6
    
    # Step 1: Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"📦 Missing packages: {', '.join(missing_deps)}")
        if input("Install missing dependencies? (y/n): ").lower() == 'y':
            if install_dependencies():
                success_count += 1
        else:
            print("⚠️  Skipping dependency installation")
    else:
        success_count += 1
    
    # Step 2: Download models
    if download_models():
        success_count += 1
    
    # Step 3: Verify database
    if verify_database():
        success_count += 1
    
    # Step 4: Test functionality
    if test_basic_functionality():
        success_count += 1
    
    # Step 5: Create examples
    if create_example_output():
        success_count += 1
    
    # Step 6: Setup complete
    success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Setup Progress: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 Setup completed successfully!")
        print("\n🚀 Quick Start Commands:")
        print("  python demo.py                    # Interactive demo")
        print("  python train_model.py             # Train the model")
        print("  python midi_generator.py          # Generate MIDI files")
        print("\n📖 For more information, see README.md")
    else:
        print("⚠️  Setup completed with some issues")
        print("   Check the error messages above and try again")
        print("   You may need to install dependencies manually")
    
    print("\n📋 Next Steps:")
    print("1. Run 'python demo.py' for an interactive demonstration")
    print("2. Try the example code in README.md")
    print("3. Train your own model with 'python train_model.py'")
    print("4. Generate MIDI files from your chord progressions")

if __name__ == "__main__":
    main()
