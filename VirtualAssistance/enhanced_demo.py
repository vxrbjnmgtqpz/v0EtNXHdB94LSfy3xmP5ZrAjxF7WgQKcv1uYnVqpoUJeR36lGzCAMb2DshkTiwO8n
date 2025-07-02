"""
Enhanced Demo for Fixed VirtualAssistance Model Stack
Demonstrates the complete integrated pipeline with all audit fixes applied
"""

import torch
import json
import os
from typing import Dict, List, Optional
import time

from chord_progression_model import ChordProgressionModel
from train_model import ModelTrainer
from wolfram_validator import WolframMusicValidator, integrate_wolfram_validation


class EnhancedMusicDemo:
    """
    Enhanced demo showcasing the complete fixed tri-model stack:
    1. Neural BERT emotion parsing (when trained)
    2. Neural mode blending (when trained) 
    3. Neural chord generation (when trained)
    4. Wolfram|Alpha music theory validation
    5. Secondary emotion verification
    """
    
    def __init__(self, model_path: Optional[str] = None, enable_wolfram: bool = True):
        print("🎵 Initializing Enhanced VirtualAssistance System...")
        print("=" * 60)
        
        # Initialize core model
        self.model = ChordProgressionModel()
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            print(f"📁 Loading trained model from {model_path}")
            trainer = ModelTrainer(self.model)
            trainer.load_model(model_path)
            self.model.enable_neural_generation()
            print("🧠 Neural generation enabled")
        else:
            print("⚠️ Using untrained model (database lookup mode)")
            self.model.disable_neural_generation()
        
        # Initialize Wolfram validator
        if enable_wolfram:
            self.wolfram_validator = WolframMusicValidator()
            if self.wolfram_validator.enabled:
                print("🧮 Wolfram|Alpha integration enabled")
            else:
                print("⚠️ Wolfram|Alpha disabled - set WOLFRAM_APP_ID to enable")
        else:
            self.wolfram_validator = None
            print("🔧 Wolfram integration disabled")
        
        print("✅ Enhanced system ready!\n")
    
    def generate_with_full_analysis(self, prompt: str, genre: str = "Pop", 
                                  num_progressions: int = 1) -> Dict:
        """
        Generate chord progressions with complete analysis pipeline
        """
        print(f"🎯 Analyzing: '{prompt}'")
        print(f"🎸 Genre: {genre}")
        print(f"🔢 Generating {num_progressions} progression(s)...")
        print("-" * 50)
        
        start_time = time.time()
        
        # 1. Generate progressions using the fixed model
        results = self.model.generate_from_prompt(prompt, genre, num_progressions)
        
        # 2. Add Wolfram validation for each result
        enhanced_results = []
        for result in results:
            if self.wolfram_validator and self.wolfram_validator.enabled:
                print(f"🧮 Running Wolfram validation...")
                enhanced_result = integrate_wolfram_validation(result, self.wolfram_validator)
            else:
                enhanced_result = result
                enhanced_result['wolfram_validation'] = {'status': 'disabled'}
            
            enhanced_results.append(enhanced_result)
        
        generation_time = time.time() - start_time
        
        # 3. Display comprehensive results
        self._display_analysis_results(enhanced_results, generation_time)
        
        return enhanced_results
    
    def _display_analysis_results(self, results: List[Dict], generation_time: float):
        """Display comprehensive analysis results"""
        
        for i, result in enumerate(results, 1):
            print(f"\n🎼 === Progression {i} Analysis ===")
            
            # Basic progression info
            print(f"🎹 Chord Progression: {' → '.join(result['chords'])}")
            print(f"🎵 Primary Mode: {result['primary_mode']}")
            print(f"🎸 Genre: {result['genre']}")
            print(f"🔧 Generation Method: {result['metadata']['generation_method']}")
            
            # Emotion analysis
            print(f"\n💭 Emotion Analysis:")
            top_emotions = sorted(result['emotion_weights'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for emotion, weight in top_emotions:
                if weight > 0.05:
                    bar = "█" * int(weight * 20)
                    print(f"   {emotion:12} {weight:.3f} |{bar}")
            
            # Mode blending
            if len([m for m, w in result['mode_blend'].items() if w > 0.1]) > 1:
                print(f"\n🌊 Mode Blend:")
                for mode, weight in result['mode_blend'].items():
                    if weight > 0.1:
                        bar = "▓" * int(weight * 15)
                        print(f"   {mode:15} {weight:.3f} |{bar}")
            
            # Wolfram validation results
            wolfram_data = result.get('wolfram_validation', {})
            if wolfram_data.get('progression_analysis', {}).get('validation_status') == 'success':
                print(f"\n🧮 Wolfram|Alpha Validation:")
                prog_analysis = wolfram_data['progression_analysis']
                mode_analysis = wolfram_data['mode_compatibility']
                
                print(f"   📊 Mode Compatibility: {mode_analysis.get('compatibility_score', 0):.2f}")
                print(f"   ✅ Fitting Chords: {mode_analysis.get('fitting_chords', [])}")
                
                non_fitting = mode_analysis.get('non_fitting_chords', [])
                if non_fitting:
                    print(f"   ⚠️ Non-fitting Chords: {non_fitting}")
                
                # Quality score if available
                if 'quality_score' in result:
                    quality = result['quality_score']
                    stars = "⭐" * int(quality * 5)
                    print(f"   🌟 Quality Score: {quality:.2f} {stars}")
            
            elif wolfram_data.get('status') == 'disabled':
                print(f"\n🧮 Wolfram Validation: Disabled (set WOLFRAM_APP_ID to enable)")
            else:
                print(f"\n🧮 Wolfram Validation: Error or unavailable")
        
        print(f"\n⏱️ Total generation time: {generation_time:.3f} seconds")
    
    def compare_neural_vs_database(self, prompt: str, genre: str = "Pop"):
        """Compare neural generation vs database lookup for the same prompt"""
        print(f"\n🔬 === Neural vs Database Comparison ===")
        print(f"🎯 Prompt: '{prompt}'")
        print(f"🎸 Genre: {genre}")
        
        # Force database mode
        original_neural_state = self.model.use_neural_generation
        self.model.disable_neural_generation()
        
        print(f"\n📚 Database Mode Result:")
        db_result = self.model.generate_from_prompt(prompt, genre, 1)[0]
        print(f"   🎹 Chords: {' → '.join(db_result['chords'])}")
        print(f"   🎼 Mode: {db_result['primary_mode']}")
        print(f"   💭 Top Emotion: {max(db_result['emotion_weights'], key=db_result['emotion_weights'].get)}")
        
        # Try neural mode (if available)
        if self.model.is_trained:
            self.model.enable_neural_generation()
            print(f"\n🧠 Neural Mode Result:")
            neural_result = self.model.generate_from_prompt(prompt, genre, 1)[0]
            print(f"   🎹 Chords: {' → '.join(neural_result['chords'])}")
            print(f"   🎼 Mode: {neural_result['primary_mode']}")
            print(f"   💭 Top Emotion: {max(neural_result['emotion_weights'], key=neural_result['emotion_weights'].get)}")
            
            # Compare differences
            if db_result['chords'] != neural_result['chords']:
                print(f"\n🔍 Differences Detected:")
                print(f"   📚 Database progression uses established patterns")
                print(f"   🧠 Neural progression shows learned creativity")
            else:
                print(f"\n✅ Both methods generated similar progressions")
        else:
            print(f"\n🧠 Neural Mode: Not available (model not trained)")
        
        # Restore original state
        if original_neural_state:
            self.model.enable_neural_generation()
        else:
            self.model.disable_neural_generation()
    
    def genre_analysis_demo(self, prompt: str):
        """Demonstrate how different genres affect the same emotional prompt"""
        print(f"\n🌍 === Genre Analysis Demo ===")
        print(f"🎯 Prompt: '{prompt}'")
        
        genres_to_test = ["Pop", "Jazz", "Classical", "Rock", "Electronic", "Folk"]
        
        for genre in genres_to_test:
            result = self.model.generate_from_prompt(prompt, genre, 1)[0]
            top_emotion = max(result['emotion_weights'], key=result['emotion_weights'].get)
            progression = ' → '.join(result['chords'])
            
            print(f"   🎸 {genre:12} | {result['primary_mode']:15} | {progression}")
        
        print(f"\n💡 Notice how genre preferences influence chord selection and mode interpretation!")
    
    def interactive_mode(self):
        """Enhanced interactive mode with all features"""
        print(f"\n🎮 === Enhanced Interactive Mode ===")
        print(f"Available commands:")
        print(f"  • Enter any emotional prompt to generate progressions")
        print(f"  • 'compare <prompt>' - Compare neural vs database modes")
        print(f"  • 'genres <prompt>' - Test prompt across different genres")
        print(f"  • 'wolfram on/off' - Toggle Wolfram validation")
        print(f"  • 'neural on/off' - Toggle neural generation (if trained)")
        print(f"  • 'help' - Show this help")
        print(f"  • 'quit' - Exit")
        print(f"\n" + "="*50)
        
        while True:
            try:
                user_input = input("\n🎯 Enter command or prompt: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print(f"\nAvailable commands:")
                    print(f"  • <prompt> - Generate progressions from emotion")
                    print(f"  • compare <prompt> - Neural vs database comparison")
                    print(f"  • genres <prompt> - Multi-genre analysis")
                    print(f"  • status - Show system status")
                    continue
                
                elif user_input.lower().startswith('compare '):
                    prompt = user_input[8:].strip()
                    if prompt:
                        self.compare_neural_vs_database(prompt)
                    continue
                
                elif user_input.lower().startswith('genres '):
                    prompt = user_input[7:].strip()
                    if prompt:
                        self.genre_analysis_demo(prompt)
                    continue
                
                elif user_input.lower() == 'status':
                    print(f"\n📊 System Status:")
                    print(f"   🧠 Neural Generation: {'✅ Enabled' if self.model.use_neural_generation else '❌ Disabled'}")
                    print(f"   📚 Model Training: {'✅ Trained' if self.model.is_trained else '❌ Untrained'}")
                    print(f"   🧮 Wolfram Validation: {'✅ Enabled' if self.wolfram_validator and self.wolfram_validator.enabled else '❌ Disabled'}")
                    continue
                
                elif user_input.lower() == 'wolfram off':
                    self.wolfram_validator = None
                    print("🧮 Wolfram validation disabled")
                    continue
                
                elif user_input.lower() == 'neural off':
                    self.model.disable_neural_generation()
                    continue
                
                elif user_input.lower() == 'neural on':
                    if self.model.is_trained:
                        self.model.enable_neural_generation()
                    else:
                        print("❌ Cannot enable neural mode: model not trained")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate progression for the prompt
                results = self.generate_with_full_analysis(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main demo function"""
    print("🎵 Enhanced VirtualAssistance Music Generation Demo")
    print("This demo showcases all the fixes from the audit:")
    print("✅ Fixed model loading bug")
    print("✅ Connected neural pipeline")
    print("✅ Fixed genre mapping")
    print("✅ Added Wolfram validation")
    print("✅ PyTorch stochasticity for creativity")
    print()
    
    # Initialize demo
    demo = EnhancedMusicDemo(
        model_path="chord_progression_model.pth",  # Try to load trained model
        enable_wolfram=True
    )
    
    # Quick demonstration
    test_prompts = [
        "romantic but anxious",
        "dark and mysterious", 
        "joyful celebration",
        "melancholy nostalgia"
    ]
    
    print("🎭 Quick Demo - Testing Various Emotions:")
    for prompt in test_prompts:
        print(f"\n" + "="*60)
        demo.generate_with_full_analysis(prompt, genre="Pop", num_progressions=1)
    
    # Show genre comparison
    print(f"\n" + "="*60)
    demo.genre_analysis_demo("romantic and dreamy")
    
    # Show neural vs database if available
    if demo.model.is_trained:
        print(f"\n" + "="*60)
        demo.compare_neural_vs_database("energetic and happy")
    
    # Enter interactive mode
    demo.interactive_mode()


if __name__ == "__main__":
    main() 