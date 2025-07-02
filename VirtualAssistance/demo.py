"""
Demo Script for Chord Progression Generation from Natural Language
Showcases the complete pipeline: Text → Emotions → Modes → Chord Progressions
"""

import torch
import json
from chord_progression_model import ChordProgressionModel
from train_model import ModelTrainer
import time


class ChordProgressionDemo:
    """Interactive demo for the chord progression generation system"""
    
    def __init__(self, model_path: str = None):
        print("🎵 Initializing Chord Progression Generator...")
        self.model = ChordProgressionModel()
        
        if model_path and torch.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            trainer = ModelTrainer(self.model)
            trainer.load_model(model_path)
        else:
            print("Using untrained model (database lookup mode)")
        
        print("✅ Model ready!\n")
    
    def generate_single(self, prompt: str, genre: str = "Pop", num_progressions: int = 1, 
                       show_details: bool = True) -> List[Dict]:
        """Generate chord progressions from a single prompt"""
        print(f"🎯 Prompt: '{prompt}'")
        print(f"🎸 Genre: {genre}")
        print(f"🔢 Generating {num_progressions} progression(s)...\n")
        
        start_time = time.time()
        results = self.model.generate_from_prompt(prompt, genre, num_progressions)
        generation_time = time.time() - start_time
        
        for i, result in enumerate(results, 1):
            print(f"--- Progression {i} ---")
            
            if show_details:
                # Show emotion analysis
                print("💭 Emotion Analysis:")
                top_emotions = sorted(result['emotion_weights'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                for emotion, weight in top_emotions:
                    if weight > 0.05:  # Only show significant emotions
                        print(f"   {emotion}: {weight:.3f}")
                
                # Show mode analysis
                print(f"\n🎼 Musical Mode: {result['primary_mode']}")
                if len([m for m, w in result['mode_blend'].items() if w > 0.1]) > 1:
                    print("🌊 Mode Blend:")
                    for mode, weight in result['mode_blend'].items():
                        if weight > 0.1:
                            print(f"   {mode}: {weight:.3f}")
            
            # Show chord progression
            print(f"\n🎹 Chord Progression: {' → '.join(result['chords'])}")
            print(f"🎵 Genre Match: {genre}")
            print()
        
        print(f"⏱️  Generated in {generation_time:.3f} seconds")
        return results
    
    def generate_batch(self, prompts: List[str], genre: str = "Pop") -> Dict[str, List[Dict]]:
        """Generate progressions for multiple prompts"""
        print(f"🔄 Batch processing {len(prompts)} prompts...")
        
        all_results = {}
        for prompt in prompts:
            results = self.model.generate_from_prompt(prompt, genre, 1)
            all_results[prompt] = results
        
        return all_results
    
    def interactive_mode(self):
        """Interactive CLI for generating progressions"""
        print("🎮 Interactive Mode - Enter prompts to generate chord progressions")
        print("Commands: 'quit' to exit, 'help' for options, 'batch' for batch mode\n")
        
        while True:
            try:
                user_input = input("🎯 Enter prompt (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'batch':
                    self._batch_mode()
                    continue
                
                elif user_input.lower() == 'examples':
                    self._show_examples()
                    continue
                
                elif user_input.lower().startswith('genre:'):
                    # Change default genre
                    genre = user_input.split(':', 1)[1].strip()
                    print(f"🎸 Default genre set to: {genre}")
                    continue
                
                elif not user_input:
                    continue
                
                # Generate progression
                self.generate_single(user_input, show_details=True)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("""
🆘 Help - Available Commands:
   
   Basic Usage:
   • Type any emotional description to generate chords
   • Examples: "happy and excited", "sad but hopeful", "angry"
   
   Commands:
   • help         - Show this help
   • examples     - Show example prompts
   • batch        - Enter batch mode for multiple prompts
   • genre:NAME   - Set default genre (e.g., "genre:Jazz")
   • quit/exit/q  - Exit the program
   
   Supported Genres:
   Pop, Rock, Jazz, Classical, Electronic, R&B, Soul, Country, Folk,
   Blues, Funk, Metal, Indie, Ambient, Cinematic, and more...
        """)
    
    def _show_examples(self):
        """Show example prompts"""
        examples = [
            ("happy and excited", "Bright, uplifting progressions"),
            ("romantic but a little anxious", "Love with subtle tension"),
            ("dark and mysterious", "Minor modes with unusual chords"),
            ("angry and aggressive", "Power chords and dissonance"),
            ("sad but beautiful", "Melancholic but melodic"),
            ("dreamy and floating", "Lydian modes and ambient textures"),
            ("confident and strong", "Stable progressions with resolve"),
            ("nostalgic and warm", "Mixolydian warmth and memory"),
            ("tense and building", "Anticipation and forward motion"),
            ("otherworldly and sublime", "Transcendent and cosmic")
        ]
        
        print("\n📝 Example Prompts:")
        for prompt, description in examples:
            print(f"   '{prompt}' - {description}")
        print()
    
    def _batch_mode(self):
        """Batch processing mode"""
        print("\n🔄 Batch Mode - Enter multiple prompts (empty line to finish)")
        prompts = []
        
        while True:
            prompt = input(f"Prompt {len(prompts)+1}: ").strip()
            if not prompt:
                break
            prompts.append(prompt)
        
        if not prompts:
            print("No prompts entered.")
            return
        
        genre = input("Genre (default: Pop): ").strip() or "Pop"
        
        results = self.generate_batch(prompts, genre)
        
        print(f"\n📊 Batch Results Summary:")
        for prompt, result_list in results.items():
            result = result_list[0]
            top_emotion = max(result['emotion_weights'], key=result['emotion_weights'].get)
            progression = ' → '.join(result['chords'])
            print(f"   '{prompt}' → {top_emotion} → {progression}")
    
    def export_session(self, results: List[Dict], filename: str = None):
        """Export results to .jam session file"""
        if filename is None:
            filename = f"session_{int(time.time())}.jam"
        
        session = self.model.export_jam_session(results)
        
        with open(filename, 'w') as f:
            json.dump(session, f, indent=2)
        
        print(f"💾 Session exported to {filename}")
        return filename
    
    def compare_genres(self, prompt: str, genres: List[str]):
        """Compare how different genres affect the same prompt"""
        print(f"🎯 Comparing genres for prompt: '{prompt}'\n")
        
        for genre in genres:
            print(f"🎸 Genre: {genre}")
            results = self.model.generate_from_prompt(prompt, genre, 1)
            result = results[0]
            progression = ' → '.join(result['chords'])
            print(f"   Progression: {progression}")
            print()
    
    def emotion_analysis_demo(self):
        """Demonstrate emotion parsing capabilities"""
        print("🧠 Emotion Analysis Demo\n")
        
        test_phrases = [
            "happy",
            "very sad", 
            "angry but hopeful",
            "romantic and dreamy",
            "anxious yet excited",
            "deeply melancholic",
            "triumphant and proud",
            "mysterious and dark",
            "peaceful and serene",
            "chaotic and intense"
        ]
        
        for phrase in test_phrases:
            emotion_weights = self.model.emotion_parser.parse_emotion_weights(phrase)
            top_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"'{phrase}' →", end=" ")
            for emotion, weight in top_emotions:
                if weight > 0.1:
                    print(f"{emotion}({weight:.2f})", end=" ")
            print()


def main():
    """Main demo function"""
    print("🎼 Chord Progression Generator - Natural Language to Music")
    print("=" * 60)
    
    # Initialize demo
    demo = ChordProgressionDemo()
    
    # Show some quick examples
    print("🚀 Quick Demo:")
    demo_prompts = [
        "happy and excited",
        "sad but beautiful", 
        "dark and mysterious",
        "romantic and warm"
    ]
    
    for prompt in demo_prompts:
        demo.generate_single(prompt, show_details=False)
        print()
    
    # Genre comparison demo
    print("\n🎸 Genre Comparison Demo:")
    demo.compare_genres("romantic and nostalgic", ["Pop", "Jazz", "Country", "Classical"])
    
    # Emotion analysis demo
    demo.emotion_analysis_demo()
    
    # Enter interactive mode
    print("\n" + "=" * 60)
    demo.interactive_mode()


if __name__ == "__main__":
    main()
