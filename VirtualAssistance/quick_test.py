#!/usr/bin/env python3
"""
Quick chord progression tester - just enter emotions, get progressions
"""

from chord_progression_model import ChordProgressionModel

def quick_test():
    """Simple emotion-to-chord testing"""
    
    print("=" * 50)
    print("QUICK CHORD PROGRESSION TESTER")
    print("Enter emotions, get Roman numeral progressions")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    model = ChordProgressionModel()
    print("✓ Model loaded!")
    
    while True:
        try:
            prompt = input("\nEmotion prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q', '']:
                print("Goodbye!")
                break
            
            # Generate with Pop as default genre
            results = model.generate_from_prompt(prompt, genre_preference="Pop", num_progressions=1)
            result = results[0]
            
            # Show just the essentials
            top_emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"🎭 Emotions: {[(k, round(v, 3)) for k, v in top_emotions]}")
            print(f"🎵 Mode: {result['primary_mode']}")  
            print(f"🎼 Chords: {' → '.join(result['chords'])}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
