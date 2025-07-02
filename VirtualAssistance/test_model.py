#!/usr/bin/env python3
"""
Simple test script for the Chord Progression Model
Generate Roman numeral chord progressions from emotional prompts
"""

from chord_progression_model import ChordProgressionModel

def test_chord_generation():
    """Test the chord progression generation with various prompts"""
    
    print("=" * 60)
    print("CHORD PROGRESSION GENERATOR")
    print("Generate Roman numeral progressions from emotional prompts")
    print("=" * 60)
    
    # Initialize the model
    print("\nInitializing model...")
    model = ChordProgressionModel()
    print("✓ Model loaded successfully!")
    
    # Test prompts with expected emotions
    test_prompts = [
        ("I feel joyful and uplifted", "Joy"),
        ("I am deeply sad and sorrowful", "Sadness"), 
        ("I feel angry and aggressive", "Anger"),
        ("I am scared and anxious", "Fear"),
        ("I feel romantic and loving", "Love"),
        ("I am disgusted and repulsed", "Disgust"),
        ("I feel surprised and amazed", "Surprise"),
        ("I trust and feel secure", "Trust"),
        ("I anticipate something great", "Anticipation"),
        ("I feel guilty and ashamed", "Shame"),
        ("I am envious and jealous", "Envy"),
        ("I feel awe and wonder", "Aesthetic Awe")
    ]
    
    print(f"\nTesting {len(test_prompts)} emotional prompts:")
    print("-" * 60)
    
    for prompt, expected_emotion in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Expected emotion: {expected_emotion}")
        
        # Generate progression
        results = model.generate_from_prompt(prompt, genre_preference="Pop", num_progressions=1)
        result = results[0]
        
        # Show results
        top_emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"Detected emotions: {[(k, round(v, 3)) for k, v in top_emotions]}")
        print(f"Primary mode: {result['primary_mode']}")
        print(f"Chord progression: {' → '.join(result['chords'])}")

def interactive_mode():
    """Interactive mode - enter your own prompts"""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Enter emotional prompts to generate chord progressions")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    model = ChordProgressionModel()
    
    while True:
        try:
            prompt = input("\nEnter your emotional prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not prompt:
                continue
                
            # Optional genre selection
            print("Available genres: Pop, Rock, Jazz, Classical, Metal, Folk, EDM")
            genre = input("Genre preference [Pop]: ").strip()
            if not genre or genre.lower() in ['default', 'pop']:
                genre = "Pop"
            else:
                # Capitalize first letter for consistency
                genre = genre.capitalize()
                
            # Generate progression
            results = model.generate_from_prompt(prompt, genre_preference=genre, num_progressions=1)
            result = results[0]
            
            # Display results
            print(f"\n--- Results ---")
            top_emotions = sorted(result['emotion_weights'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top emotions: {[(k, round(v, 3)) for k, v in top_emotions]}")
            print(f"Primary mode: {result['primary_mode']}")
            print(f"Chord progression: {' → '.join(result['chords'])}")
            print(f"Genre: {genre}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_chord_generation()
        
        # Ask if user wants interactive mode
        response = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode()
