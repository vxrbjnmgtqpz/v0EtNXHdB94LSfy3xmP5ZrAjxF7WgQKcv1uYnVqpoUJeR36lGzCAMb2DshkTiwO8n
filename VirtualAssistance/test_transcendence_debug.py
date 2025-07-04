#!/usr/bin/env python3
"""
Test Transcendence Database Integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chord_progression_model import ChordProgressionModel

def test_transcendence_loading():
    """Test if Transcendence emotion is properly loaded"""
    print("🔍 Testing Transcendence Database Loading...")
    
    model = ChordProgressionModel()
    
    # Check if Transcendence is in emotion labels
    print(f"Available emotions: {model.emotion_parser.emotion_labels}")
    print(f"Transcendence in emotion_labels: {'Transcendence' in model.emotion_parser.emotion_labels}")
    
    # Check database loading
    try:
        # The database is in model.database.emotion_progressions
        if hasattr(model.database, 'emotion_progressions'):
            emotion_pools = model.database.emotion_progressions
            print(f"Loaded emotion pools: {list(emotion_pools.keys())}")
            print(f"Transcendence in pools: {'Transcendence' in emotion_pools}")
            
            if 'Transcendence' in emotion_pools:
                transcendence_progressions = emotion_pools['Transcendence']
                print(f"Transcendence progressions loaded: {len(transcendence_progressions)}")
                if transcendence_progressions:
                    print(f"First progression: {transcendence_progressions[0].chords}")
                    print(f"First emotion path: {transcendence_progressions[0].emotion}")
        else:
            print("No emotion_progressions attribute found")
    except Exception as e:
        print(f"Error checking emotion pools: {e}")
    
    # Test progression generation with direct Transcendence request
    print("\n🎵 Testing Transcendence Progression Generation...")
    try:
        result = model.generate_from_prompt("Transcendence", genre_preference="Classical", num_progressions=1)
        if result:
            progression = result[0]
            print(f"Generated progression: {progression['chords']}")
            print(f"Emotion weights: {progression['emotion_weights']}")
            print(f"Detected sub-emotion: {progression.get('detected_sub_emotion', 'None')}")
        
        # Test with sub-emotion
        result2 = model.generate_from_prompt("Cosmic Unity transcendence", genre_preference="Classical", num_progressions=1)
        if result2:
            progression2 = result2[0]
            print(f"Sub-emotion progression: {progression2['chords']}")
            print(f"Sub-emotion weights: {progression2['emotion_weights']}")
        
    except Exception as e:
        print(f"Error generating progression: {e}")
    
    # Test emotion detection
    print("\n🧠 Testing Emotion Detection...")
    try:
        emotion_weights = model.emotion_parser.parse_emotion_weights("spiritual transcendence and divine awakening")
        print(f"Detected emotion weights: {emotion_weights}")
        transcendence_weight = emotion_weights.get('Transcendence', 0)
        print(f"Transcendence weight: {transcendence_weight}")
    except Exception as e:
        print(f"Error detecting emotion: {e}")

if __name__ == "__main__":
    test_transcendence_loading()
