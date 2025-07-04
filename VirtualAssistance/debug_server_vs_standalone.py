#!/usr/bin/env python3
"""
Debug the difference between standalone test and server
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chord_progression_model import ChordProgressionModel

def test_server_vs_standalone():
    """Compare server behavior with standalone behavior"""
    print("🔍 Testing Standalone vs Server Behavior...")
    
    # Test same code as the server would use
    model = ChordProgressionModel()
    
    test_phrases = [
        "cosmic unity transcendence",
        "spiritual transcendence and divine awakening", 
        "transcendence",
        "divine enlightenment",
        "cosmic unity",
        "mystic insight"
    ]
    
    for phrase in test_phrases:
        print(f"\n--- Testing: '{phrase}' ---")
        
        # Parse emotion weights the same way the server does
        emotion_weights = model.emotion_parser.parse_emotion_weights(phrase)
        
        transcendence_weight = emotion_weights.get('Transcendence', 0)
        reverence_weight = emotion_weights.get('Reverence', 0)
        top_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"Transcendence: {transcendence_weight:.3f}")
        print(f"Reverence: {reverence_weight:.3f}")
        print(f"Top 3: {[(e, round(w, 3)) for e, w in top_emotions if w > 0]}")
        
        # Check if sub-emotion was detected
        sub_emotion = ""
        if hasattr(model.emotion_parser, 'get_detected_sub_emotion'):
            sub_emotion = model.emotion_parser.get_detected_sub_emotion()
        print(f"Sub-emotion: {sub_emotion}")

if __name__ == "__main__":
    test_server_vs_standalone()
