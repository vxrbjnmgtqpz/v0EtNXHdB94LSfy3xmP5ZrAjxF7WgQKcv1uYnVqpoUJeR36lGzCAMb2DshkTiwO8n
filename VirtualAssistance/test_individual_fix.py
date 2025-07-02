#!/usr/bin/env python3
"""
Test script to verify individual chord emotions fix
"""

import requests
import json
import time

# Test the individual chord emotions feature
def test_individual_chord_emotions():
    base_url = "http://localhost:5002"
    
    print("Testing Individual Chord Emotions Fix...")
    print("=" * 50)
    
    # Create a session to maintain conversation context
    session = requests.Session()
    
    # Step 1: Generate a chord progression
    print("Step 1: Generating a chord progression...")
    progression_request = {
        "message": "Create a happy chord progression"
    }
    
    try:
        response = session.post(f"{base_url}/chat/integrated", 
                               json=progression_request, 
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Progression generated: {data.get('chords', [])}")
            print(f"  Intent: {data.get('intent')}")
            print(f"  Models used: {data.get('models_used', [])}")
            print()
        else:
            print(f"❌ Failed to generate progression: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Error generating progression: {e}")
        return
    
    # Wait a moment to ensure session is stored
    time.sleep(0.5)
    
    # Step 2: Request individual chord emotions
    print("Step 2: Requesting individual chord emotions...")
    individual_request = {
        "message": "Show me individual chord emotions"
    }
    
    try:
        response = session.post(f"{base_url}/chat/integrated", 
                               json=individual_request, 
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Individual analysis received")
            print(f"  Intent: {data.get('intent')}")
            print(f"  Models used: {data.get('models_used', [])}")
            print(f"  Progression breakdown: {data.get('progression_breakdown', False)}")
            
            # Check if we have individual analysis
            individual_analysis = data.get('individual_analysis', [])
            chords = data.get('chords', [])
            
            print(f"  Number of chords analyzed: {len(individual_analysis)}")
            print(f"  Chord progression: {chords}")
            print()
            
            # Display analysis for each chord
            if individual_analysis and len(individual_analysis) > 0:
                print("Individual Chord Analysis:")
                print("-" * 30)
                for i, (chord, analysis) in enumerate(zip(chords, individual_analysis)):
                    print(f"{i+1}. {chord}")
                    if analysis.get('emotion_weights'):
                        emotions = analysis['emotion_weights']
                        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        emotion_text = ", ".join([f"{k} ({v:.2f})" for k, v in top_emotions if v > 0])
                        print(f"   Emotions: {emotion_text}")
                    
                    if analysis.get('mode_context') or analysis.get('style_context'):
                        mode = analysis.get('mode_context', 'N/A')
                        style = analysis.get('style_context', 'N/A')
                        print(f"   Context: {mode}/{style}")
                    
                    if analysis.get('emotional_score') is not None:
                        print(f"   Fit: {analysis['emotional_score']:.2f}")
                    
                    if analysis.get('error'):
                        print(f"   Error: {analysis['error']}")
                    print()
                
                if len(individual_analysis) >= len(chords):
                    print("✅ SUCCESS: All chords in the progression were analyzed individually!")
                else:
                    print(f"⚠️  WARNING: Only {len(individual_analysis)} out of {len(chords)} chords were analyzed")
            else:
                print("❌ FAILURE: No individual analysis data received")
                
        else:
            print(f"❌ Failed to get individual analysis: {response.status_code}")
            if response.text:
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error requesting individual analysis: {e}")

if __name__ == "__main__":
    test_individual_chord_emotions()
