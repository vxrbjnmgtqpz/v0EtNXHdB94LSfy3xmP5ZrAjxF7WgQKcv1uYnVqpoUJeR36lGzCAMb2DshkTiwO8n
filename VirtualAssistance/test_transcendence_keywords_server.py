#!/usr/bin/env python3
"""
Test all Transcendence keywords against both server and standalone
"""

import requests
import json
import time

def test_server_transcendence_keywords():
    """Test Transcendence keywords against the running server"""
    
    # Keywords that should trigger Transcendence
    test_keywords = [
        "transcendence",
        "cosmic unity transcendence", 
        "divine enlightenment",
        "mystic insight",
        "spiritual awakening",
        "cosmic unity",
        "celestial ascension",
        "divine ecstasy",
        "ego death",
        "lucid wonder",
        "sacred dissonance",
        "mirror self",
        "serene void",
        "ethereal calm",
        "kaleidoscopic resonance",
        "epiphany",
        "inner rebirth",
        "hypnotic trance",
        "arcane mystery",
        "psychedelic spiral",
        "overlapping realities",
        "sublime vastness",
        "harmonic nirvana"
    ]
    
    working_keywords = []
    broken_keywords = []
    
    print("🔍 Testing Transcendence Keywords Against Server...")
    print("=" * 60)
    
    for keyword in test_keywords:
        try:
            response = requests.post(
                "http://localhost:5004/chat/integrated",
                json={"message": keyword},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                transcendence_weight = data.get('emotion', {}).get('Transcendence', 0)
                
                # Check emotion distribution
                emotions = data.get('emotion', {})
                non_zero_emotions = {k: v for k, v in emotions.items() if v > 0}
                
                if transcendence_weight > 0:
                    working_keywords.append((keyword, transcendence_weight, non_zero_emotions))
                    print(f"✅ '{keyword}': Transcendence = {transcendence_weight:.3f}")
                else:
                    broken_keywords.append((keyword, non_zero_emotions))
                    print(f"❌ '{keyword}': No transcendence detected. Got: {non_zero_emotions}")
            else:
                print(f"💥 '{keyword}': Server error {response.status_code}")
                broken_keywords.append((keyword, f"Server error {response.status_code}"))
                
        except Exception as e:
            print(f"💥 '{keyword}': Exception - {e}")
            broken_keywords.append((keyword, f"Exception: {e}"))
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Working keywords: {len(working_keywords)}/{len(test_keywords)}")
    print(f"❌ Broken keywords: {len(broken_keywords)}/{len(test_keywords)}")
    
    if working_keywords:
        print(f"\n🎯 WORKING TRANSCENDENCE KEYWORDS:")
        for keyword, weight, emotions in working_keywords:
            print(f"  • '{keyword}' → {weight:.3f}")
    
    if broken_keywords:
        print(f"\n🚫 BROKEN KEYWORDS:")
        for keyword, emotions in broken_keywords:
            if isinstance(emotions, dict):
                top_emotion = max(emotions.items(), key=lambda x: x[1]) if emotions else ("None", 0)
                print(f"  • '{keyword}' → {top_emotion[0]} ({top_emotion[1]:.3f})")
            else:
                print(f"  • '{keyword}' → {emotions}")
    
    # Test sub-emotion detection for working keywords
    if working_keywords:
        print(f"\n🔍 SUB-EMOTION DETECTION TEST:")
        for keyword, weight, emotions in working_keywords[:3]:  # Test first 3 working keywords
            try:
                response = requests.post(
                    "http://localhost:5004/chat/integrated",
                    json={"message": keyword},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    sub_emotion = data.get('primary_result', {}).get('detected_sub_emotion', '')
                    print(f"  • '{keyword}' → Sub-emotion: '{sub_emotion}'")
                    
            except Exception as e:
                print(f"  • '{keyword}' → Sub-emotion test failed: {e}")
            
            time.sleep(0.1)

if __name__ == "__main__":
    test_server_transcendence_keywords()
